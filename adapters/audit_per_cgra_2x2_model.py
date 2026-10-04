#!/usr/bin/env python3
"""Replay a sealed native collection; materialize pre-mapper development data.

Mapped artifacts are checked by the collection validator solely as ground truth.
Features always come from its pre-mapper graph cache. The requested baseline
report includes known test labels. Training uses a separate development cache;
alternative model predictions on the test cache require frozen selection.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / 'src', ROOT / 'adapters'):
    sys.path.insert(0, str(directory))
import train_per_cgra_2x2_model as reference
from amoeba_cost_catalog import load_mapper_model, sha256_file
from cgra_ii_predictor.mapper_model import mapper_feature_vector


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def source_kind(families):
    return 'random' if families and all(x.startswith('random-dfg/') for x in families) else 'program'


def motif(families):
    text = ' '.join(families).lower()
    for name, tokens in (
        ('attention', ('attention', 'qwen')), ('convolution', ('conv', 'shufflenet')),
        ('matrix-vector', ('gemv', 'spmv', 'matvec', 'bicg')), ('matrix-matrix', ('gemm', 'matmul')),
        ('stencil', ('stencil', 'jacobi', 'heat')), ('reduction', ('reduce', 'pool', 'histogram', 'statistics')),
        ('filter', ('fir', 'iir', 'dspstone')), ('graph', ('gcn', 'gnnbuilder')),
    ):
        if any(token in text for token in tokens):
            return name
    return 'random' if source_kind(families) == 'random' else 'other-program'


def protected_snapshot(output):
    records = {}
    for checkout in (ROOT, Path('/home/x/shiran/project/cgra-ii-predictor')):
        for directory in (checkout / 'models/final', checkout / 'models/candidates/per-cgra-2x2'):
            if directory.exists():
                for path in sorted(directory.rglob('*')):
                    if path.is_file():
                        records[str(path)] = sha256_file(path)
    write_json(output / 'protected-files.json', records)
    return records


def feature_audit(rows):
    train = [r for r in rows if r['split'] == 'train' and r.get('full_features') is not None
             and r['native_status'] == 'success' and r['true_ii'] is not None]
    names = reference.MAPPER_FEATURE_NAMES_2X2
    raw = torch.tensor([r['full_features'] for r in train], dtype=torch.float32)
    selected = [names.index(n) for n in reference.COMPACT_FEATURE_NAMES]
    low, high = raw.min(0).values, raw.max(0).values
    std = raw.std(0, unbiased=False)
    standardized = (raw - raw.mean(0)) / torch.where(std < 1e-5, torch.ones_like(std), std)
    corr = standardized.T @ standardized / len(raw)
    duplicate = defaultdict(list)
    eligible = [r for r in rows if r.get('full_features') is not None and r['true_ii'] is not None]
    for row in eligible:
        actual_input = torch.tensor(row['full_features'], dtype=torch.float32).tolist()
        signature = tuple(actual_input[i] for i in selected)
        duplicate[signature].append(row)
    collisions = []
    for entries in duplicate.values():
        if len({r['true_ii'] for r in entries}) > 1:
            collisions.append([{k: r[k] for k in ('identity', 'source_group', 'split', 'shape', 'true_ii')} for r in entries])
    # Development-only nearest neighbours avoid using test labels to choose masks.
    dev = [r for r in eligible if r['split'] in ('train', 'validation')]
    matrix = torch.tensor([r['full_features'] for r in dev])[:, selected]
    matrix = (matrix - raw.mean(0)[selected]) / torch.where(std[selected] < 1e-5, torch.ones_like(std[selected]), std[selected])
    near = []
    for start in range(0, len(dev), 128):
        distances = torch.cdist(matrix[start:start+128], matrix) / len(selected) ** .5
        for offset, index in enumerate(range(start, min(start+128, len(dev)))):
            for other in torch.where(distances[offset] <= .05)[0].tolist():
                if other <= index or dev[index]['source_group'] == dev[other]['source_group']:
                    continue
                if dev[index]['true_ii'] != dev[other]['true_ii']:
                    near.append({'left_identity': dev[index]['identity'], 'right_identity': dev[other]['identity'],
                                 'left_shape': dev[index]['shape'], 'right_shape': dev[other]['shape'],
                                 'left_ii': dev[index]['true_ii'], 'right_ii': dev[other]['true_ii'],
                                 'standardized_rms_distance': float(distances[offset, other])})
    ood = {}
    for split in ('validation', 'test'):
        subset = [r for r in eligible if r['split'] == split]
        values = torch.tensor([r['full_features'] for r in subset])
        outside = (values < low) | (values > high)
        ood[split] = {'row_count': len(subset), 'compact_rows_outside_range': int(outside[:, selected].any(1).sum()),
                      'by_feature': {n: int(outside[:, i].sum()) for i, n in enumerate(names) if outside[:, i].any()}}
    weights = reference._balanced_group_weights([{
        'group': r['source_group'], 'group_families': r['source_families']} for r in train])
    mass = {kind: float(weights[[i for i, r in enumerate(train) if r['source_kind'] == kind]].sum()) for kind in ('random', 'program')}
    group_mass = defaultdict(float)
    for row, weight in zip(train, weights.tolist()):
        group_mass[row['source_group']] += weight
    return {'constant_features': [n for n, s in zip(names, std) if s == 0],
            'near_constant_features_std_below_1e-5': [n for n, s in zip(names, std) if s < 1e-5],
            'absolute_correlation_above_0_999': [[names[i], names[j], float(corr[i,j])] for i in selected for j in selected if j > i and abs(corr[i,j]) > .999],
            'exact_compact_different_ii_collisions': collisions, 'development_near_collision_threshold': .05,
            'development_near_collisions': near, 'range_ood': ood,
            'normalization': 'unweighted successful training-row population mean/std',
            'loss_weighting_corrected_description': 'Random and program strata each receive 50% total mass when both exist; groups are uniform within each stratum; successful rows are uniform within group.',
            'effective_loss_mass': mass, 'effective_group_mass': dict(group_mass),
            'feature_limits': ['summaries discard graph identity and full adjacency', 'compact mask omits all opcode counts and node-attribute maxima',
                               'shape one-hot omitted; rows/columns/capacity remain', 'no measured native operation latency is invented']}


def development_collision_audit(rows):
    """Distinguish discarded-feature collisions from full-summary collisions."""
    names = reference.MAPPER_FEATURE_NAMES_2X2
    indices = [names.index(n) for n in reference.COMPACT_FEATURE_NAMES]
    buckets = defaultdict(list)
    for row in rows:
        if row['split'] not in ('train', 'validation') or row['true_ii'] is None or row.get('full_features') is None:
            continue
        values = torch.tensor(row['full_features'], dtype=torch.float32).tolist()
        buckets[tuple(values[i] for i in indices)].append(dict(row, full_features=values))
    collisions = []
    for entries in buckets.values():
        if len({r['true_ii'] for r in entries}) < 2:
            continue
        collisions.append({
            'different_full_feature_vectors': len({tuple(r['full_features']) for r in entries}),
            'varying_omitted_features': [name for i,name in enumerate(names) if len({r['full_features'][i] for r in entries}) > 1],
            'rows': [{k:r[k] for k in ('identity','source_group','source_families','shape','true_ii','split')} for r in entries],
        })
    return {'development_only': True, 'comparison_precision': 'float32 model inputs',
            'collision_buckets': len(collisions),
            'identities_in_collisions': len({r['identity'] for c in collisions for r in c['rows']}),
            'buckets_distinguished_by_full148': sum(c['different_full_feature_vectors'] > 1 for c in collisions),
            'buckets_still_identical_with_full148': sum(c['different_full_feature_vectors'] == 1 for c in collisions),
            'collisions': collisions}


def residual_audit(rows):
    buckets = defaultdict(list)
    for row in rows:
        if row['true_ii'] is None or row['scores'].get('ensemble') is None:
            continue
        for key in ('all', 'split:' + row['split'], 'kind:' + row['source_kind'],
                    'shape:' + 'x'.join(map(str, row['shape'])), 'family:' + '|'.join(row['source_families'])):
            buckets[key].append(row)
    result = {}
    for key, entries in buckets.items():
        zero = [r for r in entries if r['true_ii'] == r['lower_bound']]
        result[key] = {'row_count': len(entries), 'residual_distribution': dict(Counter(str(r['true_ii']-r['lower_bound']) for r in entries)),
                       'zero_residual_fraction': len(zero)/len(entries),
                       'zero_residual_prediction_bias': sum(r['scores']['ensemble']-r['true_ii'] for r in zero)/len(zero) if zero else None,
                       'clamp20_fraction': sum(r['scores']['ensemble'] >= 20 for r in entries)/len(entries),
                       'member_clamp20_fractions': {str(seed): sum(r['scores']['member_'+str(seed)] >= 20 for r in entries)/len(entries)
                                                    for seed in reference.DEFAULT_SEEDS}}
    return result


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    protected_snapshot(output)
    collection, candidate = args.collection, args.candidate
    manifest, provenance, bundle, excluded, exclusions = reference._validate_protocol_manifest(
        collection, candidate / 'source-groups.json', candidate / 'training-exclusions.json')
    bundle['excluded_groups'] = excluded
    success, counts, artifacts = reference._load_outcomes(collection, manifest, provenance, bundle, require_complete=True)
    assignment = reference._split_groups(set(g['group_id'] for g in bundle['source_groups']['groups']) - excluded)
    model, config, metadata = load_mapper_model(candidate / 'mapper.pt', torch.device('cpu'))
    torch.set_num_threads(1)
    outcomes = json.loads((collection / 'outcomes.json').read_text())
    rows = []
    for outcome in outcomes['queries']:
        identity = outcome['mapper_input_identity']
        group = bundle['group_bindings'][identity]
        families = sorted(group['source_program_families'])
        result = json.loads((collection / outcome['result_path']).read_text())
        analysis = result.get('analysis', {})
        shape = [outcome['rows'], outcome['cols']]
        row = {'identity': identity, 'source_group': group['group_id'], 'source_families': families,
               'source_kind': source_kind(families), 'motif': motif(families),
               'split': assignment.get(group['group_id'], 'excluded'), 'shape': shape,
               'rec_mii': analysis.get('rec_mii'), 'res_mii': analysis.get('res_mii'),
               'lower_bound': analysis.get('lower_bound'), 'true_ii': outcome.get('compiled_ii'),
               'native_status': 'success' if outcome['status'] == 'success' else outcome.get('censor_reason', result.get('censor_reason', result.get('reason', 'invalid_output'))),
               'result_path': outcome['result_path'], 'result_sha256': outcome['result_sha256'],
               'scores': {'analytical': analysis.get('lower_bound')}}
        if analysis.get('status') == 'success' and 0 <= analysis['lower_bound'] <= 20:
            graph = bundle['graph_cache'][identity][1]
            features = mapper_feature_vector(graph, *shape, analysis['rec_mii'], analysis['res_mii'], analysis['lower_bound'], shape_protocol=reference.SHAPE_PROTOCOL_2X2_ID)
            row['full_features'] = list(features)
            x, lb = torch.tensor([features]), torch.tensor([float(analysis['lower_bound'])])
            with torch.inference_mode():
                members = [float(m(x, lb)) for m in model.members]
                row['scores'].update({'member_' + str(seed): v for seed, v in zip(reference.DEFAULT_SEEDS, members)})
                row['scores']['ensemble'] = float(model(x, lb))
                row['member_std'] = float(torch.tensor(members).std(unbiased=False))
        else:
            row['full_features'] = None
            row['scores'].update({name: None for name in ['ensemble'] + ['member_'+str(s) for s in reference.DEFAULT_SEEDS]})
        rows.append(row)
    write_json(output / 'provenance.json', {'architecture_sha256': provenance['architecture_sha256'],
               'binary_sha256': provenance['neura_opt_sha256'], 'candidate_sha256': sha256_file(candidate / 'mapper.pt'),
               'collection_provenance': provenance, 'exclusions': exclusions, 'source_artifacts': artifacts,
               'collection': str(collection), 'candidate': str(candidate), 'inference_uses_mapped_artifacts': False,
               'baseline_reference_commit': '3ade318', 'status_counts': counts})
    with (output / 'baseline-query-rows.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps({k:v for k,v in row.items() if k != 'full_features'}, sort_keys=True) + '\n')
    columns = ['identity', 'source_group', 'source_families', 'source_kind', 'motif', 'split', 'shape', 'rec_mii', 'res_mii', 'lower_bound', 'true_ii', 'native_status', 'ensemble'] + ['member_'+str(s) for s in reference.DEFAULT_SEEDS]
    with (output / 'baseline-query-rows.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        for row in rows:
            value = {k:row.get(k) for k in columns}; value.update({k:row['scores'].get(k) for k in columns if k in row['scores']}); writer.writerow(value)
    for splitset, filename in ((('train', 'validation'), 'development.pt'), (('test',), 'frozen-test.pt')):
        torch.save({'rows': [r for r in rows if r['split'] in splitset],
                    'training_rows': [r for r in success if assignment[r['group']] in splitset],
                    'assignment': {g:s for g,s in assignment.items() if s in splitset},
                    'provenance_sha256': provenance['provenance_sha256']}, output / filename)
    torch.save({'rows': [r for r in rows if r['split'] == 'excluded']}, output / 'evaluation-only.pt')
    write_json(output / 'cache-manifest.json', {
        'schema': 'cgra-ii-2x2-cache-seal-v1',
        'files': {name: sha256_file(output / name) for name in
                  ('development.pt', 'frozen-test.pt', 'evaluation-only.pt')},
    })
    write_json(output / 'feature-audit.json', feature_audit(rows))
    write_json(output / 'development-feature-collisions.json', development_collision_audit(rows))
    write_json(output / 'residual-audit.json', residual_audit(rows))
    write_json(output / 'baseline-reproduction.json', {split: reference._score(
        [r for r in success if assignment[r['group']] == split], reference._predict(model, [r for r in success if assignment[r['group']] == split])) for split in ('train','validation','test')})
    print(json.dumps({'output': str(output), 'rows': len(rows), 'baseline_reproduced': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, default=ROOT / 'models/candidates/per-cgra-2x2')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
