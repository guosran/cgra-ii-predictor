#!/usr/bin/env python3
"""Bounded matched-seed experiments, with nested grouped development folds.

The plan/fit commands only accept development.pt. Outer-fold labels never
select epochs. Final fit uses the original train/validation assignment. The
test command requires a frozen selection record and refuses repeated opening.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / 'src', ROOT / 'adapters'):
    sys.path.insert(0, str(directory))
import train_per_cgra_2x2_model as ref
from audit_per_cgra_2x2_model import write_json, source_kind
from amoeba_cost_catalog import sha256_file, load_mapper_model, ENSEMBLE_CHECKPOINT_SCHEMA
from cgra_ii_predictor.mapper_model import DirectMapperIIModel, DirectMapperIIEnsemble, MapperModelConfig

SEEDS = (17, 41, 113, 239)
CACHE_MANIFEST_SCHEMA = 'cgra-ii-2x2-cache-seal-v1'
CACHE_FILES = ('development.pt', 'frozen-test.pt', 'evaluation-only.pt')
ADDED = ('log_count_add','log_count_mul','log_count_div','log_count_fadd','log_count_fmul',
         'log_count_fneg','log_count_fcmp','log_count_sel','log_count_fmul_fadd','log_count_load','log_count_store')


def _sidecar(path):
    return path.with_name(path.name + '.sha256')


def _write_sealed_json(path, value):
    write_json(path, value)
    _sidecar(path).write_text(sha256_file(path) + '\n')


def _verify_sealed_json(path, label):
    sidecar = _sidecar(path)
    if not path.is_file() or not sidecar.is_file():
        raise ValueError(f'{label} or its SHA-256 seal is missing')
    if sidecar.read_text().strip() != sha256_file(path):
        raise ValueError(f'{label} changed after it was sealed')


def _is_sha256(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _load_cache_seal(development_path):
    if development_path.name != 'development.pt':
        raise ValueError('training accepts only the sealed development.pt cache')
    manifest_path = development_path.parent / 'cache-manifest.json'
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError('cache manifest is missing or invalid') from error
    if not isinstance(manifest, dict):
        raise ValueError('cache manifest is invalid')
    files = manifest.get('files')
    if manifest.get('schema') != CACHE_MANIFEST_SCHEMA or not isinstance(files, dict) or set(files) != set(CACHE_FILES):
        raise ValueError('cache manifest schema or file roster is invalid')
    for name in CACHE_FILES:
        expected = files[name]
        if not _is_sha256(expected):
            raise ValueError(f'cache manifest has an invalid SHA-256 for {name}')
        path = development_path.parent / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f'cache seal does not match {name}')
    return {'path': manifest_path, 'sha256': sha256_file(manifest_path), 'files': dict(files)}


def _verify_artifact_seal(development_path, frozen):
    cache = _load_cache_seal(development_path)
    if frozen.get('development_sha256') != cache['files']['development.pt']:
        raise ValueError('sealed development cache changed after planning')
    if frozen.get('cache_manifest_sha256') != cache['sha256'] or frozen.get('cache_files') != cache['files']:
        raise ValueError('cache manifest changed after planning')
    if frozen.get('frozen_test_sha256') != cache['files']['frozen-test.pt']:
        raise ValueError('frozen test cache changed after planning')
    if frozen.get('mapper_model_sha256') != sha256_file(ROOT/'src/cgra_ii_predictor/mapper_model.py'):
        raise ValueError('mapper model changed after planning')
    if frozen.get('code_sha256') != sha256_file(Path(__file__)):
        raise ValueError('runner code changed after planning')
    return cache


def _load_verified_plan(args):
    path = args.output/'experiment-plan.json'
    _verify_sealed_json(path, 'experiment plan')
    try:
        frozen = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError('experiment plan is invalid') from error
    if frozen.get('schema') != 'cgra-ii-2x2-bounded-experiment-v1':
        raise ValueError('experiment plan schema is invalid')
    _verify_artifact_seal(args.development, frozen)
    return frozen


def _refuse_existing_outputs(paths, message):
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise ValueError(f'{message}: {", ".join(existing)}')


def _selected_checkpoint_digests(output, configs, seeds):
    digests = {}
    for name in configs:
        for seed in seeds:
            relative = Path('fits')/'original'/name/str(seed)/'selected.pt'
            path = output/relative
            if not path.is_file():
                raise ValueError(f'selected checkpoint is missing: {relative.as_posix()}')
            digests[relative.as_posix()] = sha256_file(path)
    return digests


def _load_selected_checkpoints(output, configs, seeds, expected_digests):
    loaded = {}
    for name in configs:
        for seed in seeds:
            relative = Path('fits')/'original'/name/str(seed)/'selected.pt'
            with (output/relative).open('rb') as stream:
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(block)
                if digest.hexdigest() != expected_digests[relative.as_posix()]:
                    raise ValueError(f'selected checkpoint changed before loading: {relative.as_posix()}')
                stream.seek(0)
                loaded[(name,seed)] = torch.load(stream,weights_only=False,map_location='cpu')
    return loaded


def _holdout_path(development_path):
    return development_path.parent/'frozen-test.pt'


def configs():
    base = {'feature_names': list(ref.COMPACT_FEATURE_NAMES), 'normalization': 'row',
            'output_parameterization': 'residual_softplus', 'updates': 400, 'select_validation': True}
    return [dict(base, name='reference80', updates=80, select_validation=False),
            dict(base, name='long400'), dict(base, name='weighted400', normalization='loss'),
            dict(base, name='direct400', output_parameterization='direct_softplus'),
            dict(base, name='opcode400', feature_names=list(ref.COMPACT_FEATURE_NAMES)+list(ADDED)),
            dict(base, name='opcode-weighted400', normalization='loss', feature_names=list(ref.COMPACT_FEATURE_NAMES)+list(ADDED))]


def load_development(path):
    if path.name != 'development.pt':
        raise ValueError('training accepts only the sealed development.pt cache')
    data = torch.load(path, weights_only=False, map_location='cpu')
    if any(r['split'] not in ('train','validation') for r in data['rows']):
        raise ValueError('development contains holdout or excluded rows')
    return data


def group_partitions(data):
    kinds = {r['source_group']: r['source_kind'] for r in data['rows']}
    folds = {}
    for kind in ('random','program'):
        groups = sorted(g for g,k in kinds.items() if k == kind)
        random.Random(20261004).shuffle(groups)
        for index, group in enumerate(groups):
            folds[group] = index % 3
    partitions = []
    for outer in range(3):
        holdout = sorted(g for g,f in folds.items() if f == outer)
        remaining = set(folds) - set(holdout)
        validation = []
        for kind in ('random','program'):
            groups = sorted(g for g in remaining if kinds[g] == kind)
            random.Random(20261004 + outer).shuffle(groups)
            validation += groups[:max(1, int(.2*len(groups)))]
        partitions.append({'name': 'outer'+str(outer), 'train': sorted(remaining-set(validation)),
                           'validation': sorted(validation), 'evaluation': holdout})
    partitions.append({'name':'original', 'train':sorted(g for g,s in data['assignment'].items() if s == 'train'),
                       'validation':sorted(g for g,s in data['assignment'].items() if s == 'validation'),
                       'evaluation':[]})
    return partitions


def fast_metrics(rows, values):
    values = np.asarray(values)
    truth = np.asarray([r['ii'] for r in rows])
    groups, queries = defaultdict(list), defaultdict(list)
    for index, row in enumerate(rows):
        queries[row['query']].append(index)
    for indices in queries.values():
        if len(indices) != 8 or {tuple(rows[i]['shape']) for i in indices} != set(ref.MAPPER_SHAPES):
            continue
        indices.sort(key=lambda i: ref.MAPPER_SHAPES.index(tuple(rows[i]['shape'])))
        best = truth[indices].min()
        chosen = min(indices, key=lambda i: (values[i],ref.MAPPER_SHAPES.index(tuple(rows[i]['shape']))))
        row = rows[indices[0]]
        groups[row['group']].append((float(truth[chosen]-best), float(truth[chosen] == best), source_kind(row['group_families'])))
    regret = [v[0] for entries in groups.values() for v in entries]
    hit = [v[1] for entries in groups.values() for v in entries]
    by_kind = {}
    for kind in ('random','program'):
        selected = [entries for entries in groups.values() if entries[0][2] == kind]
        if selected:
            by_kind[kind] = {'group_count':len(selected), 'dfg_count':sum(map(len,selected)),
                             'group_macro_regret':float(np.mean([np.mean([v[0] for v in entries]) for entries in selected])),
                             'group_macro_hit':float(np.mean([np.mean([v[1] for v in entries]) for entries in selected]))}
    return {'mae':float(np.abs(values-truth).mean()), 'mean_signed_error':float((values-truth).mean()),
            'mean_regret':float(np.mean(regret)) if regret else None,
            'best_set_hit':float(np.mean(hit)) if hit else None,
            'balanced_group_regret':float(np.mean([v['group_macro_regret'] for v in by_kind.values()])),
            'balanced_group_hit':float(np.mean([v['group_macro_hit'] for v in by_kind.values()])),
            'by_kind':by_kind, 'complete_dfg_count':len(regret), 'source_group_count':len(groups)}


def criterion(metrics):
    return metrics['balanced_group_regret'], -metrics['balanced_group_hit'], metrics['mae']


def fit_job(spec):
    torch.set_num_threads(1)
    if 'artifact_seal' in spec:
        _verify_artifact_seal(Path(spec['development']), spec['artifact_seal'])
    directory = Path(spec['output']); directory.mkdir(parents=True, exist_ok=False)
    data = load_development(Path(spec['development']))
    part, setting = spec['partition'], spec['config']
    training = [r for r in data['training_rows'] if r['group'] in set(part['train'])]
    validation = [r for r in data['training_rows'] if r['group'] in set(part['validation'])]
    evaluation = [r for r in data['training_rows'] if r['group'] in set(part['evaluation'])]
    if set(part['train']) & (set(part['validation']) | set(part['evaluation'])) or set(part['validation']) & set(part['evaluation']):
        raise ValueError('group partition leakage')
    raw = torch.tensor([r['full_features'] for r in training], dtype=torch.float32)
    weights = ref._balanced_group_weights(training)
    if setting['normalization'] == 'loss':
        mean = (raw * weights[:,None]).sum(0)
        scale = ((raw-mean).square() * weights[:,None]).sum(0).sqrt()
    else:
        mean, scale = raw.mean(0), raw.std(0, unbiased=False)
    scale = torch.where(scale < 1e-5,torch.ones_like(scale),scale)
    config = MapperModelConfig(shape_protocol=ref.SHAPE_PROTOCOL_2X2_ID,
        enabled_feature_names=tuple(setting['feature_names']), output_parameterization=setting['output_parameterization']).validate()
    torch.manual_seed(spec['seed'])
    model = DirectMapperIIModel(config, mean, scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)
    lower = torch.tensor([r['lower_bound'] for r in training],dtype=torch.float32)
    truth = torch.tensor([r['ii'] for r in training],dtype=torch.float32)
    queries = ref._rank_queries(training,weights)
    better, worse, pair_weights = [], [], []
    for indices, mass in queries:
        pairs = [(l,r) for l in indices for r in indices if truth[l] < truth[r]]
        better.extend(l for l,r in pairs); worse.extend(r for l,r in pairs)
        if pairs:
            pair_weights.extend([mass/len(pairs)]*len(pairs))
    pair_weights = torch.tensor(pair_weights,dtype=torch.float32)
    qindices = torch.tensor([indices for indices,mass in queries],dtype=torch.long)
    qmass = torch.tensor([mass for indices,mass in queries],dtype=torch.float32)
    qbest = truth[qindices] == truth[qindices].min(1,keepdim=True).values
    histories, states, best_state, best_key, best_epoch = [], defaultdict(list), None, None, None
    for epoch in range(1,setting['updates']+1):
        model.train(); optimizer.zero_grad()
        predicted = model.prediction_for_loss(raw, lower)
        point = (torch.nn.functional.smooth_l1_loss(predicted,truth,reduction='none')*weights).sum()
        pair = (torch.relu(.5-(predicted[worse]-predicted[better]))*pair_weights).sum() if len(pair_weights) else predicted.sum()*0
        if setting['name'] == 'reference80':
            # Preserve the established full-batch summation order exactly.
            top1 = sum(ref._set_top1_loss(predicted[indices],truth[indices])*mass for indices,mass in queries)
        else:
            logits = -predicted[qindices]
            top1 = ((torch.logsumexp(logits,1)-torch.logsumexp(logits.masked_fill(~qbest,-torch.inf),1))*qmass).sum()
        loss = point+.1*pair+.3*top1
        if not torch.isfinite(loss):
            raise ValueError('nonfinite loss')
        loss.backward(); optimizer.step(); model.eval()
        metrics = fast_metrics(validation, ref._predict(model,validation))
        key = criterion(metrics)
        if not setting['select_validation'] or best_key is None or key < best_key:
            best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
            best_key,best_epoch = key,epoch
        histories.append({'epoch':epoch, 'point_loss':float(point), 'pairwise_loss':float(pair),
                          'top1_loss':float(top1), 'total_loss':float(loss), 'validation':metrics})
        for name, parameter in model.named_parameters():
            states[name].append(parameter.detach().numpy().copy())
    # The archive contains every epoch's parameters. Normalization is static.
    np.savez_compressed(directory/'epoch-checkpoints.npz', **{k:np.stack(v) for k,v in states.items()},
                        feature_mean=mean.numpy(),feature_scale=scale.numpy())
    model.load_state_dict(best_state);model.eval()
    torch.save({'config':config.to_dict(),'state_dict':best_state, 'seed':spec['seed'], 'selected_epoch':best_epoch,
                'selection_criterion':'balanced source-group regret, then hit, then MAE; earliest tie'},directory/'selected.pt')
    predictions = {}
    for name, subset in (('validation',validation),('evaluation',evaluation)):
        predictions[name] = [{'query':r['query'],'group':r['group'],'shape':list(r['shape']), 'prediction':p} for r,p in zip(subset,ref._predict(model,subset))]
    result = {'config':setting,'seed':spec['seed'],'partition':part['name'],'selected_epoch':best_epoch,
              'validation':fast_metrics(validation,[r['prediction'] for r in predictions['validation']]),
              'evaluation':fast_metrics(evaluation,[r['prediction'] for r in predictions['evaluation']]) if evaluation else None}
    write_json(directory/'curves.json',histories);write_json(directory/'predictions.json',predictions);write_json(directory/'result.json',result)
    return str(directory)


def plan(args):
    plan_path = args.output/'experiment-plan.json'
    _refuse_existing_outputs((plan_path, _sidecar(plan_path), args.output/'frozen-selection.json',
                              _sidecar(args.output/'frozen-selection.json')),
                            'refusing to overwrite a plan or frozen selection')
    cache = _load_cache_seal(args.development)
    data = load_development(args.development)
    configurations = configs()
    for setting in configurations:
        MapperModelConfig(shape_protocol=ref.SHAPE_PROTOCOL_2X2_ID, enabled_feature_names=tuple(setting['feature_names'])).validate()
    value = {'schema':'cgra-ii-2x2-bounded-experiment-v1','development_sha256':sha256_file(args.development),
             'cache_manifest_sha256':cache['sha256'],'cache_files':cache['files'],
             'frozen_test_sha256':cache['files']['frozen-test.pt'],
             'configs':configurations,'seeds':list(SEEDS),'partitions':group_partitions(data),
             'max_parallel_processes':8,'torch_threads_per_process':1,'hidden_dimensions':[64,32],
             'loss':'stratum-balanced SmoothL1 + .1 pairwise hinge + .3 best-set softmax',
             'loss_weights':'Each source stratum has .5 mass; equal groups within stratum; equal successful rows within group.',
             'selection':'inner validation balanced-group regret, hit, MAE; outer folds only evaluate selected epochs',
             'test_policy':'freeze at most two candidates using development only, then one unified holdout evaluation',
             'selection_gate':'all three outer-fold ensemble regrets no worse than reference80 and mean balanced regret improves; original validation also must improve',
             'new_data_policy':'at most32 unique identities/256queries; admission frozen before labels; supplemental test-only, no retraining',
             'latency_policy':'target has no per-FU duration table; default unit operation latency; no invented latency feature',
             'config6_rationale':'compact collisions and stratum normalization mismatch justify opcode+weighted interaction',
             'direct_training':'positive softplus II capped20; no hard LB floor in loss; inference max(LB,min20(pred)); same loss and width',
             'optimizer':{'name':'AdamW','lr':.002,'weight_decay':.0001},
             'code_sha256':sha256_file(Path(__file__)), 'mapper_model_sha256':sha256_file(ROOT/'src/cgra_ii_predictor/mapper_model.py')}
    _write_sealed_json(plan_path,value)
    print(json.dumps({'plan':str(plan_path),'config_count':len(configurations),'jobs':96}))


def fit(args):
    frozen = _load_verified_plan(args)
    specs = []
    for partition in frozen['partitions']:
        for setting in frozen['configs']:
            for seed in frozen['seeds']:
                directory = args.output/'fits'/partition['name']/setting['name']/str(seed)
                if (directory/'result.json').exists():
                    continue
                specs.append({'development':str(args.development),'partition':partition,'config':setting,'seed':seed,
                              'output':str(directory),'artifact_seal':frozen})
    with ProcessPoolExecutor(max_workers=min(args.jobs,8)) as pool:
        futures = [pool.submit(fit_job,spec) for spec in specs]
        for future in as_completed(futures):
            print(json.dumps({'completed':future.result()}),flush=True)


def summarize(args):
    summary_path = args.output/'development-summary.json'
    selection_path = args.output/'frozen-selection.json'
    _refuse_existing_outputs((summary_path, selection_path, _sidecar(selection_path),
                              args.output/'holdout-evaluation.json'),
                            'refusing to replace an existing summary or frozen selection')
    frozen = _load_verified_plan(args)
    data = load_development(args.development)
    summary = {}
    for partition in frozen['partitions']:
        summary[partition['name']] = {}
        target = 'evaluation' if partition['evaluation'] else 'validation'
        rows = [r for r in data['training_rows'] if r['group'] in set(partition[target])]
        for setting in frozen['configs']:
            members, results = [], []
            for seed in frozen['seeds']:
                directory = args.output/'fits'/partition['name']/setting['name']/str(seed)
                pred = json.loads((directory/'predictions.json').read_text())[target]
                members.append([r['prediction'] for r in pred]); results.append(json.loads((directory/'result.json').read_text()))
                if any((r['query'],list(r['shape'])) != (p['query'],p['shape']) for r,p in zip(rows,pred)) or len(pred) != len(rows):
                    raise ValueError('ensemble row alignment changed')
            summary[partition['name']][setting['name']] = {'ensemble':fast_metrics(rows,np.mean(members,axis=0)), 'members':results}
    eligible = []
    for setting in frozen['configs'][1:]:
        name = setting['name']
        deltas = [summary['outer'+str(f)][name]['ensemble']['balanced_group_regret']-summary['outer'+str(f)]['reference80']['ensemble']['balanced_group_regret'] for f in range(3)]
        val = summary['original'][name]['ensemble']['balanced_group_regret']-summary['original']['reference80']['ensemble']['balanced_group_regret']
        if all(delta <= 0 for delta in deltas) and sum(deltas) < 0 and val < 0:
            eligible.append(name)
    eligible.sort(key=lambda name: sum(summary['outer'+str(f)][name]['ensemble']['balanced_group_regret'] for f in range(3)))
    selected_configs = ['reference80']+eligible[:2]
    selected_checkpoint_sha256 = _selected_checkpoint_digests(args.output,selected_configs,frozen['seeds'])
    selection = {'schema':'cgra-ii-2x2-frozen-selection-v1','plan_sha256':sha256_file(args.output/'experiment-plan.json'),
                 'summary_sha256':None,'test_opened':False,'selected_configs':['reference80']+eligible[:2],
                 'cache_manifest_sha256':frozen['cache_manifest_sha256'],
                 'frozen_test_sha256':frozen['frozen_test_sha256'],
                 'selected_checkpoint_sha256':selected_checkpoint_sha256,
                 'candidate_packaging_allowed':bool(eligible),'reason':'predeclared fold+original-validation stability gate',
                 'default_deployment_changed':False}
    write_json(summary_path,summary)
    selection['summary_sha256'] = sha256_file(args.output/'development-summary.json')
    _write_sealed_json(selection_path,selection)
    print(json.dumps(selection))


def test(args):
    path = args.output/'frozen-selection.json'
    _verify_sealed_json(path,'frozen selection')
    try:
        selection = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError('frozen selection is invalid') from error
    if selection.get('schema') != 'cgra-ii-2x2-frozen-selection-v1':
        raise ValueError('frozen selection schema is invalid')
    if selection.get('test_opened') is not False or (args.output/'holdout-evaluation.json').exists():
        raise ValueError('holdout already opened; frozen selection cannot be reset or replayed')
    frozen = _load_verified_plan(args)
    if selection.get('summary_sha256') != sha256_file(args.output/'development-summary.json') or selection.get('plan_sha256') != sha256_file(args.output/'experiment-plan.json'):
        raise ValueError('selection no longer binds frozen experiment')
    if (selection.get('cache_manifest_sha256') != frozen['cache_manifest_sha256'] or
            selection.get('frozen_test_sha256') != frozen['frozen_test_sha256']):
        raise ValueError('selection no longer binds the sealed cache')
    selected_configs = selection.get('selected_configs')
    config_names = {setting['name'] for setting in frozen['configs']}
    if (not isinstance(selected_configs,list) or not selected_configs or selected_configs[0] != 'reference80' or
            len(selected_configs) > 3 or any(name not in config_names for name in selected_configs)):
        raise ValueError('frozen selected-config roster is invalid')
    expected_checkpoints = _selected_checkpoint_digests(args.output,selected_configs,frozen['seeds'])
    if selection.get('selected_checkpoint_sha256') != expected_checkpoints:
        raise ValueError('selected checkpoints changed after the development freeze')
    checkpoint_items = _load_selected_checkpoints(args.output,selected_configs,frozen['seeds'],expected_checkpoints)
    test_path = _holdout_path(args.development)
    if sha256_file(test_path) != frozen['frozen_test_sha256']:
        raise ValueError('frozen test cache changed after planning')
    # Persist the one permitted opening before deserializing holdout labels. If
    # evaluation fails partway through, a second run cannot silently retry it.
    selection['test_opened'] = True
    selection['test_sha256'] = frozen['frozen_test_sha256']
    _write_sealed_json(path,selection)
    with test_path.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
        if digest.hexdigest() != frozen['frozen_test_sha256']:
            raise ValueError('frozen test cache changed before opening')
        stream.seek(0)
        data = torch.load(stream,weights_only=False,map_location='cpu')
    rows = data['training_rows']; output = {}
    for name in selection['selected_configs']:
        predictions = []
        for seed in frozen['seeds']:
            item = checkpoint_items[(name,seed)]
            model = DirectMapperIIModel(MapperModelConfig(**item['config'])); model.load_state_dict(item['state_dict']); model.eval()
            predictions.append(ref._predict(model,rows))
        averaged = np.mean(predictions,axis=0)
        output[name] = {'ensemble':fast_metrics(rows,averaged), 'per_seed':{str(s):fast_metrics(rows,p) for s,p in zip(frozen['seeds'],predictions)},
                        'rows':[{'identity':r['query'],'group':r['group'],'shape':list(r['shape']),'true_ii':r['ii'],'prediction':float(p),
                                 'members':[float(v[i]) for v in predictions]} for i,(r,p) in enumerate(zip(rows,averaged))]}
    write_json(args.output/'holdout-evaluation.json',output)
    print(json.dumps({name:value['ensemble'] for name,value in output.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('plan','fit','summarize','test'))
    parser.add_argument('--development',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=8)
    arguments = parser.parse_args()
    globals()[arguments.command](arguments)
