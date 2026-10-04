#!/usr/bin/env python3
"""Evaluate saved baseline rows without training or changing any model."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_per_cgra_2x2_model import write_json
from per_cgra_2x2_evaluation import evaluate_predictions, choose_fixed_shape, rank_decisions


def run(root, bootstrap):
    rows = [json.loads(line) for line in (root/'baseline-query-rows.jsonl').read_text().splitlines()]
    fixed = choose_fixed_shape([r for r in rows if r['split']=='validation'])
    for row in rows:
        row['scores']['validation_fixed'] = 0 if tuple(row['shape'])==fixed else 1
        row['scores']['fixed4x4'] = 0 if tuple(row['shape'])==(4,4) else 1
    methods = ['analytical','validation_fixed','fixed4x4','ensemble','member_17','member_41','member_113','member_239']
    summary = {'fixed_shape_from_validation':list(fixed),'by_split':{},'motif':{},'family':{},'shape':{},
               'fixed_shape_point_error_is_not_an_ii_prediction':True,
               'resource_limit':'II ties can consume different tile counts; shape-hit is not whole-program quality.'}
    for split in ('train','validation','test','excluded'):
        subset = [r for r in rows if r['split']==split]
        summary['by_split'][split] = {}
        for kind in ('all','random','program'):
            selected = [r for r in subset if kind=='all' or r['source_kind']==kind]
            if selected:
                summary['by_split'][split][kind] = {method:evaluate_predictions(selected,method,bootstrap_samples=bootstrap) for method in methods}
    for key in ('motif','family'):
        groups = defaultdict(list)
        for row in rows:
            if row['split']=='excluded': continue
            values = row['source_families'] if key=='family' else [row['motif']]
            for value in values: groups[row['split']+':'+value].append(row)
        summary[key] = {name:evaluate_predictions(entries,'ensemble') for name,entries in groups.items()}
    for split in ('train','validation','test'):
        for shape in sorted({tuple(r['shape']) for r in rows}):
            selected = [r for r in rows if r['split']==split and tuple(r['shape'])==shape and r['true_ii'] is not None and r['scores'].get('ensemble') is not None]
            errors = [r['scores']['ensemble']-r['true_ii'] for r in selected]
            summary['shape'][split+':'+str(shape)] = {'count':len(errors),'mae':float(np.mean(np.abs(errors))),'signed_error':float(np.mean(errors))}
    # Co-occurrence is descriptive, not a causal decomposition of softplus.
    dev = [r for r in rows if r['split'] in ('train','validation')]
    by_id = defaultdict(list)
    for row in dev: by_id[row['identity']].append(row)
    decisions = rank_decisions(dev,'ensemble')
    diagnostic = []
    for decision in decisions:
        if not decision['rank_metrics_evaluable']: continue
        entries = by_id[decision['identity']]
        ordered = sorted(entries,key=lambda r:(r['scores']['ensemble'],[(2,2),(2,4),(4,2),(2,6),(6,2),(2,8),(8,2),(4,4)].index(tuple(r['shape']))))
        chosen = ordered[0]
        diagnostic.append({'identity':decision['identity'],'group':decision['source_group'],
                           'regret':decision['absolute_regret'],'rank_failure':not decision['best_set_hit'],
                           'lower_bound_tie':sum(r['lower_bound']==chosen['lower_bound'] for r in entries)>1,
                           'small_selected_residual':chosen['true_ii']-chosen['lower_bound']<=1,
                           'any_saturation':any(r['scores']['ensemble']>=20 for r in entries),
                           'any_member_saturation':any(r['scores']['member_'+str(seed)]>=20 for r in entries for seed in (17,41,113,239)),
                           'mean_member_std':float(np.mean([r['member_std'] for r in entries])),
                           'mean_absolute_error':float(np.mean([abs(r['scores']['ensemble']-r['true_ii']) for r in entries]))})
    failed = [d for d in diagnostic if d['rank_failure']]
    stds = np.asarray([d['mean_member_std'] for d in diagnostic]); errs = np.asarray([d['mean_absolute_error'] for d in diagnostic]); failures=np.asarray([float(d['rank_failure']) for d in diagnostic])
    def corr(a,b):
        return float(np.corrcoef(a,b)[0,1]) if a.std() and b.std() else None
    write_json(root/'ranking-output-audit.json',{'development_only':True,'complete_dfg_count':len(diagnostic),
        'misrank_count':len(failed),'misrank_cooccurrence':{k:sum(d[k] for d in failed) for k in ('lower_bound_tie','small_selected_residual','any_saturation','any_member_saturation')},
        'all_complete_cooccurrence':{k:sum(d[k] for d in diagnostic) for k in ('lower_bound_tie','small_selected_residual','any_saturation','any_member_saturation')},
        'association_scope':'includes training resubstitution; use development-oof-disagreement.json for held-out association',
        'disagreement_error_pearson':corr(stds,errs),'disagreement_rank_failure_pearson':corr(stds,failures),
        'disagreement_quartiles':[{'quartile':q+1,'dfg_count':len(ix),'mean_std':float(stds[ix].mean()),'error':float(errs[ix].mean()),'rank_failure_rate':float(failures[ix].mean())} for q,ix in enumerate(np.array_split(np.argsort(stds),4))],
        'interpretation':'Retrospective development association only; not calibrated confidence, failure probability, or a safe-pruning guarantee.', 'dfgs':diagnostic})
    # Fixed selectors do not predict II; suppress their point-error tables.
    for split in summary['by_split'].values():
        for kind in split.values():
            for method in ('validation_fixed','fixed4x4'):
                kind[method]['point']={'not_applicable':'fixed selector has no numeric-II prediction'}
                kind[method]['bootstrap']['point']={'not_applicable':True}
    write_json(root/'baseline-evaluation.json',summary)
    print(json.dumps({'baseline_evaluation':str(root/'baseline-evaluation.json'),'fixed_shape':list(fixed)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--bootstrap',type=int,default=500)
    args=parser.parse_args();run(args.output,args.bootstrap)
