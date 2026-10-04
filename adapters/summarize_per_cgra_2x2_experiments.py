#!/usr/bin/env python3
"""Export development-only predictions, paired group intervals and figures."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
for directory in (ROOT/'adapters',ROOT/'src'): sys.path.insert(0,str(directory))
from audit_per_cgra_2x2_model import write_json
from run_per_cgra_2x2_experiments import load_development
from cgra_ii_predictor.mapper_model import DirectMapperIIModel,MapperModelConfig
from per_cgra_2x2_evaluation import evaluate_predictions,choose_fixed_shape,rank_decisions


def paired_intervals(rows,names,samples=2000):
    kinds={r['source_group']:r['source_kind'] for r in rows}
    records={name:{d['identity']:d for d in rank_decisions(rows,name) if d['rank_metrics_evaluable']} for name in names}
    result={}
    for name in names:
        groups=defaultdict(list)
        for identity,decision in records[name].items():
            base=records['reference80'].get(identity)
            if base is not None:
                groups[decision['source_group']].append(decision['absolute_regret']-base['absolute_regret'])
        means={g:float(np.mean(v)) for g,v in groups.items()}
        by_kind={kind:[v for g,v in means.items() if kinds[g]==kind] for kind in ('random','program')}
        rng=np.random.RandomState(20261004)
        replicas=[]
        for _ in range(samples):
            replicas.append(float(np.mean([rng.choice(values,size=len(values),replace=True).mean() for values in by_kind.values() if values])))
        result[name]={'balanced_group_regret_delta':float(np.mean([np.mean(v) for v in by_kind.values() if v])),
                      'lower95':float(np.quantile(replicas,.025)),'upper95':float(np.quantile(replicas,.975)),
                      'groups_by_kind':{k:len(v) for k,v in by_kind.items()},'bootstrap_samples':samples,
                      'definition':'paired source-group resampling within random/program strata, equal stratum mass; negative improves reference80'}
    return result


def disagreement_audit(rows):
    grouped=defaultdict(list)
    for row in rows: grouped[row['identity']].append(row)
    records=[]
    for decision in rank_decisions(rows,'reference80'):
        if not decision['rank_metrics_evaluable']: continue
        entries=grouped[decision['identity']]
        deviations=[np.std([r['scores']['reference80_'+str(seed)] for seed in (17,41,113,239)]) for r in entries]
        records.append({'identity':decision['identity'],'group':decision['source_group'],'kind':entries[0]['source_kind'],
                        'std':float(np.mean(deviations)), 'mae':float(np.mean([abs(r['scores']['reference80']-r['true_ii']) for r in entries])),
                        'rank_failure':not decision['best_set_hit']})
    summary={}
    for kind in ('all','program','random'):
        selected=[r for r in records if kind=='all' or r['kind']==kind]
        def corr(field):
            a=np.asarray([r['std'] for r in selected]);b=np.asarray([r[field] for r in selected],dtype=float)
            return float(np.corrcoef(a,b)[0,1]) if a.std() and b.std() else None
        summary[kind]={'dfg_count':len(selected),'source_group_count':len({r['group'] for r in selected}),
                       'disagreement_mae_pearson':corr('mae'),'disagreement_rank_failure_pearson':corr('rank_failure')}
    return {'summary':summary,'interpretation':'Three outer grouped folds, selected models never trained on these rows; association is not calibration. No pruning or priority policy enabled.','rows':records}


def run(root):
    torch.set_num_threads(1)
    plan=json.loads((root/'experiment-plan.json').read_text())
    data=load_development(root/'development.pt')
    names=[c['name'] for c in plan['configs']]
    rows=[]
    for part in plan['partitions']:
        if not part['evaluation']: continue
        selected=[dict(r,scores={'analytical':r['lower_bound']}) for r in data['rows'] if r['source_group'] in set(part['evaluation'])]
        for row in selected: row['outer_fold']=part['name']
        valid=[r for r in selected if r['full_features'] is not None]
        x=torch.tensor([r['full_features'] for r in valid],dtype=torch.float32)
        lb=torch.tensor([float(r['lower_bound']) for r in valid])
        for name in names:
            predictions=[]
            for seed in plan['seeds']:
                artifact=torch.load(root/'fits'/part['name']/name/str(seed)/'selected.pt',weights_only=False)
                model=DirectMapperIIModel(MapperModelConfig(**artifact['config']));model.load_state_dict(artifact['state_dict']);model.eval()
                with torch.inference_mode(): predictions.append(model(x,lb))
            averaged=torch.stack(predictions).mean(0).tolist()
            for i,(row,value) in enumerate(zip(valid,averaged)):
                row['scores'][name]=value
                for seed,member in zip(plan['seeds'],predictions): row['scores'][name+'_'+str(seed)]=float(member[i])
        inner=[dict(r,split='validation') for r in data['rows'] if r['source_group'] in set(part['validation'])]
        complete={d['identity'] for d in rank_decisions(inner,'ensemble') if d['rank_metrics_evaluable']}
        fixed=choose_fixed_shape([r for r in inner if r['identity'] in complete])
        for row in selected:
            row['scores']['inner_validation_fixed']=0 if tuple(row['shape'])==fixed else 1
            row['scores']['fixed4x4']=0 if tuple(row['shape'])==(4,4) else 1
            for name in names: row['scores'].setdefault(name,None)
        rows+=selected
    with (root/'development-oof-predictions.jsonl').open('w') as stream:
        for row in rows: stream.write(json.dumps({k:v for k,v in row.items() if k!='full_features'},sort_keys=True)+'\n')
    methods=names+['analytical','inner_validation_fixed','fixed4x4']
    metrics={kind:{name:evaluate_predictions([r for r in rows if kind=='all' or r['source_kind']==kind],name,bootstrap_samples=200) for name in methods} for kind in ('all','random','program')}
    for kind in metrics.values():
        for name in ('inner_validation_fixed','fixed4x4'):
            kind[name]['point']={'not_applicable':'fixed selector has no numeric-II prediction'}
            kind[name]['bootstrap']['point']={'not_applicable':True}
    write_json(root/'development-oof-evaluation.json',metrics)
    write_json(root/'development-paired-group-intervals.json',paired_intervals(rows,names))
    write_json(root/'development-oof-disagreement.json',disagreement_audit(rows))
    summary=json.loads((root/'development-summary.json').read_text())
    with (root/'development-summary.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['partition','config','mae','regret','hit','balanced_group_regret','program_group_regret','selected_epochs'])
        for part,settings in summary.items():
            for name,value in settings.items():
                m=value['ensemble'];writer.writerow([part,name,m['mae'],m['mean_regret'],m['best_set_hit'],m['balanced_group_regret'],m['by_kind'].get('program',{}).get('group_macro_regret'),','.join(str(x['selected_epoch']) for x in value['members'])])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for what in ('loss','validation'):
        fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
        for ax,name in zip(axes.flat,names):
            curves=[json.loads((root/'fits/original'/name/str(seed)/'curves.json').read_text()) for seed in plan['seeds']]
            epochs=np.arange(1,len(curves[0])+1)
            if what=='loss':
                for field,label in (('point_loss','SmoothL1'),('pairwise_loss','pair hinge'),('top1_loss','set top1')):
                    values=np.asarray([[r[field] for r in curve] for curve in curves]);ax.plot(epochs,values.mean(0),label=label)
                ax.set_ylabel('unscaled training loss')
            else:
                values=np.asarray([[r['validation']['balanced_group_regret'] for r in curve] for curve in curves])
                ax.plot(epochs,values.mean(0),label='mean across four seeds');ax.fill_between(epochs,values.min(0),values.max(0),alpha=.2,label='seed range')
                for seed,curve in zip(plan['seeds'],curves):
                    selected=json.loads((root/'fits/original'/name/str(seed)/'result.json').read_text())['selected_epoch']
                    ax.scatter(selected,curve[selected-1]['validation']['balanced_group_regret'],s=20)
                ax.set_ylabel('inner validation balanced group regret')
            ax.set_title(name);ax.set_xlabel('update');ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8)
        fig.savefig(root/(what+'-curves.png'),dpi=160);fig.savefig(root/(what+'-curves.pdf'));plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    for name in names:
        vals=[summary['outer'+str(i)][name]['ensemble']['balanced_group_regret'] for i in range(3)]
        ax.plot(range(3),vals,marker='o',label=name)
    ax.set_xticks(range(3),['outer fold 0','outer fold 1','outer fold 2']);ax.set_ylabel('balanced group regret');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(root/'development-folds.png',dpi=160);fig.savefig(root/'development-folds.pdf');plt.close(fig)
    print(json.dumps({'oof_rows':len(rows),'output':str(root)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);run(parser.parse_args().output)
