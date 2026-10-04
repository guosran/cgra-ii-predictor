#!/usr/bin/env python3
"""Evaluate frozen native test queries with the retained 2x2 candidate.

This is retrospective task-local native-label replay, not execution timing or
an ORBIT whole-program speedup measurement. Unknown native outcomes stay null.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
for directory in (ROOT/'src',ROOT/'adapters'):sys.path.insert(0,str(directory))
import train_per_cgra_2x2_model as ref
from audit_per_cgra_2x2_model import write_json,motif
from amoeba_cost_catalog import load_mapper_model,sha256_file,validate_model_architecture
from cgra_ii_predictor.dfg import require_neura_route_expanded_dfg
from cgra_ii_predictor.mapper_model import mapper_feature_vector
from mapping_artifact_protocol import mapper_input_identity
from per_cgra_2x2_evaluation import evaluate_predictions, choose_fixed_shape


def verify_admission(admission):
    ref._verify_embedded_digest(admission,'manifest_sha256','admission')
    if (admission.get('admission_frozen_before_mapping') is not True or
            admission.get('split')!='test_only' or
            admission.get('native_labels_present') is not False or
            admission.get('native_mapping_invoked') is not False):
        raise ValueError('supplemental collection lacks label-free frozen test-only admission')


def verify_roster(queries, bindings, admission):
    actual={(q['mapper_input_identity'],q['rows'],q['cols']) for q in queries}
    expected={(identity,*shape) for identity in bindings for shape in ref.MAPPER_SHAPES}
    if (actual!=expected or len(actual)!=len(queries) or
            admission['projected_native_query_count']!=len(queries) or
            admission['admitted_unique_mapper_input_identity_count']!=len(bindings)):
        raise ValueError('native queries do not cover the full frozen admission roster')


def evaluate(args):
    root=args.collection
    selection=json.loads(args.frozen_selection.read_text())
    if selection.get('selected_configs')!=['reference80']:
        raise ValueError('this evaluation entry point is for the retained reference candidate only')
    # Historical runs predate runner sidecars. Their pre-mapping launch record
    # is the recorded trusted hash, not a retroactively manufactured runner seal.
    launch_path=args.admission.parent/'launch-record.json'
    launch=json.loads(launch_path.read_text())
    if launch.get('model_selection_frozen_before_mapping') is not True:
        raise ValueError('missing pre-mapping model selection freeze')
    for path in (args.admission,args.frozen_selection,
                 args.frozen_selection.parent/'experiment-plan.json',root/'query-manifest.json'):
        if launch['sha256'].get(str(path.resolve()))!=sha256_file(path):
            raise ValueError('pre-mapping launch seal changed: '+str(path))
    for name,key in [('experiment-plan.json','plan_sha256'),('development-summary.json','summary_sha256'),
                     ('frozen-test.pt','test_sha256')]:
        if selection.get(key)!=sha256_file(args.frozen_selection.parent/name):
            raise ValueError('frozen selection binding changed: '+name)
    if selection.get('test_opened') is not True:
        raise ValueError('original unified holdout evaluation is not complete')
    baseline_provenance=json.loads((args.frozen_selection.parent/'provenance.json').read_text())
    if sha256_file(args.candidate/'mapper.pt')!=baseline_provenance['candidate_sha256']:
        raise ValueError('candidate is not the model frozen before supplemental labels')
    admission=json.loads(args.admission.read_text())
    verify_admission(admission)
    if sha256_file(Path(admission['source_manifest_path']))!=admission['source_manifest_sha256']:
        raise ValueError('admitted source manifest changed')
    groups_path=Path(admission['source_groups_path'])
    if sha256_file(groups_path)!=admission['source_groups_sha256']:
        raise ValueError('admitted source groups changed')
    groups=json.loads(groups_path.read_text());bindings={}
    ref._verify_embedded_digest(groups,'manifest_sha256','source groups')
    for group in groups['groups']:
        if group['split']!='test_only':raise ValueError('supplemental group is not test-only')
        for identity in group['mapper_input_identities']:
            if identity in bindings:raise ValueError('identity crosses source groups')
            bindings[identity]=group
    manifest=json.loads((root/'query-manifest.json').read_text())
    ref._verify_embedded_digest(manifest,'manifest_sha256','query manifest')
    provenance=manifest['provenance']
    ref._verify_embedded_digest(provenance,'provenance_sha256','provenance')
    if provenance!=json.loads((root/'provenance.json').read_text()):raise ValueError('provenance disagrees')
    if provenance['source_manifest_sha256']!=admission['source_manifest_sha256'] or provenance['old_native_labels_reused'] is not False:
        raise ValueError('collection is not freshly mapped from admitted sources')
    if provenance['shape_protocol_id']!=ref.SHAPE_PROTOCOL_2X2_ID or provenance['mapper_shapes']!=[list(s) for s in ref.MAPPER_SHAPES]:
        raise ValueError('shape protocol changed')
    if provenance['neura_opt_sha256']!=baseline_provenance['binary_sha256'] or provenance['architecture_sha256']!=baseline_provenance['architecture_sha256']:
        raise ValueError('native mapper or architecture changed')
    for name,digest in provenance['implementation_sha256_by_path'].items():
        if sha256_file(ROOT/name)!=digest:raise ValueError('native collection implementation changed: '+name)
    if len(manifest['queries'])>256 or len(bindings)>32:raise ValueError('supplemental collection cap exceeded')
    verify_roster(manifest['queries'],bindings,admission)
    queries={q['query_id']:q for q in manifest['queries']}
    if len(queries)!=len(manifest['queries']):raise ValueError('duplicate native query')
    outcomes=json.loads((root/'outcomes.json').read_text())
    ref._verify_embedded_digest(outcomes,'manifest_sha256','outcomes')
    if outcomes['query_manifest_sha256']!=sha256_file(root/'query-manifest.json') or outcomes['collection_provenance_sha256']!=provenance['provenance_sha256']:
        raise ValueError('outcomes bind another manifest')
    completion=json.loads((root/'collection-complete.json').read_text())
    if completion['outcomes_sha256']!=sha256_file(root/'outcomes.json') or completion['query_manifest_sha256']!=sha256_file(root/'query-manifest.json'):
        raise ValueError('collection is not complete with these outcomes')
    if len(outcomes['queries'])!=len(queries) or {q['query_id'] for q in outcomes['queries']}!=set(queries):
        raise ValueError('outcome roster differs')
    model,config,metadata=load_mapper_model(args.candidate/'mapper.pt',torch.device('cpu'))
    if config.shape_protocol!=ref.SHAPE_PROTOCOL_2X2_ID:raise ValueError('candidate domain mismatch')
    validate_model_architecture(metadata,provenance['architecture_sha256'])
    baseline=json.loads(args.baseline_evaluation.read_text());fixed=baseline['fixed_shape_from_validation']
    prior_rows=[json.loads(line) for line in
                (args.baseline_evaluation.parent/'baseline-query-rows.jsonl').read_text().splitlines()]
    validation_rows=[r for r in prior_rows if r['split']=='validation']
    if tuple(fixed)!=choose_fixed_shape(validation_rows):
        raise ValueError('fixed selector disagrees with original validation-only selection')
    torch.set_num_threads(1);cache={};rows=[]
    for outcome in outcomes['queries']:
        query=queries[outcome['query_id']]
        for field in ref._QUERY_FIELDS:
            if outcome.get(field)!=query.get(field):raise ValueError('native query lineage mismatch')
        if outcome['status']=='pending':raise ValueError('supplemental query is not terminal')
        ref._verify_result(root,query,outcome,provenance)
        identity=query['mapper_input_identity'];group=bindings[identity]
        if identity not in cache:
            path=ref._safe_path(root,query['dfg_path'],'pre-mapper DFG')
            if sha256_file(path)!=query['dfg_sha256']:raise ValueError('pre-mapper input changed')
            text=path.read_text()
            if mapper_input_identity(text,normalize_static_shapes=True)!=identity or ref._visible_identity(text)!=query['model_visible_graph_identity']:
                raise ValueError('pre-mapper graph identity changed')
            cache[identity]=require_neura_route_expanded_dfg(text)
        result=json.loads((root/outcome['result_path']).read_text());analysis=result.get('analysis',{})
        shape=[query['rows'],query['cols']];families=sorted(group['source_program_families'])
        row={'identity':identity,'source_group':group['group_id'],'source_families':families,
             'source_kind':'program','motif':motif(families),'split':'supplemental_test',
             'shape':shape,'rec_mii':analysis.get('rec_mii'),'res_mii':analysis.get('res_mii'),
             'lower_bound':analysis.get('lower_bound'),'true_ii':outcome.get('compiled_ii'),
             'native_status':'success' if outcome['status']=='success' else (outcome.get('censor_reason') or result.get('censor_reason') or result.get('reason') or 'invalid_output'),
             'scores':{'analytical':analysis.get('lower_bound'),'fixed4x4':0 if shape==[4,4] else 1,'validation_fixed':0 if shape==fixed else 1},
             'result_path':outcome['result_path'],'result_sha256':outcome['result_sha256'],
             'elapsed_seconds':result.get('elapsed_seconds'),'external_timeout_seconds':result.get('external_timeout_seconds')}
        row['native_censor_reason']=outcome.get('censor_reason',result.get('censor_reason'))
        row['analysis_reason']=analysis.get('reason')
        if analysis.get('reason')=='analysis_resource_timeout':
            row['native_status']='analysis_resource_timeout'
        row['scores'].update({name:None for name in ['ensemble']+['member_'+str(s) for s in ref.DEFAULT_SEEDS]})
        if analysis.get('status')=='success' and 0<=analysis['lower_bound']<=20:
            features=mapper_feature_vector(cache[identity],*shape,analysis['rec_mii'],analysis['res_mii'],analysis['lower_bound'],shape_protocol=ref.SHAPE_PROTOCOL_2X2_ID)
            x=torch.tensor([features],dtype=torch.float32);lb=torch.tensor([float(analysis['lower_bound'])])
            with torch.inference_mode():
                row['scores']['ensemble']=float(model(x,lb))
                for seed,member in zip(ref.DEFAULT_SEEDS,model.members):row['scores']['member_'+str(seed)]=float(member(x,lb))
        rows.append(row)
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'query-rows.jsonl').open('w') as stream:
        for row in rows:stream.write(json.dumps(row,sort_keys=True)+'\n')
    fields=['identity','source_group','source_families','motif','shape','rec_mii','res_mii','lower_bound','true_ii','native_status','ensemble']+['member_'+str(s) for s in ref.DEFAULT_SEEDS]
    with (args.output/'query-rows.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:row['scores'].get(k,row.get(k)) for k in fields})
    methods=['ensemble','analytical','validation_fixed','fixed4x4']+['member_'+str(s) for s in ref.DEFAULT_SEEDS]
    metrics={name:evaluate_predictions(rows,name,bootstrap_samples=1000) for name in methods}
    for name in ('validation_fixed','fixed4x4'):
        metrics[name]['point']={'not_applicable':'fixed selector has no numeric-II prediction'};metrics[name]['bootstrap']['point']={'not_applicable':True}
    by_motif={name:evaluate_predictions([r for r in rows if r['motif']==name],'ensemble') for name in sorted({r['motif'] for r in rows})}
    write_json(args.output/'evaluation.json',{'measurement':'retrospective replay of fresh native per-task II labels','whole_program_speedup_measured':False,
        'retained_candidate_sha256':sha256_file(args.candidate/'mapper.pt'),'frozen_selection_sha256':sha256_file(args.frozen_selection),
        'admission_sha256':sha256_file(args.admission),'baseline_fixed_selector_sha256':sha256_file(args.baseline_evaluation),
        'pre_mapping_launch_sha256':sha256_file(launch_path),
        'fixed_selector_source':'recomputed from original validation rows only',
        'collection_provenance':provenance,'outcomes_sha256':sha256_file(root/'outcomes.json'),'methods':metrics,'by_motif':by_motif})
    print(json.dumps({name:{'coverage':m['coverage'],'rank':m['rank']['metrics']} for name,m in metrics.items() if name in ('ensemble','analytical')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','admission','frozen-selection','baseline-evaluation','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--candidate',type=Path,default=ROOT/'models/candidates/per-cgra-2x2')
    evaluate(parser.parse_args())
