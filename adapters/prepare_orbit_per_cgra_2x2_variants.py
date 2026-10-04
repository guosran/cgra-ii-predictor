#!/usr/bin/env python3
"""Freeze four evidenced ORBIT transform tasks as evaluation-only native inputs.

This source preparation never maps a graph or changes ORBIT. Benchmark variants
remain excluded from training, even when they have new graph identities.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'adapters')]
import extract_amoeba_task_dfgs as extractor
import prepare_real_per_cgra_2x2_sources as real
import train_per_cgra_2x2_model as reference
from audit_per_cgra_2x2_model import write_json
from mapping_artifact_protocol import mapper_input_identity,sha256_file,canonical_json_sha256
from cgra_ii_predictor.dfg import require_neura_route_expanded_dfg

ARTIFACT=Path('/home/x/shiran/project/orbit-artifact')
SOURCES=[
 ('raytracing',ARTIFACT/'results/input0-ray-fission-split4-diagnostic/search/search-output.mlir',
  'eb7c49308524e2d1b2abf274460db4817908761decbe8789496df1fb047e70e1',
  ['Task_13.fission.0','Task_13'],'ray-carried-min-index-fission-v1'),
 ('harris',ARTIFACT/'diagnostics/20261004-search-direction/v34-2x2-harris-guidance-pilot/admission/tiling-replay/candidate.mlir',
  '34be852b09be0d45a0cb192ea1d7c271eb4ce69690681f10e08bfb45a3c0b2c9',
  ['Task_1.tile.1.0','Task_1.tile.1.1'],'post-neura-mn-tiling'),
]


def seal(path,value):
    value['manifest_sha256']=canonical_json_sha256(value)
    write_json(path,value)


def extract_with_region_identities(text):
    """Locally fingerprint missing identities; do not claim ORBIT enumeration.

    The ordinary extractor requires enumerator-issued body hashes. These saved
    transform artifacts predate that annotation. Their exact task-region text
    supplies a local provenance identity, recorded separately from its origin.
    Native graph identities normalize this metadata away.
    """
    original=extractor._source_task_body_sha256
    identities={}
    def body_hash(task,region):
        if extractor.SOURCE_TASK_BODY_SHA_ATTR in '\n'.join(region):
            digest=original(task,region);origin='existing_ORBIT_attribute'
        else:
            digest=hashlib.sha256('\n'.join(region).encode()).hexdigest()
            origin='local_sha256_of_exact_extracted_task_region_not_enumerator_attestation'
        identities[task]={'sha256':digest,'origin':origin}
        return digest
    extractor._source_task_body_sha256=body_hash
    try:return extractor.extract_task_dfg_texts(text),identities
    finally:extractor._source_task_body_sha256=original


def run(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'admission.json').exists() or (output/'source-manifest.json').exists():
        raise ValueError('refusing to overwrite frozen variant admission')
    if sha256_file(args.neura_opt)!=real.EXPECTED_NEURA_OPT_SHA256 or sha256_file(args.architecture)!=real.EXPECTED_ARCHITECTURE_SHA256:
        raise ValueError('pinned mapper/architecture changed')
    _,_,_,old=real._load_existing_training_metadata(args.base_root)
    canonical_ids,canonical_visible,_=real._canonical_identity_sets(args.canonical_map)
    prior=json.loads(args.prior_manifest.read_text())
    known_ids=old['mapper_input_identity']|canonical_ids|{x['mapper_input_identity'] for x in prior['candidates']}
    known_visible=old['model_visible_graph_identity']|canonical_visible|{x['model_visible_graph_identity'] for x in prior['candidates']}
    candidates=[];decisions=[]
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for family,source,expected,tasks,proof in SOURCES:
        if sha256_file(source)!=expected:raise ValueError('transform source changed: '+str(source))
        text=source.read_text()
        if proof not in text:raise ValueError('missing actual transform proof')
        normalization=None
        if text.startswith('"builtin.module"'):
            # Reprint generic MLIR in the registered custom syntax. No pass or
            # transformation is requested, and the source file stays untouched.
            normalized=output/(family+'-parser-reprinted.mlir')
            binary_sha=sha256_file(args.orbit_opt)
            command=[str(args.orbit_opt),str(source),'-o',str(normalized)]
            result=subprocess.run(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
            (output/(family+'-parser-reprint.stderr.log')).write_bytes(result.stderr)
            if result.returncode or sha256_file(args.orbit_opt)!=binary_sha:
                raise ValueError('ORBIT parser reprint failed or executable changed')
            normalization={'command':command,'binary_sha256':binary_sha,'output_sha256':sha256_file(normalized),
                           'passes':[],'body_hash_input':'parser-reprinted task region'}
            text=normalized.read_text()
        texts,identities=extract_with_region_identities(text)
        for task in tasks:
            directory=output/'extraction'/family/task;directory.mkdir(parents=True,exist_ok=True)
            before=directory/'before-data-mov.mlir';after=directory/'route-expanded.mlir'
            before.write_text(texts[task])
            command=[str(args.neura_opt),str(before),'--architecture-spec='+str(args.architecture),'--insert-data-mov','-o',str(after)]
            result=subprocess.run(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
            (directory/'stdout.log').write_bytes(result.stdout);(directory/'stderr.log').write_bytes(result.stderr)
            record={'source_family':family,'source_task':task,'source_file':str(source),'source_file_sha256':expected,
                    'transform_proof':proof,'parser_reprint':normalization,'body_identity':identities[task],'command':command,'exit_code':result.returncode,
                    'pre_expansion_sha256':sha256_file(before),'native_mapping_invoked':False,'benchmark_overlap':True,
                    'training_excluded':True}
            if result.returncode:raise ValueError('route expansion failed: '+str(directory))
            graph=after.read_text();require_neura_route_expanded_dfg(graph)
            identity=mapper_input_identity(graph,normalize_static_shapes=True);visible=reference._visible_identity(graph)
            record.update(mapper_input_identity=identity,model_visible_graph_identity=visible,dfg_sha256=sha256_file(after))
            if identity in known_ids or visible in known_visible:
                record['admission']='skip_existing_or_shared_graph_identity'
            else:
                record['admission']='benchmark_variant_test_only'
                candidate={'candidate_id':'orbit-final/'+family+'/'+task,'mapper_input_identity':identity,
                           'model_visible_graph_identity':visible,'source_path':str(after.relative_to(output)),
                           'source_sha256':sha256_file(after),'source_names':['orbit-final/'+family+'/'+task],
                           'source_program_families':['orbit-final/'+family],'domain':'real_program','domains':['real_program'],
                           'leakage_lineage_id':'orbit-final/'+family,'native_mapping_invoked':False,'native_labels_present':False,
                           'benchmark_overlap':True,'training_excluded':True}
                candidates.append(candidate);known_ids.add(identity);known_visible.add(visible)
            decisions.append(record)
    count=len(candidates);prior_count=prior['candidate_count']
    if count>8 or count+prior_count>32 or 8*(count+prior_count)>256:raise ValueError('combined supplemental budget exceeded')
    write_json(output/'extraction-audit.json',{'decisions':decisions,'native_mapping_invoked':False,
        'prior_source_manifest_sha256':sha256_file(args.prior_manifest),'canonical_map_sha256':sha256_file(args.canonical_map),
        'architecture_sha256':sha256_file(args.architecture),'native_binary_sha256':sha256_file(args.neura_opt),
        'training_exclusions_sha256':sha256_file(args.base_root/'training-exclusions.json'),
        'source_semantic_equivalence_executed':False,'transform_variants_are_not_new_independent_source_groups':True})
    if not candidates:raise ValueError('all transformed tasks duplicate existing graph identities; no new queries')
    manifest=output/'source-manifest.json';groups=output/'source-groups.json'
    seal(manifest,{'schema':real.SOURCE_SCHEMA,'evaluation_only':True,'training_eligible':False,
        'candidate_count':count,'query_count':count,'candidates':candidates,'shape_set':[list(s) for s in real.SHAPES_2X2]})
    grouped=real._group_admitted(candidates)
    seal(groups,{'schema':real.SOURCE_GROUP_SCHEMA,'source_manifest_sha256':sha256_file(manifest),
        'split_policy':'benchmark_variants_test_only_before_native_mapping','groups':grouped})
    seal(output/'admission.json',{'schema':real.ADMISSION_SCHEMA,'split':'test_only','admission_frozen_before_mapping':True,
        'native_mapping_invoked':False,'native_labels_present':False,'benchmark_overlap':True,'training_excluded':True,
        'source_manifest_path':str(manifest),'source_manifest_sha256':sha256_file(manifest),
        'source_groups_path':str(groups),'source_groups_sha256':sha256_file(groups),
        'admitted_unique_mapper_input_identity_count':count,'admitted_connected_source_group_count':len(grouped),
        'projected_native_query_count':count*8,'previous_supplemental_identities':prior_count,
        'total_supplemental_identities':count+prior_count,'total_supplemental_queries':8*(count+prior_count),
        'extraction_audit_sha256':sha256_file(output/'extraction-audit.json'),
        'training_exclusions_sha256':sha256_file(args.base_root/'training-exclusions.json'),
        'decisions':decisions})
    print(json.dumps({'admitted_identities':count,'queries':count*8,'source_groups':len(grouped)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prior-manifest',type=Path,required=True)
    parser.add_argument('--base-root',type=Path,default=real.DEFAULT_BASE_ROOT)
    parser.add_argument('--canonical-map',type=Path,default=real.DEFAULT_CANONICAL_MAP)
    parser.add_argument('--neura-opt',type=Path,default=real.DEFAULT_NEURA_OPT)
    parser.add_argument('--architecture',type=Path,default=real.DEFAULT_ARCHITECTURE)
    parser.add_argument('--orbit-opt',type=Path,default=Path('/home/x/shiran/project/orbit-build-publication/tools/mlir-amoeba-opt/mlir-amoeba-opt'))
    run(parser.parse_args())
