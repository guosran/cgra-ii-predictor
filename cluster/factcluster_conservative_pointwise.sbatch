#!/bin/bash
#SBATCH --job-name=cgra-ii-conservative
#SBATCH --partition=dev
#SBATCH --gres=gpu:nvidia_l20:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/fact_data/yibozhang/cgra-ii-model2/logs/conservative-%j.log

set -euo pipefail

PROJECT_ROOT=/fact_home/yibozhang/cgra-ii-predictor
DATA_ROOT=/fact_data/yibozhang/cgra-ii-model2
MANIFEST=${DATA_ROOT}/multi-cgra-v8/motif-v8-formal-seed-20260911/corpus-manifest.json
ENSEMBLE=${DATA_ROOT}/evaluations/pointwise-v8-1479107/ensemble.json
RUNS=${DATA_ROOT}/runs
PYTHON=/fact_home/yibozhang/.conda/envs/awq-vggt/bin/python
OUTPUT=${DATA_ROOT}/evaluations/conservative-v8-${SLURM_JOB_ID}.json
QUANTILE=${QUANTILE:-0.75}

cd "${PROJECT_ROOT}"
srun --unbuffered "${PYTHON}" adapters/analyze_conservative_pointwise.py \
  --manifest "${MANIFEST}" \
  --ensemble-report "${ENSEMBLE}" \
  --checkpoint continuous128="${RUNS}/pointwise-v8-continuous128-p64-s11-1478198/model.pt" \
  --checkpoint continuous128_legacy4x4="${RUNS}/pointwise-v8-continuous128-legacy4x4-s14-1478183/model.pt" \
  --checkpoint continuous160_struct="${RUNS}/pointwise-v8-continuous160-struct-p64-s13-1478200/model.pt" \
  --checkpoint residual128="${RUNS}/pointwise-v8-residual128-p64-s12-1478199/model.pt" \
  --checkpoint residual128_legacy4x4="${RUNS}/pointwise-v8-residual128-legacy4x4-s15-1478184/model.pt" \
  --checkpoint residual128_rank01="${RUNS}/pointwise-v8-residual128-rank01-s12-1478681/model.pt" \
  --expert-checkpoint "${RUNS}/pointwise-v8-continuous160-struct-s13-1478079/model.pt" \
  --expert-name continuous160_struct_legacy_routing_norm \
  --quantile "${QUANTILE}" \
  --device cuda \
  --output "${OUTPUT}"

sha256sum "${OUTPUT}"
