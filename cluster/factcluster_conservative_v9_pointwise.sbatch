#!/bin/bash
#SBATCH --job-name=cgra-ii-v9-upper
#SBATCH --partition=dev
#SBATCH --gres=gpu:nvidia_l20:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/fact_data/yibozhang/cgra-ii-model2/logs/conservative-v9-%j.log

set -euo pipefail

PROJECT_ROOT=/fact_home/yibozhang/cgra-ii-predictor
DATA_ROOT=/fact_data/yibozhang/cgra-ii-model2
RUNS=${DATA_ROOT}/runs
PYTHON=/fact_home/yibozhang/.conda/envs/awq-vggt/bin/python
MANIFEST=${DATA_ROOT}/multi-cgra-v8/motif-v8-formal-seed-20260911/corpus-manifest.json
ENSEMBLE=${DATA_ROOT}/evaluations/pointwise-v8-v9-largeop-1480204.json
LARGEOP=${RUNS}/pointwise-v8-v9-largeop-residual128-rank01-s12-1480202/model.pt
OUTPUT=${DATA_ROOT}/evaluations/conservative-v8-v9-${SLURM_JOB_ID}.json

cd "${PROJECT_ROOT}"
srun --unbuffered "${PYTHON}" adapters/analyze_conservative_pointwise.py \
  --manifest "${MANIFEST}" \
  --ensemble-report "${ENSEMBLE}" \
  --checkpoint largeop_rank01="${LARGEOP}" \
  --checkpoint residual128="${RUNS}/pointwise-v8-residual128-p64-s12-1478199/model.pt" \
  --checkpoint residual128_rank01="${RUNS}/pointwise-v8-residual128-rank01-s12-1478681/model.pt" \
  --expert-checkpoint "${RUNS}/pointwise-v8-continuous160-struct-s13-1478079/model.pt" \
  --expert-name continuous160_struct_legacy_routing_norm \
  --quantile 0.85 \
  --seed 20260911 \
  --device cuda \
  --output "${OUTPUT}"

sha256sum "${OUTPUT}"
