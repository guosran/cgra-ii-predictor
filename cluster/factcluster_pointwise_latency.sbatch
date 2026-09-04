#!/bin/bash
#SBATCH --job-name=cgra-ii-latency
#SBATCH --partition=dev
#SBATCH --gres=gpu:nvidia_l20:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/fact_data/yibozhang/cgra-ii-model2/logs/latency-%j.log

set -euo pipefail

PROJECT_ROOT=/fact_home/yibozhang/cgra-ii-predictor
DATA_ROOT=/fact_data/yibozhang/cgra-ii-model2/current
RUN_ROOT=/fact_data/yibozhang/cgra-ii-model2/runs
PYTHON=/fact_home/yibozhang/.conda/envs/awq-vggt/bin/python
MANIFEST=${DATA_ROOT}/training-manifest.json
OUTPUT_DIR=${RUN_ROOT}/pointwise-latency-${SLURM_JOB_ID}
EXPECTED_MANIFEST_SHA256=74f19a30e3cd2ce843a8f50edd7b8ef97f50df0c952ec71c0fa72011de76a6f9

if [[ "$#" -lt 1 ]]; then
  echo "usage: sbatch $0 NAME=CHECKPOINT [NAME=CHECKPOINT ...]" >&2
  exit 2
fi

actual_manifest_sha256=$(sha256sum "${MANIFEST}" | awk '{print $1}')
if [[ "${actual_manifest_sha256}" != "${EXPECTED_MANIFEST_SHA256}" ]]; then
  echo "training manifest hash mismatch" >&2
  exit 2
fi

checkpoint_arguments=()
for specification in "$@"; do
  checkpoint_arguments+=(--checkpoint "${specification}")
done

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"
srun --unbuffered "${PYTHON}" adapters/benchmark_pointwise_latency.py \
  --manifest "${MANIFEST}" \
  "${checkpoint_arguments[@]}" \
  --output "${OUTPUT_DIR}/report.json" \
  --samples 32 \
  --warmups 20 \
  --repetitions 5 \
  --seed 20260906 \
  --threads 1 \
  --device cuda
