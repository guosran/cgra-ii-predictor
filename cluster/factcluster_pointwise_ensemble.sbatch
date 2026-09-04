#!/bin/bash
#SBATCH --job-name=cgra-ii-ensemble
#SBATCH --partition=dev
#SBATCH --gres=gpu:nvidia_l20:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/fact_data/yibozhang/cgra-ii-model2/logs/ensemble-%j.log

set -euo pipefail

PROJECT_ROOT=/fact_home/yibozhang/cgra-ii-predictor
DATA_ROOT=/fact_data/yibozhang/cgra-ii-model2/current
RUN_ROOT=/fact_data/yibozhang/cgra-ii-model2/runs
PYTHON=/fact_home/yibozhang/.conda/envs/awq-vggt/bin/python
MANIFEST=${DATA_ROOT}/training-manifest.json
OUTPUT_DIR=${RUN_ROOT}/pointwise-ensemble-${SLURM_JOB_ID}
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
  if [[ "${specification}" != *=* ]]; then
    echo "invalid checkpoint specification: ${specification}" >&2
    exit 2
  fi
  checkpoint=${specification#*=}
  if [[ ! -f "${checkpoint}" ]]; then
    echo "checkpoint does not exist: ${checkpoint}" >&2
    exit 2
  fi
  checkpoint_arguments+=(--checkpoint "${specification}")
done

mkdir -p "${OUTPUT_DIR}"
printf '%s\n' "$@" > "${OUTPUT_DIR}/checkpoint-specifications.txt"
cd "${PROJECT_ROOT}"

echo "job_id=${SLURM_JOB_ID}"
echo "host=${SLURMD_NODENAME:-unknown}"
echo "manifest_sha256=${actual_manifest_sha256}"
echo "selection_split=validation_only"
echo "primary_metric=successful_candidate_continuous_ii_mae"
echo "output_dir=${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

srun --unbuffered "${PYTHON}" adapters/ensemble_pointwise_checkpoints.py \
  --manifest "${MANIFEST}" \
  "${checkpoint_arguments[@]}" \
  --output "${OUTPUT_DIR}/report.json" \
  --batch-size 32 \
  --seed 20260906 \
  --device cuda
