#!/bin/bash
# Evaluate one validation-selected ranking checkpoint and emit the frozen
# multi-nested task/shape catalogue used by the independent program scorer.
#SBATCH --job-name=cgra-ii-v8-rank-eval
#SBATCH --partition=dev
#SBATCH --gres=gpu:nvidia_l20:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/fact_data/yibozhang/cgra-ii-model2/logs/pointwise-v8-rank-eval-%j.log

set -euo pipefail

PROJECT_ROOT=/fact_home/yibozhang/cgra-ii-predictor
DATA_ROOT=/fact_data/yibozhang/cgra-ii-model2/multi-cgra-v8
EVALUATION_ROOT=/fact_data/yibozhang/cgra-ii-model2/evaluations
E2E_INPUT_ROOT=/fact_data/yibozhang/cgra-ii-model2/e2e-input
PYTHON=/fact_home/yibozhang/.conda/envs/awq-vggt/bin/python
MANIFEST=${DATA_ROOT}/motif-v8-formal-seed-20260911/corpus-manifest.json

: "${EXPECTED_MANIFEST_SHA256:?set EXPECTED_MANIFEST_SHA256}"
: "${RANKING_CHECKPOINT:?set RANKING_CHECKPOINT}"
: "${EXPECTED_RANKING_CHECKPOINT_SHA256:?set EXPECTED_RANKING_CHECKPOINT_SHA256}"

RUN_LABEL=${RUN_LABEL:-rank-selected}
OUTPUT_DIR=${EVALUATION_ROOT}/pointwise-v8-${RUN_LABEL}-${SLURM_JOB_ID}
ENSEMBLE_REPORT=${OUTPUT_DIR}/ensemble.json
MULTI_CATALOG=${OUTPUT_DIR}/multi-nested-costs.json
MULTI_TIMING=${OUTPUT_DIR}/multi-nested-predictor-timing.json
EXPECTED_MULTI_MANIFEST_SHA256=04f50ec7198191209fb2eb6d8c4472d88ed5fe5d821757a8d733db73e2d8e9d4
EXPECTED_MULTI_FEATURES_SHA256=74b8cf445f241f15e6bbe8e3f2fa8fc370c298ad731025e7848ea472be0f8171

verify_sha256() {
  local path=$1
  local expected=$2
  local actual
  actual=$(sha256sum "${path}" | awk '{print $1}')
  if [[ "${actual}" != "${expected}" ]]; then
    echo "artifact hash mismatch: ${path}" >&2
    exit 2
  fi
}

verify_sha256 "${MANIFEST}" "${EXPECTED_MANIFEST_SHA256}"
verify_sha256 "${RANKING_CHECKPOINT}" "${EXPECTED_RANKING_CHECKPOINT_SHA256}"
verify_sha256 \
  "${E2E_INPUT_ROOT}/multi/amoeba-v8-multi-nested-candidates.jsonl" \
  "${EXPECTED_MULTI_MANIFEST_SHA256}"
verify_sha256 \
  "${E2E_INPUT_ROOT}/multi/amoeba-v8-multi-nested-query-features.json" \
  "${EXPECTED_MULTI_FEATURES_SHA256}"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

echo "job_id=${SLURM_JOB_ID}"
echo "host=${SLURMD_NODENAME:-unknown}"
echo "selection_split=validation_only"
echo "run_label=${RUN_LABEL}"
echo "ranking_checkpoint=${RANKING_CHECKPOINT}"
echo "ranking_checkpoint_sha256=${EXPECTED_RANKING_CHECKPOINT_SHA256}"
echo "output_dir=${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

srun --unbuffered "${PYTHON}" adapters/ensemble_pointwise_checkpoints.py \
  --manifest "${MANIFEST}" \
  --checkpoint "rank_selected=${RANKING_CHECKPOINT}" \
  --output "${ENSEMBLE_REPORT}" \
  --batch-size 32 \
  --seed 20260911 \
  --device cuda

srun --unbuffered "${PYTHON}" adapters/amoeba_cost_catalog.py \
  --manifest "${E2E_INPUT_ROOT}/multi/amoeba-v8-multi-nested-candidates.jsonl" \
  --analytical-input "${E2E_INPUT_ROOT}/multi/amoeba-v8-multi-nested-query-features.json" \
  --task-dfg "Task_0=${E2E_INPUT_ROOT}/multi/dfgs/Task_0.mlir" \
  --task-dfg "Task_1=${E2E_INPUT_ROOT}/multi/dfgs/Task_1.mlir" \
  --task-dfg "Task_2=${E2E_INPUT_ROOT}/multi/dfgs/Task_2.mlir" \
  --task-dfg "Task_3=${E2E_INPUT_ROOT}/multi/dfgs/Task_3.mlir" \
  --task-dfg "Task_4=${E2E_INPUT_ROOT}/multi/dfgs/Task_4.mlir" \
  --checkpoint "rank_selected=${RANKING_CHECKPOINT}" \
  --ensemble-report "${ENSEMBLE_REPORT}" \
  --output "${MULTI_CATALOG}" \
  --timing-output "${MULTI_TIMING}" \
  --device cuda

sha256sum \
  "${RANKING_CHECKPOINT}" \
  "${ENSEMBLE_REPORT}" \
  "${MULTI_CATALOG}" \
  "${MULTI_TIMING}"
