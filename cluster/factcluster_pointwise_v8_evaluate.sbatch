#!/bin/bash
#SBATCH --job-name=cgra-ii-v8-eval
#SBATCH --partition=dev
#SBATCH --gres=gpu:nvidia_l20:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/fact_data/yibozhang/cgra-ii-model2/logs/pointwise-v8-eval-%j.log

set -euo pipefail

PROJECT_ROOT=/fact_home/yibozhang/cgra-ii-predictor
DATA_ROOT=/fact_data/yibozhang/cgra-ii-model2/multi-cgra-v8
EVALUATION_ROOT=/fact_data/yibozhang/cgra-ii-model2/evaluations
E2E_INPUT_ROOT=/fact_data/yibozhang/cgra-ii-model2/e2e-input
PYTHON=/fact_home/yibozhang/.conda/envs/awq-vggt/bin/python
MANIFEST=${DATA_ROOT}/motif-v8-formal-seed-20260911/corpus-manifest.json
LEGACY_ENSEMBLE=${PROJECT_ROOT}/evaluations/model2-pointwise-ensemble-2026-09-05.json
: "${EXPECTED_MANIFEST_SHA256:?set EXPECTED_MANIFEST_SHA256}"
: "${CONTINUOUS_CHECKPOINT:?set CONTINUOUS_CHECKPOINT}"
: "${DISTRIBUTION_CHECKPOINT:?set DISTRIBUTION_CHECKPOINT}"
: "${STRUCTURAL_CHECKPOINT:?set STRUCTURAL_CHECKPOINT}"

OUTPUT_DIR=${EVALUATION_ROOT}/pointwise-v8-${SLURM_JOB_ID}
ENSEMBLE_REPORT=${OUTPUT_DIR}/ensemble.json
LATENCY_REPORT=${OUTPUT_DIR}/latency-gpu.json
COMPARISON_REPORT=${OUTPUT_DIR}/legacy-v8-mapper4x4.json
HYBRID_REPORT=${OUTPUT_DIR}/analytical-hybrid.json
PARALLEL_CATALOG=${OUTPUT_DIR}/parallel-nested-costs.json
PARALLEL_TIMING=${OUTPUT_DIR}/parallel-nested-predictor-timing.json
MULTI_CATALOG=${OUTPUT_DIR}/multi-nested-costs.json
MULTI_TIMING=${OUTPUT_DIR}/multi-nested-predictor-timing.json
EXPECTED_PARALLEL_MANIFEST_SHA256=29103f5dd1003a4d60140598a84738fa7cb8e2bacbfb90af56d647a155c84d4a
EXPECTED_PARALLEL_FEATURES_SHA256=4e237ea3ec367538fc062f818751b35af22ada1c55e456c738ee8748bee7ba6b
EXPECTED_MULTI_MANIFEST_SHA256=04f50ec7198191209fb2eb6d8c4472d88ed5fe5d821757a8d733db73e2d8e9d4
EXPECTED_MULTI_FEATURES_SHA256=74b8cf445f241f15e6bbe8e3f2fa8fc370c298ad731025e7848ea472be0f8171

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

actual_manifest_sha256=$(sha256sum "${MANIFEST}" | awk '{print $1}')
if [[ "${actual_manifest_sha256}" != "${EXPECTED_MANIFEST_SHA256}" ]]; then
  echo "evaluation manifest hash mismatch" >&2
  exit 2
fi

for checkpoint in \
  "${CONTINUOUS_CHECKPOINT}" \
  "${DISTRIBUTION_CHECKPOINT}" \
  "${STRUCTURAL_CHECKPOINT}"; do
  test -f "${checkpoint}"
done
if [[ -n "${AUGMENTED_CONTINUOUS_CHECKPOINT:-}" ]]; then
  test -f "${AUGMENTED_CONTINUOUS_CHECKPOINT}"
fi
if [[ -n "${AUGMENTED_RESIDUAL_CHECKPOINT:-}" ]]; then
  test -f "${AUGMENTED_RESIDUAL_CHECKPOINT}"
fi
if [[ -n "${RANKING_CHECKPOINT:-}" ]]; then
  test -f "${RANKING_CHECKPOINT}"
fi

verify_sha256() {
  local path=$1
  local expected=$2
  local actual
  actual=$(sha256sum "${path}" | awk '{print $1}')
  if [[ "${actual}" != "${expected}" ]]; then
    echo "frozen e2e input hash mismatch: ${path}" >&2
    exit 2
  fi
}
verify_sha256 \
  "${E2E_INPUT_ROOT}/parallel/amoeba-static-dse-candidates-v2.jsonl" \
  "${EXPECTED_PARALLEL_MANIFEST_SHA256}"
verify_sha256 \
  "${E2E_INPUT_ROOT}/parallel/amoeba-parallel-query-features-v1.json" \
  "${EXPECTED_PARALLEL_FEATURES_SHA256}"
verify_sha256 \
  "${E2E_INPUT_ROOT}/multi/amoeba-v8-multi-nested-candidates.jsonl" \
  "${EXPECTED_MULTI_MANIFEST_SHA256}"
verify_sha256 \
  "${E2E_INPUT_ROOT}/multi/amoeba-v8-multi-nested-query-features.json" \
  "${EXPECTED_MULTI_FEATURES_SHA256}"

echo "job_id=${SLURM_JOB_ID}"
echo "host=${SLURMD_NODENAME:-unknown}"
echo "manifest_sha256=${actual_manifest_sha256}"
echo "selection_split=validation_only"
echo "output_dir=${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

CHECKPOINT_ARGS=(
  --checkpoint "continuous128=${CONTINUOUS_CHECKPOINT}"
  --checkpoint "residual128=${DISTRIBUTION_CHECKPOINT}"
  --checkpoint "continuous160_struct=${STRUCTURAL_CHECKPOINT}"
)
if [[ -n "${AUGMENTED_CONTINUOUS_CHECKPOINT:-}" ]]; then
  CHECKPOINT_ARGS+=(
    --checkpoint "continuous128_legacy4x4=${AUGMENTED_CONTINUOUS_CHECKPOINT}"
  )
fi
if [[ -n "${AUGMENTED_RESIDUAL_CHECKPOINT:-}" ]]; then
  CHECKPOINT_ARGS+=(
    --checkpoint "residual128_legacy4x4=${AUGMENTED_RESIDUAL_CHECKPOINT}"
  )
fi
if [[ -n "${RANKING_CHECKPOINT:-}" ]]; then
  CHECKPOINT_ARGS+=(
    --checkpoint "residual128_rank01=${RANKING_CHECKPOINT}"
  )
fi

srun --unbuffered "${PYTHON}" adapters/ensemble_pointwise_checkpoints.py \
  --manifest "${MANIFEST}" \
  "${CHECKPOINT_ARGS[@]}" \
  --output "${ENSEMBLE_REPORT}" \
  --batch-size 32 \
  --seed 20260911 \
  --device cuda

srun --unbuffered "${PYTHON}" adapters/benchmark_pointwise_latency.py \
  --manifest "${MANIFEST}" \
  "${CHECKPOINT_ARGS[@]}" \
  --output "${LATENCY_REPORT}" \
  --samples 32 \
  --warmups 20 \
  --repetitions 5 \
  --seed 20260911 \
  --threads 1 \
  --device cuda

srun --unbuffered "${PYTHON}" adapters/compare_legacy_v8_mapper4x4.py \
  --manifest "${MANIFEST}" \
  --legacy-ensemble "${LEGACY_ENSEMBLE}" \
  --v8-ensemble "${ENSEMBLE_REPORT}" \
  --output "${COMPARISON_REPORT}" \
  --seed 20260911 \
  --batch-size 32 \
  --device cuda

# Fit every deployment-safe analytical fallback on validation only.  The
# true-II strata in this report are diagnostic and never participate in rule
# selection.
srun --unbuffered "${PYTHON}" adapters/analyze_pointwise_hybrid.py \
  --manifest "${MANIFEST}" \
  "${CHECKPOINT_ARGS[@]}" \
  --output "${HYBRID_REPORT}" \
  --seed 20260911 \
  --batch-size 32 \
  --device cuda

srun --unbuffered "${PYTHON}" adapters/amoeba_cost_catalog.py \
  --manifest "${E2E_INPUT_ROOT}/parallel/amoeba-static-dse-candidates-v2.jsonl" \
  --analytical-input "${E2E_INPUT_ROOT}/parallel/amoeba-parallel-query-features-v1.json" \
  --task-dfg "Task_0=${E2E_INPUT_ROOT}/parallel/amoeba-task-0-neura.mlir" \
  --task-dfg "Task_1=${E2E_INPUT_ROOT}/parallel/amoeba-task-1-neura.mlir" \
  "${CHECKPOINT_ARGS[@]}" \
  --ensemble-report "${ENSEMBLE_REPORT}" \
  --analytical-hybrid-report "${HYBRID_REPORT}" \
  --output "${PARALLEL_CATALOG}" \
  --timing-output "${PARALLEL_TIMING}" \
  --device cuda

srun --unbuffered "${PYTHON}" adapters/amoeba_cost_catalog.py \
  --manifest "${E2E_INPUT_ROOT}/multi/amoeba-v8-multi-nested-candidates.jsonl" \
  --analytical-input "${E2E_INPUT_ROOT}/multi/amoeba-v8-multi-nested-query-features.json" \
  --task-dfg "Task_0=${E2E_INPUT_ROOT}/multi/dfgs/Task_0.mlir" \
  --task-dfg "Task_1=${E2E_INPUT_ROOT}/multi/dfgs/Task_1.mlir" \
  --task-dfg "Task_2=${E2E_INPUT_ROOT}/multi/dfgs/Task_2.mlir" \
  --task-dfg "Task_3=${E2E_INPUT_ROOT}/multi/dfgs/Task_3.mlir" \
  --task-dfg "Task_4=${E2E_INPUT_ROOT}/multi/dfgs/Task_4.mlir" \
  "${CHECKPOINT_ARGS[@]}" \
  --ensemble-report "${ENSEMBLE_REPORT}" \
  --analytical-hybrid-report "${HYBRID_REPORT}" \
  --output "${MULTI_CATALOG}" \
  --timing-output "${MULTI_TIMING}" \
  --device cuda

sha256sum \
  "${CONTINUOUS_CHECKPOINT}" \
  "${DISTRIBUTION_CHECKPOINT}" \
  "${STRUCTURAL_CHECKPOINT}" \
  "${ENSEMBLE_REPORT}" \
  "${LATENCY_REPORT}" \
  "${COMPARISON_REPORT}" \
  "${HYBRID_REPORT}" \
  "${PARALLEL_CATALOG}" \
  "${PARALLEL_TIMING}" \
  "${MULTI_CATALOG}" \
  "${MULTI_TIMING}"
