#!/bin/bash
#SBATCH --job-name=cgra-ii-v8-hybrid
#SBATCH --partition=dev
#SBATCH --gres=gpu:nvidia_l20:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/fact_data/yibozhang/cgra-ii-model2/logs/pointwise-v8-hybrid-%j.log

set -euo pipefail

PROJECT_ROOT=/fact_home/yibozhang/cgra-ii-predictor
DATA_ROOT=/fact_data/yibozhang/cgra-ii-model2/multi-cgra-v8
EVALUATION_ROOT=/fact_data/yibozhang/cgra-ii-model2/evaluations
PYTHON=/fact_home/yibozhang/.conda/envs/awq-vggt/bin/python
MANIFEST=${DATA_ROOT}/motif-v8-formal-seed-20260911/corpus-manifest.json
: "${EXPECTED_MANIFEST_SHA256:?set EXPECTED_MANIFEST_SHA256}"
: "${SOURCE_EVALUATION_JOB_ID:?set SOURCE_EVALUATION_JOB_ID}"
: "${CONTINUOUS_CHECKPOINT:?set CONTINUOUS_CHECKPOINT}"
: "${DISTRIBUTION_CHECKPOINT:?set DISTRIBUTION_CHECKPOINT}"
: "${STRUCTURAL_CHECKPOINT:?set STRUCTURAL_CHECKPOINT}"

actual_manifest_sha256=$(sha256sum "${MANIFEST}" | awk '{print $1}')
if [[ "${actual_manifest_sha256}" != "${EXPECTED_MANIFEST_SHA256}" ]]; then
  echo "hybrid-analysis manifest hash mismatch" >&2
  exit 2
fi

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

OUTPUT_DIR=${EVALUATION_ROOT}/pointwise-v8-${SOURCE_EVALUATION_JOB_ID}
OUTPUT=${OUTPUT_DIR}/analytical-hybrid.json
mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

echo "job_id=${SLURM_JOB_ID}"
echo "source_evaluation_job_id=${SOURCE_EVALUATION_JOB_ID}"
echo "manifest_sha256=${actual_manifest_sha256}"
echo "selection_split=validation_only"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

srun --unbuffered "${PYTHON}" adapters/analyze_pointwise_hybrid.py \
  --manifest "${MANIFEST}" \
  "${CHECKPOINT_ARGS[@]}" \
  --output "${OUTPUT}" \
  --seed 20260911 \
  --batch-size 32 \
  --device cuda

sha256sum "${OUTPUT}"
