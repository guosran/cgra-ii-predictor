#!/bin/bash
#SBATCH --job-name=cgra-ii-pointwise
#SBATCH --partition=dev
#SBATCH --gres=gpu:nvidia_l20:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/fact_data/yibozhang/cgra-ii-model2/logs/pointwise-v7-%j.log

set -euo pipefail

PROJECT_ROOT=/fact_home/yibozhang/cgra-ii-predictor
DATA_ROOT=/fact_data/yibozhang/cgra-ii-model2/current
RUN_ROOT=/fact_data/yibozhang/cgra-ii-model2/runs
PYTHON=/fact_home/yibozhang/.conda/envs/awq-vggt/bin/python
MANIFEST=${DATA_ROOT}/training-manifest.json
ADDITIONAL_MANIFEST=${DATA_ROOT}/motif-v7-formal-seed-20260906/corpus-manifest.json
PLACEMENT_SUPERVISION=${PLACEMENT_SUPERVISION-${DATA_ROOT}/placement-supervision.json}
OUTPUT_DIR=${RUN_ROOT}/pointwise-v7-${SLURM_JOB_ID}
EXPECTED_MANIFEST_SHA256=74f19a30e3cd2ce843a8f50edd7b8ef97f50df0c952ec71c0fa72011de76a6f9
EXPECTED_ADDITIONAL_SHA256=9ef9ee0ba648430c6a594a20f3e32fb55038af505560dfc2f9b91d613ea6b062
EXPECTED_PLACEMENT_SHA256=9ad9434c4207a77bb5b549b23485a973e7c0168ba4626edc17aa5c547a75aa43
EPOCHS=${EPOCHS:-80}
PATIENCE=${PATIENCE:-${EPOCHS}}
PLACEMENT_LOSS_WEIGHT=${PLACEMENT_LOSS_WEIGHT:-0.1}
INTERACTION_MODE=${INTERACTION_MODE:-discrete_pointwise}

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=20260906
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

actual_manifest_sha256=$(sha256sum "${MANIFEST}" | awk '{print $1}')
actual_additional_sha256=$(sha256sum "${ADDITIONAL_MANIFEST}" | awk '{print $1}')
if [[ "${actual_manifest_sha256}" != "${EXPECTED_MANIFEST_SHA256}" ]]; then
  echo "training manifest hash mismatch" >&2
  exit 2
fi
if [[ "${actual_additional_sha256}" != "${EXPECTED_ADDITIONAL_SHA256}" ]]; then
  echo "additional training manifest hash mismatch" >&2
  exit 2
fi
placement_arguments=()
actual_placement_sha256=none
if [[ -n "${PLACEMENT_SUPERVISION}" ]]; then
  actual_placement_sha256=$(sha256sum "${PLACEMENT_SUPERVISION}" | awk '{print $1}')
  if [[ "${actual_placement_sha256}" != "${EXPECTED_PLACEMENT_SHA256}" ]]; then
    echo "placement supervision hash mismatch" >&2
    exit 2
  fi
  placement_arguments=(
    --placement-supervision "${PLACEMENT_SUPERVISION}"
    --placement-loss-weight "${PLACEMENT_LOSS_WEIGHT}"
  )
elif [[ "${PLACEMENT_LOSS_WEIGHT}" != "0" && "${PLACEMENT_LOSS_WEIGHT}" != "0.0" ]]; then
  echo "placement loss requires placement supervision" >&2
  exit 2
fi

project_revision=unknown
if [[ -r .factcluster-synced-commit ]]; then
  read -r project_revision < .factcluster-synced-commit
elif git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  project_revision=$(git rev-parse HEAD)
fi

echo "job_id=${SLURM_JOB_ID}"
echo "host=${SLURMD_NODENAME:-unknown}"
echo "project_commit=${project_revision}"
echo "manifest_sha256=${actual_manifest_sha256}"
echo "additional_manifest_sha256=${actual_additional_sha256}"
echo "placement_supervision_sha256=${actual_placement_sha256}"
echo "architecture=${INTERACTION_MODE}"
echo "primary_metric=successful_candidate_continuous_ii_mae"
echo "epochs=${EPOCHS}"
echo "patience=${PATIENCE}"
echo "placement_loss_weight=${PLACEMENT_LOSS_WEIGHT}"
echo "output_dir=${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

srun --unbuffered "${PYTHON}" adapters/neura_graph_experiment.py \
  --manifest "${MANIFEST}" \
  --additional-training-manifest "${ADDITIONAL_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch-size 32 \
  --threads "${SLURM_CPUS_PER_TASK}" \
  --seed 20260906 \
  --interaction-mode "${INTERACTION_MODE}" \
  --discrete-ii-loss-weight 1.0 \
  --discrete-success-threshold 0.90 \
  --discrete-ii-decision floor \
  "${placement_arguments[@]}" \
  --device cuda
