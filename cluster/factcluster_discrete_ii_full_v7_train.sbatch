#!/bin/bash
#SBATCH --job-name=cgra-discrete-full
#SBATCH --partition=dev
#SBATCH --gres=gpu:nvidia_l20:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/fact_data/yibozhang/cgra-ii-model2/logs/discrete-full-v7-%j.log

set -euo pipefail

PROJECT_ROOT=/fact_home/yibozhang/cgra-ii-predictor
DATA_ROOT=/fact_data/yibozhang/cgra-ii-model2/current
RUN_ROOT=/fact_data/yibozhang/cgra-ii-model2/runs
PYTHON=/fact_home/yibozhang/.conda/envs/awq-vggt/bin/python
MANIFEST=${DATA_ROOT}/training-manifest.json
ADDITIONAL_MANIFEST=${DATA_ROOT}/motif-v7-formal-seed-20260906/corpus-manifest.json
OUTPUT_DIR=${RUN_ROOT}/discrete-full-v7-${SLURM_JOB_ID}
EXPECTED_MANIFEST_SHA256=74f19a30e3cd2ce843a8f50edd7b8ef97f50df0c952ec71c0fa72011de76a6f9
EXPECTED_ADDITIONAL_SHA256=9ef9ee0ba648430c6a594a20f3e32fb55038af505560dfc2f9b91d613ea6b062
EPOCHS=${EPOCHS:-40}
PATIENCE=${PATIENCE:-${EPOCHS}}

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
echo "architecture=calibrated_discrete_ii_ranker_full_v7_training_only"
echo "epochs=${EPOCHS}"
echo "patience=${PATIENCE}"
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
  --strict-tiebreak-loss-weight 1.0 \
  --interaction-mode discrete_routing_set \
  --candidate-set-layers 2 \
  --candidate-set-heads 4 \
  --discrete-ii-loss-weight 1.0 \
  --discrete-success-threshold 0.90 \
  --discrete-ii-decision floor \
  --device cuda
