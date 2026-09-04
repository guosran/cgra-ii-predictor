#!/bin/bash
#SBATCH --job-name=cgra-cross-attn
#SBATCH --partition=dev
#SBATCH --gres=gpu:nvidia_l20:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/fact_data/yibozhang/cgra-ii-model2/logs/cross-attention-%j.log

set -euo pipefail

PROJECT_ROOT=/fact_home/yibozhang/cgra-ii-predictor
DATA_ROOT=/fact_data/yibozhang/cgra-ii-model2/current
RUN_ROOT=/fact_data/yibozhang/cgra-ii-model2/runs
PYTHON=/fact_home/yibozhang/.conda/envs/awq-vggt/bin/python
MANIFEST=${DATA_ROOT}/training-manifest.json
OUTPUT_DIR=${RUN_ROOT}/cross-attention-${SLURM_JOB_ID}
EXPECTED_MANIFEST_SHA256=74f19a30e3cd2ce843a8f50edd7b8ef97f50df0c952ec71c0fa72011de76a6f9

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=20260906
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

actual_manifest_sha256=$(sha256sum "${MANIFEST}" | awk '{print $1}')
if [[ "${actual_manifest_sha256}" != "${EXPECTED_MANIFEST_SHA256}" ]]; then
  echo "training manifest hash mismatch" >&2
  echo "expected=${EXPECTED_MANIFEST_SHA256}" >&2
  echo "actual=${actual_manifest_sha256}" >&2
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
echo "architecture=candidate_conditioned_cross_attention"
echo "output_dir=${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

srun --unbuffered "${PYTHON}" -c \
  'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} device={torch.cuda.get_device_name(0)}", flush=True)'

srun --unbuffered "${PYTHON}" adapters/neura_graph_experiment.py \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 40 \
  --patience 8 \
  --batch-size 32 \
  --threads "${SLURM_CPUS_PER_TASK}" \
  --seed 20260906 \
  --strict-tiebreak-loss-weight 1.0 \
  --interaction-mode cross_attention \
  --device cuda
