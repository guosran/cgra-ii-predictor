#!/bin/bash
#SBATCH --job-name=cgra-ii-v9-eval
#SBATCH --partition=dev
#SBATCH --gres=gpu:nvidia_l20:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/fact_data/yibozhang/cgra-ii-model2/logs/pointwise-v9-eval-%j.log

set -euo pipefail

PROJECT_ROOT=/fact_home/yibozhang/cgra-ii-predictor
DATA_ROOT=/fact_data/yibozhang/cgra-ii-model2
PYTHON=/fact_home/yibozhang/.conda/envs/awq-vggt/bin/python
MANIFEST=${DATA_ROOT}/multi-cgra-v8/motif-v8-formal-seed-20260911/corpus-manifest.json
OUTPUT=${DATA_ROOT}/evaluations/pointwise-v8-v9-largeop-${SLURM_JOB_ID}.json
RESIDUAL_CHECKPOINT=${DATA_ROOT}/runs/pointwise-v8-residual128-p64-s12-1478199/model.pt
RANKING_CHECKPOINT=${DATA_ROOT}/runs/pointwise-v8-residual128-rank01-s12-1478681/model.pt
: "${MODEL_CHECKPOINT:?set MODEL_CHECKPOINT}"
: "${EXPECTED_MODEL_SHA256:?set EXPECTED_MODEL_SHA256}"

cd "${PROJECT_ROOT}"
actual_model_sha256=$(sha256sum "${MODEL_CHECKPOINT}" | awk '{print $1}')
actual_manifest_sha256=$(sha256sum "${MANIFEST}" | awk '{print $1}')
actual_residual_sha256=$(sha256sum "${RESIDUAL_CHECKPOINT}" | awk '{print $1}')
actual_ranking_sha256=$(sha256sum "${RANKING_CHECKPOINT}" | awk '{print $1}')
if [[ "${actual_model_sha256}" != "${EXPECTED_MODEL_SHA256}" ]]; then
  echo "model checkpoint hash mismatch" >&2
  exit 2
fi
if [[ "${actual_manifest_sha256}" != "33ef4a2dc8385b8245234cb4cc95817ab321c35291a2663f0f09103f0888823f" ]]; then
  echo "frozen v8 manifest hash mismatch" >&2
  exit 2
fi
if [[ "${actual_residual_sha256}" != "621e2c9fbdad63f5bc44467fbf6483d6793a113b5482f590b47e406680f4883f" ]]; then
  echo "frozen residual checkpoint hash mismatch" >&2
  exit 2
fi
if [[ "${actual_ranking_sha256}" != "ae9273e2bec0c864b4dce26369cefc8a20ecac4314f76bc0e2277b8b6238cd75" ]]; then
  echo "frozen ranking checkpoint hash mismatch" >&2
  exit 2
fi

echo "job_id=${SLURM_JOB_ID}"
echo "selection_split=validation_only"
echo "attention_labels_used=false"
echo "model_checkpoint=${MODEL_CHECKPOINT}"
echo "model_sha256=${actual_model_sha256}"

srun --unbuffered "${PYTHON}" adapters/ensemble_pointwise_checkpoints.py \
  --manifest "${MANIFEST}" \
  --checkpoint "residual128=${RESIDUAL_CHECKPOINT}" \
  --checkpoint "residual128_rank01=${RANKING_CHECKPOINT}" \
  --checkpoint "largeop_rank01=${MODEL_CHECKPOINT}" \
  --output "${OUTPUT}" \
  --batch-size 32 \
  --seed 20260911 \
  --device cuda

sha256sum "${OUTPUT}"
