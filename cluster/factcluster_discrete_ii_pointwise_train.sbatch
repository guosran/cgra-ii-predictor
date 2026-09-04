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
DFG_REPRESENTATION=${DFG_REPRESENTATION:-semantic_v1}
DFG_MESSAGE_MODE=${DFG_MESSAGE_MODE:-sum}
DFG_POOL_MODE=${DFG_POOL_MODE:-all}
DFG_SUMMARY_MODE=${DFG_SUMMARY_MODE:-none}
LEARNING_RATE=${LEARNING_RATE:-1e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
BATCH_SIZE=${BATCH_SIZE:-32}
HIDDEN_DIMENSION=${HIDDEN_DIMENSION:-64}
MESSAGE_PASSING_LAYERS=${MESSAGE_PASSING_LAYERS:-3}
DROPOUT=${DROPOUT:-0.10}
LEARNING_RATE_SCHEDULE=${LEARNING_RATE_SCHEDULE:-none}
MINIMUM_LEARNING_RATE=${MINIMUM_LEARNING_RATE:-1e-5}
DISCRETE_II_LOSS_WEIGHT=${DISCRETE_II_LOSS_WEIGHT:-1.0}
TRAINING_SEED=${TRAINING_SEED:-20260906}
TRAINING_FRACTION=${TRAINING_FRACTION:-1.0}

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
echo "dfg_representation=${DFG_REPRESENTATION}"
echo "dfg_message_mode=${DFG_MESSAGE_MODE}"
echo "dfg_pool_mode=${DFG_POOL_MODE}"
echo "dfg_summary_mode=${DFG_SUMMARY_MODE}"
echo "primary_metric=successful_candidate_continuous_ii_mae"
echo "epochs=${EPOCHS}"
echo "patience=${PATIENCE}"
echo "placement_loss_weight=${PLACEMENT_LOSS_WEIGHT}"
echo "learning_rate=${LEARNING_RATE}"
echo "weight_decay=${WEIGHT_DECAY}"
echo "batch_size=${BATCH_SIZE}"
echo "hidden_dimension=${HIDDEN_DIMENSION}"
echo "message_passing_layers=${MESSAGE_PASSING_LAYERS}"
echo "dropout=${DROPOUT}"
echo "learning_rate_schedule=${LEARNING_RATE_SCHEDULE}"
echo "minimum_learning_rate=${MINIMUM_LEARNING_RATE}"
echo "discrete_ii_loss_weight=${DISCRETE_II_LOSS_WEIGHT}"
echo "split_seed=20260906"
echo "training_seed=${TRAINING_SEED}"
echo "training_fraction=${TRAINING_FRACTION}"
echo "output_dir=${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

srun --unbuffered "${PYTHON}" adapters/neura_graph_experiment.py \
  --manifest "${MANIFEST}" \
  --additional-training-manifest "${ADDITIONAL_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch-size "${BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --learning-rate-schedule "${LEARNING_RATE_SCHEDULE}" \
  --minimum-learning-rate "${MINIMUM_LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --threads "${SLURM_CPUS_PER_TASK}" \
  --seed 20260906 \
  --training-seed "${TRAINING_SEED}" \
  --training-fraction "${TRAINING_FRACTION}" \
  --interaction-mode "${INTERACTION_MODE}" \
  --dfg-representation "${DFG_REPRESENTATION}" \
  --dfg-message-mode "${DFG_MESSAGE_MODE}" \
  --dfg-pool-mode "${DFG_POOL_MODE}" \
  --dfg-summary-mode "${DFG_SUMMARY_MODE}" \
  --hidden-dimension "${HIDDEN_DIMENSION}" \
  --message-passing-layers "${MESSAGE_PASSING_LAYERS}" \
  --dropout "${DROPOUT}" \
  --discrete-ii-loss-weight "${DISCRETE_II_LOSS_WEIGHT}" \
  --discrete-success-threshold 0.90 \
  --discrete-ii-decision floor \
  "${placement_arguments[@]}" \
  --device cuda
