#!/bin/bash
#SBATCH --job-name=cgra-ii-v8-train
#SBATCH --partition=dev
#SBATCH --gres=gpu:nvidia_l20:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/fact_data/yibozhang/cgra-ii-model2/logs/pointwise-v8-%j.log

set -euo pipefail

PROJECT_ROOT=/fact_home/yibozhang/cgra-ii-predictor
DATA_ROOT=/fact_data/yibozhang/cgra-ii-model2/multi-cgra-v8
RUN_ROOT=/fact_data/yibozhang/cgra-ii-model2/runs
PYTHON=/fact_home/yibozhang/.conda/envs/awq-vggt/bin/python
MANIFEST=${DATA_ROOT}/motif-v8-formal-seed-20260911/corpus-manifest.json
PLACEMENT_SUPERVISION=${PLACEMENT_SUPERVISION-${DATA_ROOT}/placement-supervision.json}
RUN_LABEL=${RUN_LABEL:-model}
OUTPUT_DIR=${RUN_ROOT}/pointwise-v8-${RUN_LABEL}-${SLURM_JOB_ID}
: "${EXPECTED_MANIFEST_SHA256:?set EXPECTED_MANIFEST_SHA256 after collection}"
: "${EXPECTED_PLACEMENT_SHA256:?set EXPECTED_PLACEMENT_SHA256 after collection}"

EPOCHS=${EPOCHS:-80}
PATIENCE=${PATIENCE:-${EPOCHS}}
PLACEMENT_LOSS_WEIGHT=${PLACEMENT_LOSS_WEIGHT:-0.1}
INTERACTION_MODE=${INTERACTION_MODE:-residual_pointwise}
DFG_REPRESENTATION=${DFG_REPRESENTATION:-route_expanded_v2}
DFG_MESSAGE_MODE=${DFG_MESSAGE_MODE:-dual_mean}
DFG_POOL_MODE=${DFG_POOL_MODE:-all}
DFG_SUMMARY_MODE=${DFG_SUMMARY_MODE:-none}
LEARNING_RATE=${LEARNING_RATE:-1e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
BATCH_SIZE=${BATCH_SIZE:-16}
HIDDEN_DIMENSION=${HIDDEN_DIMENSION:-128}
MESSAGE_PASSING_LAYERS=${MESSAGE_PASSING_LAYERS:-9}
DROPOUT=${DROPOUT:-0.10}
LEARNING_RATE_SCHEDULE=${LEARNING_RATE_SCHEDULE:-none}
MINIMUM_LEARNING_RATE=${MINIMUM_LEARNING_RATE:-1e-5}
DISCRETE_II_LOSS_WEIGHT=${DISCRETE_II_LOSS_WEIGHT:-1.0}
PAIRWISE_RANKING_LOSS_WEIGHT=${PAIRWISE_RANKING_LOSS_WEIGHT:-0.0}
PAIRWISE_RANKING_MARGIN=${PAIRWISE_RANKING_MARGIN:-0.5}
TRAINING_SEED=${TRAINING_SEED:-20260911}
LEGACY_MAPPER4X4_TRAINING_MANIFEST=${LEGACY_MAPPER4X4_TRAINING_MANIFEST:-}
LEGACY_MAPPER4X4_TRAINING_FRACTION=${LEGACY_MAPPER4X4_TRAINING_FRACTION:-0.35}
ROUTING_PEAK_NORMALIZATION=${ROUTING_PEAK_NORMALIZATION:-protocol_max_tiles_v1}

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=20260911
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

actual_manifest_sha256=$(sha256sum "${MANIFEST}" | awk '{print $1}')
actual_placement_sha256=$(sha256sum "${PLACEMENT_SUPERVISION}" | awk '{print $1}')
if [[ "${actual_manifest_sha256}" != "${EXPECTED_MANIFEST_SHA256}" ]]; then
  echo "training manifest hash mismatch" >&2
  exit 2
fi
if [[ "${actual_placement_sha256}" != "${EXPECTED_PLACEMENT_SHA256}" ]]; then
  echo "placement supervision hash mismatch" >&2
  exit 2
fi

echo "job_id=${SLURM_JOB_ID}"
echo "host=${SLURMD_NODENAME:-unknown}"
echo "manifest_sha256=${actual_manifest_sha256}"
echo "placement_supervision_sha256=${actual_placement_sha256}"
echo "shape_protocol=amoeba-static-rectangles-4x4-tiles-v1"
echo "routing_peak_normalization=${ROUTING_PEAK_NORMALIZATION}"
echo "selection_split=validation_only"
echo "primary_metric=successful_candidate_continuous_ii_mae"
echo "interaction_mode=${INTERACTION_MODE}"
echo "dfg_representation=${DFG_REPRESENTATION}"
echo "dfg_message_mode=${DFG_MESSAGE_MODE}"
echo "hidden_dimension=${HIDDEN_DIMENSION}"
echo "message_passing_layers=${MESSAGE_PASSING_LAYERS}"
echo "pairwise_ranking_loss_weight=${PAIRWISE_RANKING_LOSS_WEIGHT}"
echo "pairwise_ranking_margin=${PAIRWISE_RANKING_MARGIN}"
echo "training_seed=${TRAINING_SEED}"
echo "run_label=${RUN_LABEL}"
echo "output_dir=${OUTPUT_DIR}"
echo "legacy_mapper4x4_training_manifest=${LEGACY_MAPPER4X4_TRAINING_MANIFEST:-none}"
echo "legacy_mapper4x4_training_fraction=${LEGACY_MAPPER4X4_TRAINING_FRACTION}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

PLACEMENT_ARGS=()
if [[ "${PLACEMENT_LOSS_WEIGHT}" != "0" && "${PLACEMENT_LOSS_WEIGHT}" != "0.0" ]]; then
  PLACEMENT_ARGS=(
    --placement-supervision "${PLACEMENT_SUPERVISION}"
    --placement-loss-weight "${PLACEMENT_LOSS_WEIGHT}"
  )
fi
LEGACY_ARGS=()
if [[ -n "${LEGACY_MAPPER4X4_TRAINING_MANIFEST}" ]]; then
  LEGACY_ARGS=(
    --legacy-mapper4x4-training-manifest \
      "${LEGACY_MAPPER4X4_TRAINING_MANIFEST}"
    --legacy-mapper4x4-training-fraction \
      "${LEGACY_MAPPER4X4_TRAINING_FRACTION}"
  )
fi

srun --unbuffered "${PYTHON}" adapters/neura_graph_experiment.py \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch-size "${BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --learning-rate-schedule "${LEARNING_RATE_SCHEDULE}" \
  --minimum-learning-rate "${MINIMUM_LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --threads "${SLURM_CPUS_PER_TASK}" \
  --seed 20260911 \
  --training-seed "${TRAINING_SEED}" \
  --interaction-mode "${INTERACTION_MODE}" \
  --dfg-representation "${DFG_REPRESENTATION}" \
  --dfg-message-mode "${DFG_MESSAGE_MODE}" \
  --dfg-pool-mode "${DFG_POOL_MODE}" \
  --dfg-summary-mode "${DFG_SUMMARY_MODE}" \
  --shape-protocol amoeba-static-rectangles-4x4-tiles-v1 \
  --routing-peak-normalization "${ROUTING_PEAK_NORMALIZATION}" \
  --hidden-dimension "${HIDDEN_DIMENSION}" \
  --message-passing-layers "${MESSAGE_PASSING_LAYERS}" \
  --dropout "${DROPOUT}" \
  --discrete-ii-loss-weight "${DISCRETE_II_LOSS_WEIGHT}" \
  --pairwise-ranking-loss-weight "${PAIRWISE_RANKING_LOSS_WEIGHT}" \
  --pairwise-ranking-margin "${PAIRWISE_RANKING_MARGIN}" \
  --discrete-success-threshold 0.5 \
  --discrete-ii-decision floor \
  "${PLACEMENT_ARGS[@]}" \
  "${LEGACY_ARGS[@]}" \
  --device cuda

sha256sum "${OUTPUT_DIR}/model.pt" "${OUTPUT_DIR}/report.json"
