#!/bin/bash
#SBATCH --job-name=cgra-ii-v9-train
#SBATCH --partition=dev
#SBATCH --gres=gpu:nvidia_l20:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/fact_data/yibozhang/cgra-ii-model2/logs/pointwise-v9-%j.log

set -euo pipefail

PROJECT_ROOT=/fact_home/yibozhang/cgra-ii-predictor
V8_ROOT=/fact_data/yibozhang/cgra-ii-model2/multi-cgra-v8
V9_ROOT=/fact_data/yibozhang/cgra-ii-model2/multi-cgra-v9
RUN_ROOT=/fact_data/yibozhang/cgra-ii-model2/runs
PYTHON=/fact_home/yibozhang/.conda/envs/awq-vggt/bin/python
V8_MANIFEST=${V8_ROOT}/motif-v8-formal-seed-20260911/corpus-manifest.json
V9_MANIFEST=${V9_ROOT}/motif-v9-largeop-seed-20260913-r3/corpus-manifest.json
PLACEMENT=${V9_ROOT}/placement-supervision-v8-v9.json
EXPECTED_V8_MANIFEST_SHA256=33ef4a2dc8385b8245234cb4cc95817ab321c35291a2663f0f09103f0888823f
: "${EXPECTED_V9_MANIFEST_SHA256:?set EXPECTED_V9_MANIFEST_SHA256 after collection}"
: "${EXPECTED_PLACEMENT_SHA256:?set EXPECTED_PLACEMENT_SHA256 after collection}"

RUN_LABEL=${RUN_LABEL:-residual128-rank01-s12}
OUTPUT_DIR=${RUN_ROOT}/pointwise-v8-v9-largeop-${RUN_LABEL}-${SLURM_JOB_ID}
INTERACTION_MODE=${INTERACTION_MODE:-residual_pointwise}
HIDDEN_DIMENSION=${HIDDEN_DIMENSION:-128}
MESSAGE_PASSING_LAYERS=${MESSAGE_PASSING_LAYERS:-9}
PAIRWISE_RANKING_LOSS_WEIGHT=${PAIRWISE_RANKING_LOSS_WEIGHT:-0.1}
TRAINING_SEED=${TRAINING_SEED:-20260912}
PLACEMENT_LOSS_WEIGHT=${PLACEMENT_LOSS_WEIGHT:-0.1}

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=20260911
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

verify_sha256() {
  local path=$1
  local expected=$2
  local actual
  actual=$(sha256sum "${path}" | awk '{print $1}')
  if [[ "${actual}" != "${expected}" ]]; then
    echo "input hash mismatch: ${path}" >&2
    exit 2
  fi
}
verify_sha256 "${V8_MANIFEST}" "${EXPECTED_V8_MANIFEST_SHA256}"
verify_sha256 "${V9_MANIFEST}" "${EXPECTED_V9_MANIFEST_SHA256}"
verify_sha256 "${PLACEMENT}" "${EXPECTED_PLACEMENT_SHA256}"

echo "job_id=${SLURM_JOB_ID}"
echo "formal_selection_and_test_manifest=${V8_MANIFEST}"
echo "additional_training_only_manifest=${V9_MANIFEST}"
echo "attention_labels_used=false"
echo "interaction_mode=${INTERACTION_MODE}"
echo "pairwise_ranking_loss_weight=${PAIRWISE_RANKING_LOSS_WEIGHT}"
echo "training_seed=${TRAINING_SEED}"
echo "output_dir=${OUTPUT_DIR}"

srun --unbuffered "${PYTHON}" adapters/neura_graph_experiment.py \
  --manifest "${V8_MANIFEST}" \
  --additional-training-manifest "${V9_MANIFEST}" \
  --placement-supervision "${PLACEMENT}" \
  --placement-loss-weight "${PLACEMENT_LOSS_WEIGHT}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 80 \
  --patience 80 \
  --batch-size 16 \
  --learning-rate 1e-3 \
  --learning-rate-schedule none \
  --minimum-learning-rate 1e-5 \
  --weight-decay 1e-4 \
  --threads "${SLURM_CPUS_PER_TASK}" \
  --seed 20260911 \
  --training-seed "${TRAINING_SEED}" \
  --interaction-mode "${INTERACTION_MODE}" \
  --dfg-representation route_expanded_v2 \
  --dfg-message-mode dual_mean \
  --dfg-pool-mode all \
  --dfg-summary-mode none \
  --shape-protocol amoeba-static-rectangles-4x4-tiles-v1 \
  --routing-peak-normalization protocol_max_tiles_v1 \
  --hidden-dimension "${HIDDEN_DIMENSION}" \
  --message-passing-layers "${MESSAGE_PASSING_LAYERS}" \
  --dropout 0.10 \
  --discrete-ii-loss-weight 1.0 \
  --pairwise-ranking-loss-weight "${PAIRWISE_RANKING_LOSS_WEIGHT}" \
  --pairwise-ranking-margin 0.5 \
  --discrete-success-threshold 0.5 \
  --discrete-ii-decision floor \
  --device cuda

sha256sum "${OUTPUT_DIR}/model.pt" "${OUTPUT_DIR}/report.json"
