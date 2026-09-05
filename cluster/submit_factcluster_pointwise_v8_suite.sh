#!/bin/bash
# Submit three validation-selected v8 pointwise candidates after label collection.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 MANIFEST_SHA256 PLACEMENT_SHA256" >&2
  exit 2
fi

MANIFEST_SHA256=$1
PLACEMENT_SHA256=$2
SCRIPT=cluster/factcluster_pointwise_v8_train.sbatch
EVALUATION_SCRIPT=cluster/factcluster_pointwise_v8_evaluate.sbatch
COMMON="ALL,EXPECTED_MANIFEST_SHA256=${MANIFEST_SHA256},EXPECTED_PLACEMENT_SHA256=${PLACEMENT_SHA256},EPOCHS=80,PATIENCE=80,BATCH_SIZE=16,DFG_REPRESENTATION=route_expanded_v2,DFG_MESSAGE_MODE=dual_mean,DFG_POOL_MODE=all"

continuous=$(sbatch --parsable --export="${COMMON},RUN_LABEL=continuous128-p64-s11,INTERACTION_MODE=continuous_residual_pointwise,HIDDEN_DIMENSION=128,MESSAGE_PASSING_LAYERS=9,PLACEMENT_LOSS_WEIGHT=0.1,TRAINING_SEED=20260911,LEARNING_RATE_SCHEDULE=cosine" "${SCRIPT}")
distribution=$(sbatch --parsable --export="${COMMON},RUN_LABEL=residual128-p64-s12,INTERACTION_MODE=residual_pointwise,HIDDEN_DIMENSION=128,MESSAGE_PASSING_LAYERS=9,PLACEMENT_LOSS_WEIGHT=0.1,TRAINING_SEED=20260912,LEARNING_RATE_SCHEDULE=none" "${SCRIPT}")
structural=$(sbatch --parsable --export="${COMMON},RUN_LABEL=continuous160-struct-p64-s13,INTERACTION_MODE=continuous_residual_pointwise,HIDDEN_DIMENSION=160,MESSAGE_PASSING_LAYERS=7,PLACEMENT_LOSS_WEIGHT=0,TRAINING_SEED=20260913,DFG_SUMMARY_MODE=structural_v1,LEARNING_RATE_SCHEDULE=cosine" "${SCRIPT}")
evaluation=$(sbatch --parsable \
  --dependency="afterok:${continuous}:${distribution}:${structural}" \
  --export="ALL,EXPECTED_MANIFEST_SHA256=${MANIFEST_SHA256},CONTINUOUS_CHECKPOINT=/fact_data/yibozhang/cgra-ii-model2/runs/pointwise-v8-continuous128-p64-s11-${continuous}/model.pt,DISTRIBUTION_CHECKPOINT=/fact_data/yibozhang/cgra-ii-model2/runs/pointwise-v8-residual128-p64-s12-${distribution}/model.pt,STRUCTURAL_CHECKPOINT=/fact_data/yibozhang/cgra-ii-model2/runs/pointwise-v8-continuous160-struct-p64-s13-${structural}/model.pt" \
  "${EVALUATION_SCRIPT}")

printf 'continuous=%s\ndistribution=%s\nstructural=%s\nevaluation=%s\n' \
  "${continuous}" "${distribution}" "${structural}" "${evaluation}"
