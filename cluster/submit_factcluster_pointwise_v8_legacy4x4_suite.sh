#!/bin/bash
# Add only genuine legacy mapper-4x4 labels to the formal v8 training split.

set -euo pipefail

MANIFEST_SHA256=33ef4a2dc8385b8245234cb4cc95817ab321c35291a2663f0f09103f0888823f
PLACEMENT_SHA256=71f00fe2d29895704ea6edbc352c9f3adf2caad04fc39749e07519b4b16bebc3
LEGACY_MANIFEST=/fact_data/yibozhang/cgra-ii-model2/current/training-manifest.json
TRAIN_SCRIPT=cluster/factcluster_pointwise_v8_train.sbatch
EVALUATION_SCRIPT=cluster/factcluster_pointwise_v8_evaluate.sbatch
COMMON="ALL,EXPECTED_MANIFEST_SHA256=${MANIFEST_SHA256},EXPECTED_PLACEMENT_SHA256=${PLACEMENT_SHA256},EPOCHS=80,PATIENCE=80,BATCH_SIZE=16,DFG_REPRESENTATION=route_expanded_v2,DFG_MESSAGE_MODE=dual_mean,DFG_POOL_MODE=all,HIDDEN_DIMENSION=128,MESSAGE_PASSING_LAYERS=9,PLACEMENT_LOSS_WEIGHT=0.1,LEGACY_MAPPER4X4_TRAINING_MANIFEST=${LEGACY_MANIFEST},LEGACY_MAPPER4X4_TRAINING_FRACTION=0.35"

continuous=$(sbatch --parsable \
  --export="${COMMON},RUN_LABEL=continuous128-legacy4x4-s14,INTERACTION_MODE=continuous_residual_pointwise,TRAINING_SEED=20260914,LEARNING_RATE_SCHEDULE=cosine" \
  "${TRAIN_SCRIPT}")
residual=$(sbatch --parsable \
  --export="${COMMON},RUN_LABEL=residual128-legacy4x4-s15,INTERACTION_MODE=residual_pointwise,TRAINING_SEED=20260915,LEARNING_RATE_SCHEDULE=none" \
  "${TRAIN_SCRIPT}")

evaluation=$(sbatch --parsable \
  --dependency="afterok:${continuous}:${residual}" \
  --export="ALL,EXPECTED_MANIFEST_SHA256=${MANIFEST_SHA256},CONTINUOUS_CHECKPOINT=/fact_data/yibozhang/cgra-ii-model2/runs/pointwise-v8-continuous128-s11-1478077/model.pt,DISTRIBUTION_CHECKPOINT=/fact_data/yibozhang/cgra-ii-model2/runs/pointwise-v8-residual128-s12-1478078/model.pt,STRUCTURAL_CHECKPOINT=/fact_data/yibozhang/cgra-ii-model2/runs/pointwise-v8-continuous160-struct-s13-1478079/model.pt,AUGMENTED_CONTINUOUS_CHECKPOINT=/fact_data/yibozhang/cgra-ii-model2/runs/pointwise-v8-continuous128-legacy4x4-s14-${continuous}/model.pt,AUGMENTED_RESIDUAL_CHECKPOINT=/fact_data/yibozhang/cgra-ii-model2/runs/pointwise-v8-residual128-legacy4x4-s15-${residual}/model.pt" \
  "${EVALUATION_SCRIPT}")

printf 'continuous_legacy4x4=%s\nresidual_legacy4x4=%s\nevaluation=%s\n' \
  "${continuous}" "${residual}" "${evaluation}"
