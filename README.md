# CGRA II predictor

This repository contains the final pointwise model used to rank static Amoeba
CGRA shapes. For each `(task DFG, mapper shape)` query it predicts continuous
compiled II. Amoeba retains ownership of candidate generation, exact rectangle
packing, task dependencies, and program-level scoring.

The deployed predictor is a validation-selected uncertainty-gated ensemble:

- `models/final/large-operation.pt`
- `models/final/baseline.pt`
- `models/final/ranking.pt`
- `models/final/ensemble.json`

This ordinary ensemble is the sole deployment model and supplies the shape
ordering score directly.

## Model

The model uses a route-expanded Neura DFG, an oriented mapper-tile mesh, nine
dual-mean message-passing layers, candidate-conditioned operation-to-PE
attention, routing context, and residual pointwise II prediction. The primary
checkpoints use 128 hidden channels. The ranking checkpoints add pairwise
supervision so that the relative order of shapes is trained directly.

The finite physical-to-mapper shape domain is:

```text
1x1 -> 4x4    1x2 -> 4x8    2x1 -> 8x4    1x3 -> 4x12
3x1 -> 12x4  1x4 -> 4x16   2x2 -> 8x8    4x1 -> 16x4
```

## Results

The frozen validation/test continuous-II MAE is `0.33521 / 0.38050`.

The same-architecture true-mapper DSE comparison is recorded in
`evaluations/dse-vs-legacy-amoeba-2026-09-06.json`. With fusion/fission
disabled, ResNet's selected allocation reduces the objective interval by 25%
relative to the earlier Amoeba allocation. Attention has the same steady-state
interval and a 6.28% lower corrected dependency-DAG makespan.

## Usage

Install PyTorch and run the cost adapter from the repository root. It uses the
three checkpoints and `models/final/ensemble.json` by default:

```sh
python3 adapters/amoeba_cost_catalog.py \
  --manifest candidates.jsonl \
  --analytical-input analytical.json \
  --task-dfg TASK=task.mlir \
  --output costs.json
```

Run the repository checks with:

```sh
python3 -m pytest -q
```

Amoeba's current external JSON identifiers retain their upstream names for
wire compatibility. They are not model revisions.
