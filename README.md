# CGRA II predictor

This repository contains the final pointwise model used to rank static Amoeba
CGRA shapes. For each `(task DFG, mapper shape)` query it predicts continuous
compiled II. Amoeba owns candidate enumeration, task dependencies, physical
capacity, and program-level scoring. The current candidate protocol retains
exactly the fixed-orientation shape tuples whose task rectangles can all
coexist without overlap on the physical multi-CGRA grid; concrete origins
remain a downstream heuristic choice.

The current contract is shape-only and co-resident: temporal reuse cannot make
an otherwise unpackable tuple legal. TODO: a future analytical
spatial-temporal scheduler may add cross-time tile reuse as a separate search
scope.

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

The frozen validation/test continuous-II MAE is `0.33521 / 0.38050`, measured
only on the 655/649 successful mapper candidates. The corresponding full split
sizes are 1776/1824; failures and timeouts remain censored rather than being
converted into numeric II labels.

The true-mapper policy comparison in
`evaluations/dse-vs-legacy-amoeba-2026-09-06.json` is not evidence of learned
ranking quality: its ResNet run had only one surviving candidate under an older
pruned search policy. It is retained as historical mapper evidence only.

The frozen checkpoints are bound to architecture SHA-256
`f244f15be30604eb32eb96e4837a4bf1ce5c34961c3a46299b90931505cc97e6`.
Catalog generation rejects any other architecture. In particular, Amoeba's
current `architecture_with_counter.yaml` needs newly collected labels and a
retrained model before its rankings are valid.

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

`mapper_success_probability` is exported only for diagnostics. It does not
change query support, predicted II, candidate score, or top-k order. Revisit
this decision only if real mapper failures in the selected shortlist become a
measured problem; any threshold must then be selected on validation data.
