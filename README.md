# CGRA heuristic-mapper II predictor

This repository contains one compact surrogate for the fixed Neura heuristic
mapper. For each `(pre-mapper task DFG, mapper shape)` query, it predicts only
the final `compiled_ii` that the mapper would return. It does not predict an
ideal placement, construct a soft placement, or use mapped output as an input.

The only deployed model artifacts are:

- `models/final/mapper.pt`
- `models/final/model.json`

## Model contract

The input is a 112-value vector computed before mapping:

- 76 DFG summaries: log counts of 45 route-expanded operation types, mean and
  maximum of 14 node properties, and node/raw-edge/semantic-edge counts;
- 13 topology summaries: materialized and movement counts, forward depths and
  width, sources/sinks, degree and density, and recurrence-node count;
- 23 query summaries: rows, columns, tiles, aspect ratio, links, bisection,
  RecMII, ResMII, lower bound, an exact one-hot encoding of the eight shapes,
  and six operation/edge-to-capacity interactions.

The network standardizes these values using training-split statistics and
applies `112 -> 64 -> 32 -> 1` fully connected layers with GELU activations.
It has 9,345 trainable parameters. The scalar is interpreted as a non-negative
residual above `max(RecMII, ResMII)` and clamped to the mapper search ceiling of
20. The output is one continuous predicted II; there is no placement, routing,
uncertainty, or success head.

The finite shape domain is:

```text
1x1 -> 4x4    1x2 -> 4x8    2x1 -> 8x4    1x3 -> 4x12
3x1 -> 12x4  1x4 -> 4x16   2x2 -> 8x8    4x1 -> 16x4
```

## Training and selection

Training reads only each candidate's pre-mapper `input.mlir`, analytical
bounds, requested shape, and real heuristic-mapper `compiled_ii` label. Mapped
artifacts are never read as features. Failures and timeouts have no numeric II
and remain censored.

The split is deterministic, family-stratified, and grouped by
`ranking_query_id`, which is one-to-one with leakage lineage. The successful
candidate counts are 3,138 train, 655 validation, and 649 test. The loss is
Smooth L1 on final II plus pairwise shape ranking (weight 0.1) and query-level
top-1 selection (weight 0.3). Checkpoints are selected only on validation,
prioritizing top-1 regret, top-1 hit rate, pairwise accuracy, then MAE.

Against the removed 10,576,581-parameter graph ensemble on the exact same
corpus and split:

| Split | Model | MAE | Pairwise | Top-1 hit | Mean regret |
|---|---|---:|---:|---:|---:|
| Validation | Removed graph ensemble | 0.3352 | 0.7283 | 0.8036 | 0.2768 |
| Validation | Direct mapper model | 0.4307 | 0.7383 | 0.8661 | 0.1875 |
| Test | Removed graph ensemble | 0.3805 | 0.7401 | 0.7523 | 0.3945 |
| Test | Direct mapper model | 0.4381 | 0.7781 | 0.8073 | 0.2477 |

The direct model trades some pointwise MAE for consistently better shape
ordering and substantially lower selection regret, which is the DSE priority.
It is about 1,132 times smaller than the removed ensemble.

## Architecture boundary

The checkpoint is valid only for architecture SHA-256
`f244f15be30604eb32eb96e4837a4bf1ce5c34961c3a46299b90931505cc97e6`.
Catalog generation rejects any other architecture. Amoeba's current
`architecture_with_counter.yaml` has SHA-256
`5c228166de4ceacf49b0ea6286a4a8c1ce45a718ef3da8899a21cea07aa7a174`,
so it needs labels collected on that exact architecture and a retrained model
before deployment.

## Usage

Generate a cost catalog:

```sh
python3 adapters/amoeba_cost_catalog.py \
  --manifest candidates.jsonl \
  --analytical-input analytical.json \
  --task-dfg TASK=task.mlir \
  --output costs.json
```

Reproduce training with the selected defaults:

```sh
python3 adapters/train_mapper_surrogate.py \
  --manifest /path/to/corpus-manifest.json \
  --output-dir reports/direct-mapper
```

Run repository checks with `python3 -m pytest -q`.

Amoeba owns candidate enumeration, co-resident rectangle packing, task
dependencies, and program-level scoring. Fusion/fission remains disabled. The
predictor does not estimate mapper failure probability; shortlist candidates
must still be confirmed by the real mapper.
