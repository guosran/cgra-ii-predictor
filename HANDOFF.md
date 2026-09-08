# Final model handoff

Last updated: 2026-09-08 (Asia/Hong_Kong)

## Deployment contract

- Predict one continuous compiled II per `(task DFG, mapper shape)`.
- Use `models/final/ensemble.json` and its three checkpoints for DSE ordering.
- The ordinary three-checkpoint ensemble is the only deployment model.
- Enumerate every fixed-orientation task-shape tuple in Amoeba, then retain it
  only if all task rectangles have an exact simultaneous, non-overlapping
  placement on the physical grid. Total area is only a quick necessary check;
  temporal reuse must not make an over-capacity tuple legal.
- Keep concrete placement coordinates in the downstream heuristic for the
  current shape-only phase, but preserve the selected orientation because
  `1xN` and `Nx1` are separately scored mapper shapes.
- Keep task dependencies, fusion/fission decisions, and program scoring in
  Amoeba.
- Treat mapper failures and timeouts as censored outcomes, never as numeric II.
- Keep `mapper_success_probability` diagnostic-only. It must not affect
  support, score, or top-k ordering in this phase.
- The current taskflow decision is to disable resource-aware fusion/fission.

## Frozen results

- Point ensemble validation/test II MAE: `0.335206 / 0.380498`.
- Attention no-fusion DSE: interval `29,360,129`, dependency-DAG makespan
  `62,588,435`; corrected earlier Amoeba makespan `66,782,741`.
- ResNet no-fusion DSE: interval `7,077,891`, dependency-DAG makespan
  `7,119,366`; earlier Amoeba interval `9,437,190`, makespan `18,958,393`.

Machine-readable evidence is in `evaluations/final-model.json` and
`evaluations/dse-vs-legacy-amoeba-2026-09-06.json`.

The final checkpoints are valid only for architecture SHA-256
`f244f15be30604eb32eb96e4837a4bf1ce5c34961c3a46299b90931505cc97e6`.
The graph's historical boundary feature is geometric; it is not a memory-tile
capability mask. The current Amoeba `architecture_with_counter.yaml` has hash
`5c228166de4ceacf49b0ea6286a4a8c1ce45a718ef3da8899a21cea07aa7a174`
and is therefore rejected. Collect mapper labels on that exact
architecture and retrain before enabling its catalog.

TODO: if shortlist mapper failures later become material, evaluate whether the
diagnostic success probability improves top-k recall. Select any policy and
threshold on validation only; do not retrofit it from test outcomes.

## Repository boundary

Detailed local training reports and mapper logs remain untracked and must not
be added accidentally.
