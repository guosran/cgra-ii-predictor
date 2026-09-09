# Final model handoff

Last updated: 2026-09-09 (Asia/Hong_Kong)

## Deployment contract

- Use only `models/final/mapper.pt` and `models/final/model.json`.
- Predict one continuous heuristic-mapper `compiled_ii` per pre-mapper task DFG
  and mapper shape. Do not consume mapped artifacts as model inputs.
- Rank shapes directly by predicted II. The model has no placement, routing,
  uncertainty, or mapper-success output.
- Keep mapper failures and timeouts censored; confirm selected candidates with
  the real mapper and fall back to the next candidate on failure.
- Enumerate fixed-orientation shape tuples in Amoeba and retain only tuples
  whose task rectangles have an exact simultaneous non-overlapping placement.
- Keep concrete origins, dependencies, program scoring, and top-k selection in
  Amoeba. Fusion/fission remains disabled.

## Frozen model

- Architecture: standardized 112-value input, `64 -> 32` GELU hidden layers,
  scalar residual output; 9,345 trainable parameters.
- Target: final `compiled_ii` from the fixed Neura heuristic mapper.
- Training inputs: pre-mapper DFG structure, mapper shape, RecMII, ResMII, and
  their lower bound; no mapped-result feature or placement supervision.
- Successful split counts: 3,138 train / 655 validation / 649 test.
- Validation: MAE `0.430708`, pairwise `0.738333`, top-1 hit `0.866071`, mean
  regret `0.187500`.
- Test: MAE `0.438052`, pairwise `0.778146`, top-1 hit `0.807339`, mean regret
  `0.247706`.

On the identical split, the removed graph ensemble achieved validation/test
MAE `0.335206 / 0.380498`, pairwise `0.728333 / 0.740066`, top-1 hit
`0.803571 / 0.752294`, and regret `0.276786 / 0.394495`. The direct model was
selected because DSE ordering and regret take precedence over point MAE.

Machine-readable evidence is in `evaluations/final-model.json`. Historical
true-mapper DSE comparisons remain in
`evaluations/dse-vs-legacy-amoeba-2026-09-06.json`.

## Architecture boundary

The checkpoint supports only architecture SHA-256
`f244f15be30604eb32eb96e4837a4bf1ce5c34961c3a46299b90931505cc97e6`.
The current Amoeba counter architecture has SHA-256
`5c228166de4ceacf49b0ea6286a4a8c1ce45a718ef3da8899a21cea07aa7a174`
and is rejected. Collect mapper labels and retrain before enabling its catalog.

Local training reports and mapper logs are ignored and must remain untracked.
