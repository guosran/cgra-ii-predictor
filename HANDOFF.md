# Final model handoff

Last updated: 2026-09-06 (Asia/Hong_Kong)

## Deployment contract

- Predict one continuous compiled II per `(task DFG, mapper shape)`.
- Use `models/final/ensemble.json` and its three checkpoints for DSE ordering.
- The ordinary three-checkpoint ensemble is the only deployment model.
- Keep exact rectangle packing, task dependencies, fusion/fission decisions,
  and program scoring in Amoeba.
- Treat mapper failures and timeouts as censored outcomes, never as numeric II.
- The current taskflow decision is to disable resource-aware fusion/fission.

## Frozen results

- Point ensemble validation/test II MAE: `0.335206 / 0.380498`.
- Attention no-fusion DSE: interval `29,360,129`, dependency-DAG makespan
  `62,588,435`; corrected earlier Amoeba makespan `66,782,741`.
- ResNet no-fusion DSE: interval `7,077,891`, dependency-DAG makespan
  `7,119,366`; earlier Amoeba interval `9,437,190`, makespan `18,958,393`.

Machine-readable evidence is in `evaluations/final-model.json` and
`evaluations/dse-vs-legacy-amoeba-2026-09-06.json`.

## Repository boundary

This cleanup affects only `cgra-ii-predictor`. The Amoeba and Neura working
trees were not changed. Detailed local training reports and mapper logs remain
untracked and must not be added accidentally.
