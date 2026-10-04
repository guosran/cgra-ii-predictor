# Deployment adapters

The retained adapters implement only the final Amoeba path:

- consume the packing-pruned static-shape manifest emitted by Amoeba;
- extract task DFGs and obtain analysis-only RecMII/ResMII;
- generate the direct heuristic-mapper II cost catalog;
- score frozen candidates and compare them with true mapper oracles.

The model never chooses a program. Amoeba's current candidate protocol guarantees
that all fixed-orientation task rectangles in a candidate can simultaneously
fit without overlap on the physical grid. Temporal reuse does not relax this
capacity constraint. Compiler output containing mapper labels is rejected by
the analysis-only feature parser.

TODO: a future analytical spatial-temporal scheduler may model cross-time tile
reuse. That scheduler will be a separate search scope; the current adapter
must continue to reject over-capacity candidates.

The external candidate, score, and cost schema identifiers are dictated by
the current Amoeba executable. Keep those wire identifiers unchanged unless
both repositories are migrated together.

Candidate enumeration also writes `amoeba.source_task_body_sha256` onto each
source task in its MLIR output. Run `extract_amoeba_task_dfgs.py` on that bound
output, not on the pre-enumeration input. The extractor carries the identity
into each standalone DFG; feature generation, catalog generation, and Amoeba
all reject a missing or mismatched identity before scoring.

## Per-CGRA 2x2 audit and bounded experiments

`audit_per_cgra_2x2_model.py` verifies the complete native collection and
replays `models/candidates/per-cgra-2x2/mapper.pt`. Its joined CSV/JSONL includes
the known baseline test labels, as required for baseline reproduction. Missing
native labels remain null. Features come exclusively from pre-mapper DFGs;
mapped artifacts are read only to validate native ground truth.

Use a new output directory for each study:

```sh
python3 adapters/audit_per_cgra_2x2_model.py \
  --collection /path/to/native-collection-2x2-v1 --output /path/to/new-study
python3 adapters/evaluate_per_cgra_2x2_baseline.py \
  --output /path/to/new-study --bootstrap 1000
python3 adapters/run_per_cgra_2x2_experiments.py plan \
  --development /path/to/new-study/development.pt --output /path/to/new-study
python3 adapters/run_per_cgra_2x2_experiments.py fit \
  --development /path/to/new-study/development.pt --output /path/to/new-study --jobs 8
python3 adapters/run_per_cgra_2x2_experiments.py summarize \
  --development /path/to/new-study/development.pt --output /path/to/new-study
python3 adapters/run_per_cgra_2x2_experiments.py test \
  --development /path/to/new-study/development.pt --output /path/to/new-study
python3 adapters/summarize_per_cgra_2x2_experiments.py --output /path/to/new-study
```

The runner fixes six configurations, four matched seeds, three outer grouped
development folds, and the original train/validation split. Inner validation
selects each checkpoint by source-stratum-balanced group regret, then hit,
then MAE. Outer folds assess stability; test predictions for alternative
models are produced only after selection is frozen. Random and program
sources each receive half the loss mass, with equal groups within each stratum
and equal successful rows within each group. Network hidden widths stay 64/32.

`epoch-checkpoints.npz` stores every epoch's parameters plus static normalization;
it supports inference replay, while exact optimizer replay starts from the
recorded seed. `selected.pt` is an experimental checkpoint, not a deployment
package. The runner seals cache, code, selected weights, and test hashes and
rejects overwritten plans or reopened test selections. No command promotes a
model or changes an ORBIT default. The 2026-10-04 executed runner is archived
with the study because additional CLI seal checks were added after those fits;
training calculations and the selected results were unchanged.

`per_cgra_2x2_evaluation.py` reports complete-eight-shape selection quality,
incomplete-query coverage, paired-shape ordering, source-group bootstrap
intervals, and native-confirmation budgets. II ties can have different tile
occupancy; these metrics alone are not a whole-program performance result.

`prepare_real_per_cgra_2x2_sources.py` lowers pinned real C sources and then
separately freezes an evaluation-only admission manifest after graph/source
overlap checks. It requires the local pinned auxiliary importer and compiler
toolchain recorded in its provenance. `evaluate_real_per_cgra_2x2_collection.py`
verifies completed supplemental labels and evaluates the retained candidate.
It requires the pre-mapping `launch-record.json` beside the admission, binding
admission, selection, experiment plan and query manifest by SHA-256. It checks
the complete admitted roster and recomputes the fixed selector using original
validation rows only. A replay with stronger validation preserves the first
result directory rather than overwriting it.
The collector's timeout applies separately to analysis and mapper search;
`--timeout-seconds 120` permits up to roughly 240 seconds plus termination
overhead for a query, not a single shared 120-second deadline.

Study report and compact result bundle: `../docs/PER_CGRA_2X2_IMPROVEMENT_20261004.txt`.

`prepare_orbit_per_cgra_2x2_variants.py` extracts the four pinned, evidenced
Ray fission/Harris tiling tasks, parser-reprints generic MLIR when necessary,
route-expands with the pinned Neura executable and removes existing/shared graph
identities. Its three admitted identities remain benchmark evaluation-only.
It never maps, trains or changes an ORBIT source/default. Missing body hashes
are explicitly marked local provenance fingerprints, not enumerator attestations.
The recorded command and pre-mapping launch hashes are in the result bundle's
`orbit-variant-source-preparation/`; its native timeout has the same per-phase
semantics described above.
