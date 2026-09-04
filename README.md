# CGRA II Predictor

The current implementation status, validation evidence, censored samples, and
known review issues are recorded in [MODEL_STATUS.md](MODEL_STATUS.md). The
development/frozen distinction and holdout rules are specified in
[EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md). The exact prediction-time
inputs, formula, outputs, and commands are in
[II_PREDICTION.md](II_PREDICTION.md). The weighted objective, identity levels,
and nested training algorithm are in [TRAINING.md](TRAINING.md).
The concrete local benchmark inventory and the predeclared generated-training/
frozen-test allocation are in
[BENCHMARK_COMPATIBILITY.md](BENCHMARK_COMPATIBILITY.md).

This repository studies prediction of the initiation interval produced by an
existing CGRA mapper. The model is intentionally separate from compiler
correctness and mapping. Its current operational scope is an offline continuous
point estimate of the mapper's compiled II. It is never a pruning lower bound,
a feasibility proof, or a replacement mapping.

The core package is compiler-agnostic. It consumes portable JSON samples with:

- a stable sample identifier;
- a leakage group identifying one source-algorithm lineage or one generated
  base DFG/common-ancestor mutation family;
- both integer RecMII/ResMII components and their exact fixed maximum;
- the mapper's actual compiled II;
- numeric DFG, architecture, and mapper-configuration features.

Neura-specific lowering and feature collection live only in adapters. Neura
and the frozen real-suite source are pinned as submodules. Initialize both once
after cloning:

~~~sh
git submodule update --init third_party/neura third_party/machsuite
~~~

The adapter discovers the initialized submodule automatically. `--neura-root`
or `NEURA_ROOT` can override it for an intentional compatibility experiment.
The Python package does not import or build Neura.

## Primary experimental split

A generality claim must cover a distribution of triples:

~~~text
(DFG, architecture, mapper configuration) -> compiled II
~~~

The primary paper protocol assigns different sources to different roles:

1. Training and every fitted-model/hyperparameter decision use generated DFGs
   only. `motif-v3` contains chain, fanout, reduction, diamond, random-DAG,
   recurrence-chain, predicated-diamond, memory-stream, and pointer-chase
   families. Generated base graphs, not individual rows, are the split unit.
2. The fixed MachSuite revision is the final real-program test. No MachSuite
   mapper label has been accessed while designing or fitting Model 1. Its
   label-free preflight status and covariates have been inspected, so the
   current protocol is label-blind, not covariate-blind; that limitation is
   stated rather than presented as a stronger statistical-blindness claim.
3. The physical architecture is the exact pinned Neura 4x4 YAML. The generated
   stratum varies only rectangular active domains through Neura's existing
   `x-tiles`/`y-tiles` options. Tile masks and other architecture changes are
   outside the current frozen training protocol.
4. Mapper configuration varies only when configuration is part of the target;
   otherwise it remains fixed and recorded.

All shapes, masks, compiler variants, and generated mutations of one source
program belong to the same leakage group. Evaluation should include:

- nested generated-base lineage holdout for model selection;
- generator-family holdout inside generated development data;
- within-DFG target-shape ranking on the pinned YAML; the current corpus does
  not support a cross-YAML architecture-family claim;
- the separately sealed MachSuite test, with unsupported variants retained as
  censored records in the declared-suite denominator.

Timeouts are censored observations, not numeric compiled-II labels. They must
be stored separately and excluded from ordinary regression metrics unless a
censor-aware model is used.

## Portable dataset

See schema/example-dataset.json. A minimal sample is:

~~~json
{
  "sample_id": "suite/kernel/4x4",
  "group": "suite/kernel",
  "lower_bound": 5,
  "rec_mii": 5,
  "res_mii": 3,
  "compiled_ii": 8,
  "features": {
    "semantic_edges": 42,
    "memory_path": 4,
    "compatible_slot_pressure": 2.5
  },
  "metadata": {
    "lower_bound_source": "rec_res_max_v1",
    "suite": "example",
    "lineage": "example/kernel",
    "leakage_lineage_id": "example/kernel",
    "base_dfg_id": "sha256-of-canonical-dfg",
    "ranking_query_id": "sha256-of-canonical-dfg",
    "architecture_id": "mesh-4x4",
    "architecture_variant": "4x4",
    "candidate_id": "mesh-4x4:heuristic-v1",
    "mapper_id": "heuristic-v1",
    "training_stratum": "real"
  }
}
~~~

`lower_bound`, `rec_mii`, and `res_mii` are required contract fields outside
the feature object. `rec_mii` and `res_mii` are non-negative integers;
`lower_bound=max(rec_mii,res_mii)` must be positive, so one component may be
zero but both may not. Dataset/model loaders reject these fields if selected
as Ridge features.

Flat Neura experiment reports are accepted as an adapter compatibility format;
new adapters should emit the nested portable schema.

## Usage

From the repository root:

~~~sh
PYTHONPATH=src python3 -m cgra_ii_predictor.cli schema/example-dataset.json
~~~

Pure inference is a separate command and does not accept `compiled_ii` labels.
This self-contained example first trains a tiny illustrative model:

~~~sh
PYTHONPATH=src python3 -m cgra_ii_predictor.cli \
  schema/example-dataset.json --output /tmp/example-model.json

PYTHONPATH=src python3 -m cgra_ii_predictor.predict \
  --model-report /tmp/example-model.json \
  --input schema/example-prediction-input.json \
  --output /tmp/example-predictions.json
~~~

Historical exploratory weights, if present in a working copy, use the old
feature contract and are not a final model. The command validates a model's
canonical hash and numeric shape before predicting.

Legacy family aliases and additional domain holdouts are explicit:

~~~sh
PYTHONPATH=src python3 -m cgra_ii_predictor.cli dataset.json \
  --group-alias old-a=common-lineage \
  --group-alias old-b=common-lineage \
  --holdout-metadata-key architecture_id
~~~

To collect or reuse Neura experiments:

~~~sh
git submodule update --init third_party/neura third_party/machsuite
python3 adapters/neura_experiment.py --help
~~~

To reproduce the historical motif-v3 generated-training stratum (the default
count is zero, so ordinary invocations do not run it):

~~~sh
python3 adapters/neura_experiment.py \
  --motif-generator-version motif-v3 \
  --motif-samples-per-family 250 \
  --metadata-holdout-key generator_family \
  --motif-jobs 12 --motif-checkpoint-every 32 \
  --tree-depth 0 \
  --timeout 60 \
  --output-dir /path/to/random-training
~~~

This first materializes every source/architecture candidate and atomically
writes `corpus-manifest.json` before invoking any compiler subprocess. Candidate
collection is bounded by `--motif-jobs` (default `1`); Rec/Res analysis and
mapping remain serial within each candidate. A base DFG's shape and
architecture variants share one lineage and source/canonical hash; mapper
timeouts remain censored manifest entries. Each base declares 4x4 plus one
balanced secondary rectangle from the 2x2-through-4x4 scan. Only bases that
succeed in both cells enter validation or fitting; partial successes remain
auditable but excluded. Checkpoints are atomic, manifest records remain in
candidate-declaration order, and only the main coordinator updates the
manifest. `--samples` is retained as a legacy narrow random-DAG generator and
is not automatically mixed into this motif corpus. `--tree-depth 0` keeps only
a constant-residual tree diagnostic; the frozen model class is predeclared
Ridge, so an exhaustive threshold tree is not needed at formal scale. See
[CORPUS_PROTOCOL.md](CORPUS_PROTOCOL.md) for benchmark
roles, shape/op-count rules, and the generated-only training protocol.

The most recently executed protocol is `motif-v4`, machine-readably frozen in
[`protocols/motif-v4.json`](protocols/motif-v4.json). It must be predeclared in
a separate command before any mapper label is collected. Predeclaration does
not execute the compiler, but it requires an already built `mlir-neura-opt`
(resolved under `--neura-root`, or supplied explicitly with `--opt`) so the
binary identity can be frozen in the manifest:

~~~sh
python3 adapters/neura_experiment.py \
  --neura-root /path/to/pinned/neura \
  --motif-generator-version motif-v4 \
  --motif-samples-per-family 250 \
  --motif-predeclare-only \
  --seed 20260903 \
  --timeout 60 \
  --output-dir /path/to/motif-v4-formal
~~~

This writes all inputs and a 1,500-base/3,900-candidate label-free manifest,
then exits before executing the compiler or invoking the mapper. Six path
contexts are each crossed with layered/sparse, reconvergent, long-range
cutwidth, live-range, and mixed-path pressure profiles. The balanced blocks
keep 4x4 on every base and keep 2x3/3x2, 2x4/4x2, and 3x4/4x3 together on the
same source hash, ranking query, and leakage lineage.

The materialized local declaration and its exact protocol/generator/adapter,
toolchain, and manifest hashes are recorded in
[`protocols/motif-v4-predeclaration.json`](protocols/motif-v4-predeclaration.json).
The attested `corpus-manifest.predeclared.json` remains immutable while resume
updates the active `corpus-manifest.json`. The attestation explicitly is not
an external trusted timestamp.

The predeclared 2026-09-03 run has now completed. Its immutable result summary
and artifact hashes are recorded in
[`protocols/motif-v4-result.json`](protocols/motif-v4-result.json). The run
failed the predeclared acceptance gates and is therefore closed as a diagnostic
result: its fitted Ridge model must not be frozen, censored candidates must not
be retried under v4, and MachSuite mapper labels must remain unrevealed. Any
follow-up generator, timeout, feature, model, or acceptance-policy change needs
a new protocol version.

That next version is now predeclared as
[`protocols/motif-v5.json`](protocols/motif-v5.json). V5 keeps the v4 graph
grammar and balanced shape blocks but draws a disjoint seed. It changes the
statistical contract in two important ways: all successful mapper rows,
including successes from an otherwise partial shape block, train the point
model; only complete declared shape blocks enter ranking metrics. The point
predictor is a fixed hybrid: shapes with at most nine tiles and
`ResMII >= RecMII` use the learned nonnegative residual, while all other inputs
return the analytical `max(RecMII, ResMII)` result. RecMII and ResMII route this
predeclared gate but are not learned features.

Mapper timeout/nonzero outcomes are fitted and reported separately as a binary
risk estimate. They are never converted to a numeric II. GPRM-style joint
DFG/CGRA modeling, motivated by `references/TCAD-2026-1434_Proof_hi.pdf`, is a
backup if the simple hybrid fails; online RL remains deferred until after a
supervised or bandit-warm-start baseline.

The v5 label-free corpus was materialized locally at
`corpora/motif-v5-formal-seed-20260904`: 1,500 bases and 3,900 declared
candidates, with zero label or Rec/Res artifact fields. Its local hashes are in
[`protocols/motif-v5-predeclaration.json`](protocols/motif-v5-predeclaration.json).
No v5 mapper collection was started; v6 superseded it while it was still
label-free.

The active experiment is
[`protocols/motif-v6.json`](protocols/motif-v6.json). V6 retains the fixed
hybrid predictor and the same unlabelled 1,500 base DFGs, but crosses each DFG
with all 16 oriented rectangles from 1x1 through 4x4. Exact Top-1 shape
accuracy is primary: oracle and prediction minimize II, then break ties by tile
count, rows, columns, and candidate ID. Hyperparameters are selected by Top-1
accuracy first; compiled-II regret, shape-balanced error, and pairwise
concordance are secondary. Reproduce the 24,000-candidate declaration with:

~~~sh
python3 adapters/neura_experiment.py \
  --neura-root /path/to/pinned/neura \
  --motif-generator-version motif-v6 \
  --motif-samples-per-family 250 \
  --motif-predeclare-only \
  --seed 20260904 \
  --timeout 60 \
  --output-dir /path/to/motif-v6-formal
~~~

The materialized declaration and its implementation/toolchain hashes are in
[`protocols/motif-v6-predeclaration.json`](protocols/motif-v6-predeclaration.json).

After reviewing and preserving that manifest, collection is a separate resume:

~~~sh
python3 adapters/neura_experiment.py \
  --neura-root /path/to/pinned/neura \
  --output-dir /path/to/motif-v6-formal \
  --motif-generator-version motif-v6 \
  --motif-resume \
  --metadata-holdout-key generator_family \
  --motif-jobs 12 --motif-checkpoint-every 32 \
  --tree-depth 0 --timeout 60
~~~

V6 collection completed with 20,188 successful candidates and 3,812 censored
outcomes. Its complete-16 coverage gate failed: only 352 bases had all 16
successful mappings, and memory, pointer, and mixed had no complete blocks.
V6 is therefore Model-2 development data, not a successful formal result.

Model 2 is a joint DFG/CGRA graph model with separate mapper-success and
successful-II heads plus a listwise 16-shape loss. Censored candidates train
only the success head and never receive a fabricated II. On the fixed v6
development test split, the selected configuration improved strict Top-1 from
12.50% to 23.21%, optimal-II rate from 31.25% to 49.11%, selected-success rate
from 68.30% to 92.41%, and timeout-penalized regret from 3.77 to 1.54. These
are disclosed development results, not blind evidence.

The held-out contract is [`protocols/motif-v7.json`](protocols/motif-v7.json).
It freezes that configuration, refits exactly 28 epochs on all v6 queries, and
then evaluates without an optimizer on a canonically disjoint v7 draw:

~~~sh
python3 adapters/neura_graph_frozen.py freeze \
  --training-manifest corpora/motif-v6-formal-seed-20260904/corpus-manifest.json \
  --development-report corpora/model2-v6-development-strict1/report.json \
  --output-dir models/model2-v7-frozen

python3 adapters/neura_experiment.py \
  --neura-root /path/to/pinned/neura \
  --motif-generator-version motif-v7 \
  --motif-samples-per-family 250 \
  --motif-predeclare-only --seed 20260906 --timeout 60 \
  --output-dir /path/to/motif-v7-formal

python3 adapters/neura_experiment.py \
  --neura-root /path/to/pinned/neura \
  --motif-generator-version motif-v7 --motif-resume \
  --motif-collect-only --motif-jobs 12 \
  --motif-checkpoint-every 32 --timeout 60 \
  --output-dir /path/to/motif-v7-formal

python3 adapters/neura_graph_frozen.py evaluate \
  --model models/model2-v7-frozen/model.pt \
  --manifest /path/to/motif-v7-formal/corpus-manifest.json \
  --output-dir /path/to/motif-v7-formal/model2-evaluation
~~~

Strict deterministic Top-1 is primary. Optimal-II and selected-success rates
must not regress overall or in any family, timeout-penalized regret must
improve, and every family must retain at least 200 ranking-eligible bases with
two or more successful shapes. The local declaration is tamper-evident but is
not an external trusted timestamp.

That statement describes the historical frozen v7 ranking protocol, not the
current deployment contract. In the program-level DSE frontend, the predictor
is called on one `(task DFG, assigned CGRA candidate)` pair at a time. The
frontend analytical model combines independently predicted task IIs to score
complete spatiotemporal candidates. `discrete_pointwise` implements this
contract: it has no cross-candidate attention or listwise loss and emits a
continuous expected II, integer mode, class distribution, uncertainty, and
mapper-success probability. Its checkpoint-selection metric is successful-
candidate continuous II MAE; shape rankings are downstream diagnostics only.
`residual_pointwise` keeps the same independent interface but models the
nonnegative integer residual above `max(RecMII, ResMII)` instead of absolute II
classes, so residual behavior is shared across candidates with different lower
bounds and the continuous estimate respects the analytical floor by design.

V4 fails closed unless generator-family LOGO MAE strictly improves on the
Rec/Res floor, held-out predictions recover positive residuals in every
family, positive-subset and shape-balanced MAE improve, tie-aware shape
ranking does not regress, and every family-by-shape, family-by-profile, and
family-by-operation-band marginal cell passes coverage. Rec/Res analysis
facts, the mapper-II ceiling, mapper-attempt
status, successful labels, and feasible censorship have separate denominators;
an `LB > 20` candidate is recorded outside the search interval and never
fabricated as `II=21`. Formal mapper collection has intentionally not been
started by this code/protocol change.

To resume that collection after an interruption, use the same output directory
and omit `--clean`:

~~~sh
python3 adapters/neura_experiment.py \
  --output-dir /path/to/random-training \
  --motif-resume \
  --metadata-holdout-key generator_family \
  --motif-jobs 12 --motif-checkpoint-every 32 \
  --tree-depth 0 \
  --timeout 60
~~~

Resume validates the manifest, generator/timeout/compiler identity, candidate
inputs, and cached artifact hashes before invoking a subprocess. It reconstructs
cached successes without the compiler, skips censored candidates without
retrying them, and fails before invocation if a cached artifact is corrupt. A
fully terminal resume needs no compiler probe. `--clean` and `--motif-resume`
are mutually exclusive. Omitted generator choices are recovered from the
manifest. Motif-v4 runs the required generator-family holdout automatically;
historical v3 commands must still repeat
`--metadata-holdout-key generator_family`. SIGINT exits 130 after
draining at most the configured in-flight candidates and does not train a model.
The manifest is a single-coordinator/single-writer contract with no
cross-process lock; do not run two fresh or resume processes against one
output directory concurrently.

To predict a lowered DFG without labels, refitting, or invoking the mapper:

~~~sh
python3 adapters/neura_experiment.py \
  --model-report /path/to/frozen-random-dfg-model.json \
  --predict-fixture kernel=/path/to/lowered-kernel.mlir
~~~

With no `--predict-shape`, inference records 1x1, a canonical 1x2 two-tile
strip, and all nine 2x2-through-4x4 rectangles. The frozen v3 corpus trained
only on the nine larger rectangles, so 1x1/1x2 are explicitly marked
`stress_only_untrained_shape` and excluded from the automatic Pareto frontier
and mapper-verification order. They remain useful audit records. The report
also flags a candidate when `max(RecMII, ResMII) > 20`, because the pinned
mapper then has an empty II search interval. A single final shape is
objective-dependent; the predictor never treats a ranking as proof of
feasibility.

The current model is residual Ridge regression above the fixed floor
`max(RecMII, ResMII)`, recorded as `rec_res_max_v1`. Neither the floor nor
`rec_mii`/`res_mii` belongs to the primary Ridge feature vector. Regularization
and the small-residual dead zone are selected only inside nested group holdout.
Each outer or inner validation call independently uses leave-one-lineage-out when
its current split has at most 20 groups; otherwise it uses up to 10
deterministic stratified group folds. Generated lineages and distinct
observations are balanced while preserving the Ridge penalty scale. Rows
sharing one observation identity under the active model projection share one
observation's weight.

## Frozen MachSuite test

`benchmarks/machsuite-v1.json` pins 19 variants at MachSuite commit
`6236e593012cb86b0d2f08d9fb9ba0411ff989b4`. The executable protocol is:

~~~sh
# Train/select only on generated motif and random-DFG data.
python3 adapters/neura_experiment.py \
  --motif-generator-version motif-v3 \
  --motif-samples-per-family 250 \
  --metadata-holdout-key generator_family \
  --motif-jobs 12 --motif-checkpoint-every 32 \
  --tree-depth 0 \
  --timeout 60 \
  --output-dir /path/to/random-training

# Remove training rows and freeze the selected generated-only model.
python3 adapters/machsuite_frozen.py freeze-model \
  --training-report /path/to/random-training/report.json \
  --output /path/to/frozen-random-model.json

# Label-free compatibility/feature preflight; this never runs the mapper.
python3 adapters/machsuite_frozen.py preflight \
  --output-dir /path/to/machsuite-preflight

# Predict and hash-seal the result before labels are accessible.
python3 adapters/machsuite_frozen.py predict \
  --preflight /path/to/machsuite-preflight/preflight.json \
  --model /path/to/frozen-random-model.json \
  --output /path/to/machsuite-predictions.json \
  --seal /path/to/machsuite-prediction-seal.json

# Publish or externally timestamp the seal here; SHA-256 alone does not prove
# chronology if every local artifact can be recreated later.

# Only now run the mapper and join labels to sealed predictions.
python3 adapters/machsuite_frozen.py reveal \
  --preflight /path/to/machsuite-preflight/preflight.json \
  --model /path/to/frozen-random-model.json \
  --predictions /path/to/machsuite-predictions.json \
  --seal /path/to/machsuite-prediction-seal.json \
  --output-dir /path/to/machsuite-revealed
~~~

The currently implemented MachSuite freezer deliberately remains tied to the
historical motif-v3 evidence. That formal contract predeclares exactly 250
bases in each of nine families:
2,250 bases and 4,500 paired shape candidates. At least 200 complete bases per
family must remain, giving at least 1,800 fitted lineages and 3,600 rows.
The 13-feature design plus intercept must also be full rank before freezing.
`--allow-small-smoke` can exercise serialization on a tiny corpus, but marks
the artifact smoke-only and `predict` refuses it for frozen MachSuite.
`freeze-model` also re-generates each source, verifies all source,
architecture, Rec/Res and mapped artifacts, independently reproduces the
nested selection and final Ridge fit, and requires generated nested-lineage
macro MAE to be strictly below the raw Rec/Res-floor MAE. The v2 artifact
stores the exact sorted training canonical-hash set; `predict` and `reveal`
reject any hash shared with a ready MachSuite DFG.

With the pinned Neura/LLVM toolchain, the label-free preflight currently makes
11/19 variants ready and records 8/19 as lowering-censored. This is a frozen
compatible-subset evaluation with 19 declared inputs, not a claim of full
MachSuite frontend coverage.

The CLI reports an empirical unseen-group interval calibrated from the maximum
error in each nested held-out group; it is uncertainty about the point estimate,
not a feasibility bound or formal coverage guarantee. It reports DSE ranking
only inside an explicit `ranking_query_id` (one exact base DFG) whose candidate
IIs vary. A leakage lineage is never silently treated as a ranking query. The
legacy corpus lacks these identities and cannot support a ranking claim.

LISA motivates the generated-graph-training/real-program-test split and a
later graph- and mapping-aware Model 2; this is inspiration, not a reproduction.
LISA predicts node/edge guidance labels with a GNN and does not directly
predict final compiled II. More complex graph models should be considered only
after the linear baseline has enough independent source and architecture
families for the holdouts described above.
