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
   only. `motif-v2` contains chain, fanout, reduction, diamond, mixed,
   random-DAG, recurrence-chain, predicated-diamond, and pointer-chase
   families. Generated base graphs, not individual rows, are the split unit.
2. The fixed MachSuite revision is the final real-program test. No MachSuite
   mapper label has been accessed while designing or fitting Model 1. Its
   label-free preflight status and covariates have been inspected, so the
   current protocol is label-blind, not covariate-blind; that limitation is
   stated rather than presented as a stronger statistical-blindness claim.
3. The implemented generated stratum varies array shape and homogeneous versus
   split-domain FU placement. Tile masks, register capacity, memory-tile
   placement, link width/bandwidth, and latency are separate architecture
   extensions; they are not claimed as dimensions of the current frozen
   training protocol.
4. Mapper configuration varies only when configuration is part of the target;
   otherwise it remains fixed and recorded.

All shapes, masks, compiler variants, and generated mutations of one source
program belong to the same leakage group. Evaluation should include:

- nested generated-base lineage holdout for model selection;
- generator-family holdout inside generated development data;
- architecture-family holdout for YAML/topology transfer;
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

To collect the fixed generated-training stratum (the default is zero, so
ordinary invocations do not run it):

~~~sh
python3 adapters/neura_experiment.py \
  --motif-samples-per-family 250 \
  --motif-shape 3x3 --motif-shape 3x4 --motif-shape 4x4 \
  --metadata-holdout-key generator_family \
  --motif-jobs 4 --motif-checkpoint-every 32 \
  --timeout 60
~~~

This first materializes every source/architecture candidate and atomically
writes `corpus-manifest.json` before invoking any compiler subprocess. Candidate
collection is bounded by `--motif-jobs` (default `1`); Rec/Res analysis and
mapping remain serial within each candidate. A base DFG's shape and
architecture variants share one lineage and source/canonical hash; mapper
timeouts remain censored manifest entries. Only bases that succeed in all six
shape/layout cells enter validation or fitting; partial successes remain
auditable but excluded. Checkpoints are atomic, manifest records remain in
candidate-declaration order, and only the main coordinator updates the
manifest. `--samples` is retained as a legacy narrow random-DAG generator and
is not automatically mixed into this
motif corpus.  See [CORPUS_PROTOCOL.md](CORPUS_PROTOCOL.md) for benchmark
roles, shape/op-count rules, and the generated-only training protocol.

To resume that collection after an interruption, use the same output directory
and omit `--clean`:

~~~sh
python3 adapters/neura_experiment.py \
  --output-dir /path/to/random-training \
  --motif-resume \
  --motif-jobs 4 --motif-checkpoint-every 32 \
  --timeout 60
~~~

Resume validates the manifest, generator/timeout/compiler identity, candidate
inputs, and cached artifact hashes before invoking a subprocess. It reconstructs
cached successes without the compiler, skips censored candidates without
retrying them, and fails before invocation if a cached artifact is corrupt. A
fully terminal resume needs no compiler probe. `--clean` and `--motif-resume`
are mutually exclusive; SIGINT exits 130 after draining at most the configured
in-flight candidates and does not train a model.
The manifest is a single-coordinator/single-writer contract with no
cross-process lock; do not run two fresh or resume processes against one
output directory concurrently.

To predict a lowered DFG without labels, refitting, or invoking the mapper:

~~~sh
python3 adapters/neura_experiment.py \
  --model-report /path/to/frozen-random-dfg-model.json \
  --predict-fixture kernel=/path/to/lowered-kernel.mlir \
  --predict-shape 4x4
~~~

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
  --motif-samples-per-family 250 \
  --motif-shape 3x3 --motif-shape 3x4 --motif-shape 4x4 \
  --metadata-holdout-key generator_family \
  --motif-jobs 4 --motif-checkpoint-every 32 \
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

The formal contract predeclares exactly 250 bases in each of nine families:
2,250 bases and 13,500 shape/layout candidates. At least 200 complete bases per
family must remain, giving at least 1,800 fitted lineages and 10,800 rows.
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
