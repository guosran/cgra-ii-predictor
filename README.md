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

1. Training and every model/hyperparameter decision use generated DFGs only:
   chain, fanout, reduction, diamond, mixed, and random-DAG families. Generated
   base graphs, not individual rows, are the split unit.
2. The fixed MachSuite revision is the final real-program test. It is never
   used to select features, Ridge strength, dead zone, generator parameters,
   or frontend support policy.
3. Training varies array shape and masks, heterogeneous FU placement,
   memory-tile placement, register capacity, link width/bandwidth, and latency.
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

To explicitly collect the deterministic compute-motif stratum (the default
is zero, so ordinary invocations do not run it):

~~~sh
python3 adapters/neura_experiment.py \
  --motif-samples-per-family 200 \
  --motif chain,fanout,reduction,diamond,mixed,random_dag \
  --motif-shape 3x3 --motif-shape 3x4 --motif-shape 4x4
~~~

This first materializes every source/architecture candidate and atomically
writes `corpus-manifest.json` before invoking the mapper.  A base DFG's shape
and architecture variants share one lineage and source/canonical hash; mapper
timeouts remain censored manifest entries.  `--samples` is retained as a
legacy narrow random-DAG generator and is not automatically mixed into this
motif corpus.  See [CORPUS_PROTOCOL.md](CORPUS_PROTOCOL.md) for benchmark
roles, shape/op-count rules, and the generated-only training protocol.

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
  --motif-samples-per-family 200 \
  --motif chain,fanout,reduction,diamond,mixed,random_dag \
  --motif-shape 3x3 --motif-shape 3x4 --motif-shape 4x4 \
  --metadata-holdout-key generator_family \
  --timeout 120 \
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

`freeze-model` defaults to a predeclared minimum of 1,000 distinct generated
base DFGs across all six generator families. `--allow-small-smoke` can exercise
serialization on a tiny corpus, but marks the artifact smoke-only and
`predict` refuses to use it for a frozen MachSuite run.
It also re-generates each source from its seed, verifies source/architecture/
mapped-artifact hashes and labels, reproduces nested hyperparameter selection
and the final Ridge fit, and accepts only the direct default-shape/two-variant
training protocol shown above.

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

LISA motivates a later graph- and mapping-aware Model 2, not replacing this
model with a GNN immediately. LISA predicts node/edge labels to guide a mapper,
retraining per accelerator with a much larger generated graph corpus; it does
not directly predict final compiled II. More complex graph models should be
considered only after enough independent source and architecture families exist
for the holdouts described above.
