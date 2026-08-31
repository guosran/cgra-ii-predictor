# CGRA II Predictor

This repository studies prediction of the initiation interval produced by an
existing CGRA mapper. The model is intentionally separate from compiler
correctness and mapping: it predicts a mapper result for offline evaluation and
DSE ranking, and is never a pruning lower bound or a replacement mapping.

The core package is compiler-agnostic. It consumes portable JSON samples with:

- a stable sample identifier;
- a leakage group identifying the source program or generator template;
- a proven lower bound;
- the mapper's actual compiled II;
- numeric DFG, architecture, and mapper-configuration features.

Neura-specific lowering and feature collection live only in adapters. A Neura
checkout may be supplied with --neura-root or NEURA_ROOT. It may also be added
as a local submodule under third_party/neura, but the Python package does not
import or build Neura.

## Training data for a general predictor

A generality claim must cover a distribution of triples:

~~~text
(DFG, architecture, mapper configuration) -> compiled II
~~~

It must not be based mainly on one specialized DFG family. The intended corpus
has four complementary strata:

1. Real programs from independent suites: dense linear algebra, stencil and
   signal processing, sparse/graph kernels, reductions, pointer chasing,
   control-heavy kernels, and ML operators.
2. Motif-based generated DFGs: recurrence, fanout, merge, memory chain,
   reduction, and mixed-control motifs composed by multiple independent
   generators. Uniform random DAGs are a supplement, not the test set.
3. Architecture variation: array shape and masks, heterogeneous FU placement,
   memory-tile placement, register capacity, link width/bandwidth, and latency.
4. Mapper configuration variation when configuration is part of the prediction
   target. Otherwise the mapper configuration must remain fixed and recorded.

All shapes, masks, compiler variants, and generated mutations of one source
program belong to the same leakage group. Evaluation should include:

- nested source-family holdout for model selection;
- whole-suite holdout for cross-domain transfer;
- architecture-family holdout for YAML/topology transfer;
- a final frozen blind corpus that was not used to select features.

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
  "compiled_ii": 8,
  "features": {
    "semantic_edges": 42,
    "memory_path": 4,
    "compatible_slot_pressure": 2.5
  },
  "metadata": {
    "suite": "example",
    "architecture_id": "mesh-4x4",
    "mapper_id": "heuristic-v1"
  }
}
~~~

Flat Neura experiment reports are accepted as an adapter compatibility format;
new adapters should emit the nested portable schema.

## Usage

From the repository root:

~~~sh
PYTHONPATH=src python3 -m cgra_ii_predictor.cli schema/example-dataset.json
~~~

To collect or reuse Neura experiments:

~~~sh
python3 adapters/neura_experiment.py --neura-root /path/to/neura --help
~~~

The current model is residual Ridge regression above a proven lower bound. Its
regularization and small-residual dead zone are selected only inside nested
group holdout. More complex models should be compared only after the number of
independent source families, not merely the number of shapes, is large enough.

