# Compiled-II Point Prediction

This document describes only point inference. Training, model selection, DSE
ranking, and frozen-evaluation claims are separate concerns.

## Target and prediction-time inputs

The target is the `compiled_ii` that the recorded mapper configuration would
produce for one `(DFG, architecture, mapper configuration)` candidate. It is
not the globally optimal II and it is not a mapping-feasibility proof.

Prediction needs exactly two kinds of numeric facts:

1. The authoritative `lower_bound = max(RecMII, ResMII)`, recorded with
   `lower_bound_source=rec_res_max_v1`. Both integer components are required
   and the loader independently checks the exact maximum.
2. Every feature named by the loaded model's `feature_names`, computed before
   heuristic mapping. The primary model deliberately excludes `baseline_lb`,
   `lower_bound`, `proven_lower_bound`, `rec_mii`, and `res_mii`. Model and
   dataset loaders reject artifacts that try to select any of them.

Model 1 freezes 13 structure and structure/resource-interaction features:

~~~text
semantic_depth, semantic_width, sources, semantic_branch_density,
semantic_cut_fraction, multi_input_density, memory_op_density,
pointer_path_fraction, memory_path_fraction, compute_fu_peak_pressure,
memory_fu_pressure, routing_cut_pressure, register_pressure
~~~

The pressure and density ratios use active tiles, memory-capable boundary tiles,
directed mesh links, bisection links, and physical registers from the pinned
Neura YAML. They are descriptive features, not extra lower-bound terms.
Raw graph-size counts that caused each label-free MachSuite task to exceed the
training range on at least one feature are deliberately excluded; depth, width,
sources, normalized topology densities, and resource pressures retain the
relevant structure.

`compiled_ii` is a training/evaluation label. It is forbidden at the sample,
feature, and metadata levels of the standalone prediction-input schema and is
never used during inference. A full historical training report may contain
labels and is parsed as a JSON container to extract its model; prefer a slim
frozen model artifact when the distinction matters operationally.

## Exact calculation

For feature `j`, the model first reuses the training-set normalization saved in
the artifact:

~~~text
z_j = (feature_j - mean_j) / scale_j
~~~

The residual Ridge model then computes:

~~~text
raw_residual = weights[0] + sum(weights[j + 1] * z_j)
nonnegative_residual = max(0, raw_residual)

predicted_residual = 0                         if nonnegative_residual < dead_zone
                     nonnegative_residual       otherwise

predicted_compiled_ii = lower_bound + predicted_residual
~~~

The non-negative floor is a hard semantic constraint: final mapper II cannot be
below `max(RecMII, ResMII)`. The dead zone is a validation-selected rule for
small predicted residuals; it is not an additional analytical theorem.

The authoritative result is a continuous floating-point point estimate. The
inference command deliberately does not round it. Any integer conversion must
be a separately named downstream policy, because nearest, ceiling, and floor
have different meanings.

If the model contains a held-out-group error radius, the command also reports:

~~~text
interval_lower = max(lower_bound, predicted_compiled_ii - radius)
interval_upper = predicted_compiled_ii + radius
~~~

This is an empirical diagnostic from earlier held-out groups, not formal
coverage and not a mapping-feasibility interval.

## Standalone inference from numeric features

First obtain a report containing `trained_full_model`, or use a direct model
artifact. Historical local exploratory artifacts are deliberately ignored and
are not part of a clean checkout; generate a fresh report with the current
feature/lower-bound contract or freeze the large generated-DFG run.

The prediction-only input is:

~~~json
{
  "schema_version": "compiled-ii-prediction-input-v1",
  "samples": [
    {
      "sample_id": "kernel/4x4",
      "lower_bound": 5,
      "rec_mii": 5,
      "res_mii": 3,
      "features": {
        "semantic_edges": 20,
        "semantic_depth": 8
      },
      "metadata": {
        "lower_bound_source": "rec_res_max_v1",
        "source_sha256": "...",
        "architecture_id": "...",
        "mapper_id": "neura-heuristic",
        "mapper_revision": "...",
        "mapper_config": "mapping-strategy=heuristic"
      }
    }
  ]
}
~~~

Run:

~~~sh
PYTHONPATH=src python3 -m cgra_ii_predictor.predict \
  --model-report /path/to/current-model.json \
  --input prediction-input.json \
  --output predictions.json
~~~

For a report container, the loader requires and verifies its canonical model
SHA-256. It also checks the model type, array dimensions,
finite parameters, positive scales and Ridge coefficient, and non-negative dead
zone/interval radius. Prediction rejects missing or non-finite features,
duplicate sample IDs, labels, negative/non-integral Rec/Res components (zero is
allowed for either component when the resulting lower bound is positive), and
any disagreement between `lower_bound`, `proven_lower_bound`, `baseline_lb`,
and the exact `max(RecMII, ResMII)` contract.

For a self-contained format smoke test, train the tiny example first:

~~~sh
PYTHONPATH=src python3 -m cgra_ii_predictor.cli \
  schema/example-dataset.json \
  --ridge-candidates 1 --dead-zone-candidates 0 \
  --output /tmp/example-ii-model-report.json

PYTHONPATH=src python3 -m cgra_ii_predictor.predict \
  --model-report /tmp/example-ii-model-report.json \
  --input schema/example-prediction-input.json \
  --output /tmp/example-ii-predictions.json
~~~

## End-to-end Neura prediction without mapping

For a lowered Neura DFG, the adapter can load the existing model, run only the
analysis-only RecMII/ResMII pass plus static feature extraction, and predict:

~~~sh
python3 adapters/neura_experiment.py \
  --model-report /path/to/current-model.json \
  --predict-fixture kernel=/path/to/lowered-kernel.mlir \
  --output-dir /tmp/kernel-ii-prediction
~~~

In this mode:

- no candidate `compiled_ii` is accepted or used (a full historical model
  report may still be parsed as the model container);
- the model is not selected or refitted;
- the heuristic mapper is not invoked;
- `mlir-neura-opt --analyze-rec-res-mii` supplies RecMII/ResMII by directly
  calling the same Neura C++ functions used by the mapper; the pass emits no
  `compiled_ii` and performs no placement or routing, while static DFG
  analysis supplies the structural features;
- `report.json` records source, architecture, mapper and model identities,
  the exact model features, residual post-processing, interval semantics, and
  any revision or dirty-producer warning;
- failure to produce every requested prediction is reported as incomplete and
  returns a nonzero process status.

## Per-task target-shape selection

Without `--predict-shape`, the adapter evaluates all nine prefix rectangles
`R x C` with `R,C in {2,3,4}`. Every candidate uses the same byte-exact Neura
4x4 YAML; both the Rec/Res analysis and later mapper verification receive the
same existing `x-tiles=C y-tiles=R` override. Non-rectangular `valid-tiles`
masks are not used because that path is not reliable in the pinned Neura
revision.

For each task, the report contains:

1. one continuous compiled-II estimate per rectangle;
2. the nondominated `(active tile count, predicted II)` Pareto candidates;
3. the throughput-first candidate, breaking an II tie toward fewer tiles;
4. a prediction-ranked order for verification with the unchanged Neura mapper.

There is intentionally no universal single “best” shape. Throughput-first work
chooses the lowest predicted II; an area-constrained caller chooses the lowest
II point within its tile budget; an area-first caller starts at the smallest
Pareto point. A prediction ranks mapper attempts but never proves feasibility.
The transposed shapes may receive the same estimate when their Rec/Res facts
and pressure features match, which is consistent with the x/y symmetry of the
pinned mesh and boundary-memory layout; the mapper order remains the final
tie-break experiment.

The final fitted artifact stores each feature's observed min, 1st percentile,
99th percentile, and max. Values outside the central 98% are reported as tail
diagnostics; values outside the observed min/max produce a hard OOD warning.
This is a feature-wise support check, not proof of multivariate in-distribution
generalization.
Hard-OOD shape candidates remain in the prediction audit record but are
excluded from the automatic Pareto frontier and mapper-verification order.

Historical Model 1 artifacts use an earlier feature contract and are not valid
for the primary frozen MachSuite workflow. A final artifact must be trained on
generated DFGs with `rec_res_max_v1`, use the predeclared structure-only
feature list, and be sealed before MachSuite mapper labels are revealed. These
limitations affect confidence in a prediction; they do not change the formula.

The frozen container is `compiled-ii-model-artifact-v2`. It carries the exact
sorted set of canonical DFG hashes used to fit the model, its count, and a set
digest. The generic loader retains v1 compatibility for non-frozen use, but the
MachSuite `predict` path accepts only v2 and rejects any ready test DFG whose
canonical hash occurs in that training set. `reveal` repeats the overlap check
before it can invoke the mapper.
