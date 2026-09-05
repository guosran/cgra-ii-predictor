# Amoeba static multi-CGRA II protocol

This protocol is intentionally separate from legacy submesh checkpoints.  A
legacy `rows=1, columns=2` sample means two mapper PEs.  Under the Amoeba
protocol, physical `1x2` means a `4x8` mapper-tile array with 32 PEs.  Only a
legacy label whose mapper dimensions are genuinely `4x4` can describe the new
physical `1x1` query.

## Finite shape domain

| Physical CGRAs | Mapper tiles |
| --- | --- |
| `1x1` | `4x4` |
| `1x2` | `4x8` |
| `2x1` | `8x4` |
| `1x3` | `4x12` |
| `3x1` | `12x4` |
| `1x4` | `4x16` |
| `2x2` | `8x8` |
| `4x1` | `16x4` |

Rows and columns are oriented.  Transposes are different inputs and different
labels.  The deployed model rejects every mapper shape outside this table;
dynamic and non-rectangular shapes remain TODO.

`motif-v8` manifests use `cgra-ii-motif-corpus-v8`.  Every candidate records
all four fields `physical_cgra_rows`, `physical_cgra_cols`,
`mapper_tile_rows`, and `mapper_tile_cols`.  Compatibility fields `rows` and
`columns` are exact aliases of mapper-tile dimensions, never physical units.
The cache identity is `(canonical DFG SHA-256, mapper_tile_rows,
mapper_tile_cols)`.

## Label contract

The generator freezes the complete manifest and
`corpus-manifest.predeclared.json` before invoking a tool.  Each candidate then
runs the pinned Neura analysis with explicit `x-tiles/y-tiles`, followed by the
pinned heuristic mapper with the same dimensions.  It records RecMII, ResMII,
`max(RecMII, ResMII)`, invocation outcome, and a compiled II only on success.
Timeouts and failures are censored categorical outcomes; they never receive a
numeric II.  Splits use canonical DFG identity, so eight shapes from one DFG
cannot cross train/validation/test boundaries.

Pinned producer identities:

- Neura revision: `47b7e3a68c321075293e6fcb45fb3b1cabb93b88`
- `mlir-neura-opt` SHA-256:
  `7c8b0753609c9045fd4dabd3041f5a5311ce3922f431526c9a61ea4d794e6d49`
- Architecture YAML SHA-256:
  `f244f15be30604eb32eb96e4837a4bf1ce5c34961c3a46299b90931505cc97e6`
- Mapping strategy: `heuristic`; mapper II ceiling: 20

## Model contract

`Model2Config.shape_protocol` is stored in every new checkpoint.  Omitted
metadata resolves only to the legacy protocol, preserving old checkpoint
semantics.  New graph/context normalization divides mapper rows and columns by
16, tile count by 64, directed mesh links by 224, boundary-memory tiles by 19,
bisection links by 16, and Manhattan distance by 18.  These constants cover
the finite domain without relying on values greater than one.

Soft-placement peak concentration has its own artifact field,
`routing_peak_normalization`.  Historical checkpoints omit it and retain the
original fixed-16 scale.  Final multi-CGRA checkpoints record
`protocol_max_tiles_v1`, which divides by the protocol maximum of 64 and
prevents the large-shape feature from saturating under a hidden 4x4
assumption.  Loading an old artifact therefore cannot silently reinterpret
this feature.

The network remains pointwise: it receives one task DFG, one mapper-tile graph,
RecMII, ResMII, and the analytical lower bound.  It emits continuous II mean,
II standard deviation, mapper-success probability, and optional integer
diagnostics.  Residual outputs are structurally clamped above the lower bound.
It neither scores complete programs nor invents startup cycles.

## Amoeba adapter

`adapters/amoeba_cost_catalog.py` consumes:

- an `amoeba-analytical-task-candidates-v2` JSONL manifest;
- repeated `--task-dfg TASK=PATH` pre-mapper DFGs;
- a `cgra-ii-amoeba-query-features-v1` JSON file containing per-query
  RecMII/ResMII/lower bound and frontend-provided startup cycles;
- named pointwise checkpoints and a validation-selected ensemble report.

When the frontend has a lowered Taskflow module rather than already-separated
DFGs, `adapters/extract_amoeba_task_dfgs.py` writes one standalone Neura
kernel module per accelerator task. It accepts `inputs`, `iter_args_init`, or
both, and normalizes Neura's non-round-trippable empty-index
`store_indexed` custom syntax to an equivalent generic operation with explicit
operand segments. Tasks without a Neura kernel are omitted.

For a large static Cartesian space,
`adapters/enumerate_amoeba_pruned_candidates.py` generates a v2 manifest with
two distinct filters. The per-task maximum area
`ceil(materialized_ops / tiles_per_physical_CGRA) + slack` is a search
heuristic and is recorded as such in `fixed_axes.candidate_pruning`. Exact
incremental oriented-rectangle packing on the physical grid is a hard
constraint. Do not describe the op-count cap as legality, and do not turn a
mapper timeout outside the cap into a numeric II label.

Generate the analytical input with
`adapters/generate_amoeba_query_features.py`.  It invokes only Neura's
analysis pass for each frozen `(task, mapper rows, mapper columns)` query.  It
derives startup from the unit-latency critical-path depth of the semantic
pre-mapper DFG (routing-only moves collapsed), which is frontend information;
it never invokes the mapper or reads a compiled-II label.

It parses each task once, batches unique cost queries, and writes
`amoeba-task-shape-cost-v2`.  Supported entries include prediction diagnostics
that the current scorer safely ignores.  Unsupported shapes contain neither a
numeric II nor startup cycles.  The namespace hashes feature schema, protocol,
candidate and analytical inputs, checkpoints, ensemble weights, uncertainty
gate, and selected static/gated mode.

Example:

```sh
python3 adapters/generate_amoeba_query_features.py \
  --manifest candidates.jsonl \
  --task-dfg Task_0=task-0.mlir \
  --task-dfg Task_1=task-1.mlir \
  --neura-opt /path/to/mlir-neura-opt \
  --architecture /path/to/architecture.yaml \
  --output query-features.json

python3 adapters/amoeba_cost_catalog.py \
  --manifest candidates.jsonl \
  --analytical-input query-features.json \
  --task-dfg Task_0=task-0.mlir \
  --task-dfg Task_1=task-1.mlir \
  --checkpoint seed1=model-1.pt \
  --checkpoint seed2=model-2.pt \
  --ensemble-report ensemble.json \
  --analytical-hybrid-report analytical-hybrid.json \
  --output costs.json \
  --timing-output latency.json

/path/to/mlir-amoeba-opt lowered-taskflow.mlir \
  --architecture-spec=/path/to/architecture_4x4.yaml \
  "--score-analytical-task-candidates=candidates=candidates.jsonl cost-file=costs.json output=scores.jsonl top-k=3" \
  -o scored.mlir
```

`adapters/evaluate_amoeba_scores.py` independently checks that the score file
covers every frozen program candidate, validates Amoeba's cache accounting,
and combines real per-query heuristic-mapper oracle labels with frontend
startup/trip counts to report top-1/top-k oracle recall, objective regret, and
the projected program-candidate mapper-call reduction.  It reports an actual
reduction only when the caller explicitly attests faithful program replay;
the current Amoeba interface is recorded as blocked instead.
`adapters/build_amoeba_query_oracle.py` turns standalone real heuristic-mapper
artifacts into its oracle input while verifying the recorded `x_tiles`,
`y_tiles`, analytical floor, and every physical placement coordinate.

The optional analytical-hybrid report is also validation-only and hash-locks
the same checkpoints, weights, gate exponent, and training manifest.  The
adapter adopts a rule only when its validation MAE is strictly lower than the
ensemble.  The current rule changes only the II mean to the analytical lower
bound when `RecMII > ResMII`; uncertainty and mapper-success probability stay
model-derived.  Rules keyed by the unknown true mapped II are never
deployable.
