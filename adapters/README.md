# Adapters

Adapters own all compiler- and mapper-specific behavior. They may invoke an
external toolchain to lower programs, read architecture descriptions, collect
pre-mapping features, and obtain labels from an unchanged mapper.

The Neura adapter is migrated from the original cost-model experiment. It
remains useful for corpus reproduction, but its textual MLIR features are not
part of the generic package API.

The repository pins Neura at `third_party/neura`. Initialize that submodule
with `git submodule update --init third_party/neura`; the adapter uses it by
default, while `--neura-root` and `NEURA_ROOT` remain explicit overrides. Neura's
nested benchmark submodules are not needed for ordinary feature collection.

The pinned Neura `main` revision is the authoritative producer for Model 1.
Its fixed lower-bound contract is `max(RecMII, ResMII)`; no unpublished
placement or route-bound extension is required. The analysis-only pass calls
the mapper's shared C++ recurrence/resource functions rather than reproducing
their formulas in Python.

An adapter must never turn a timeout into a compiled-II label, and must record
the source family, architecture identity, compiler revision, mapper revision,
and mapper configuration needed to reproduce every successful label.

The Neura adapter recomputes `baseline_lb = max(rec_mii, res_mii)` and records
`lower_bound_source=rec_res_max_v1`. An explicitly supplied `baseline_lb` or
portable `lower_bound` must agree exactly. RouteMII, RegMII, MemMII, and
`analytical_ii` are absent from the main-based Model 1 record.
Each RecMII/ResMII component is a non-negative integer, while the derived
lower bound must be positive; one component may therefore be zero, but not
both.

For point inference, `--model-report MODEL --predict-fixture NAME=PATH` loads a
hash-checked residual-Ridge artifact and runs only the analysis-only RecMII/
ResMII pass plus static DFG feature extraction. This mode rejects label-collection options,
skips every holdout and fit, and never invokes the heuristic mapper. Its report
contains a continuous compiled-II estimate, the raw/constrained residual,
authoritative lower-bound source, exact model inputs, model identity, and any
empirical interval. If any requested candidate cannot be predicted, the report
is marked incomplete and the adapter exits nonzero.
See [II_PREDICTION.md](../II_PREDICTION.md) for the exact contract.

Architecture/suite/generator nested holdout is opt-in with repeated
`--metadata-holdout-key architecture_id|suite|generator_family`; it can be expensive when every
sample has a distinct domain identity. Source-lineage weights remain in force
inside these alternate splits.

Subprocess diagnostics are written to a disk-backed temporary stream and only
a bounded head/tail excerpt enters the report. Timeouts and nonzero exits are
stored under censored/failure records, never as numerical II labels.

Rows imported through the legacy `--input-report` path are always marked
`rec_res_evidence=imported_report_unverified`, even if the source report claims
otherwise, and their report is marked unverified/exploratory. The frozen
training command rejects all input-report reuse; its generated rows must carry
the hashed `rec_res_mii_info` artifact produced in the same direct run.

## Deterministic motif corpora

The historical/default `--motif-generator-version motif-v3` mode uses
`neura_motifs.py` to generate nine lowered families: `chain`, `fanout`,
`reduction`, `diamond`, `random_dag`, `recurrence_chain`,
`predicated_diamond`, `memory_stream`, and `pointer_chase`. By default the
declared shape population is every rectangle from 2x2 through 4x4; each base
runs on 4x4 plus one balanced secondary shape. Repeat `--motif-shape ROWSxCOLS`
to narrow this population. Every candidate references the exact pinned Neura
4x4 YAML, and shape is applied through the existing pass dimension options.
This nine-shape population remains the frozen training contract. Prediction
also accepts 1x1 and a canonical 1x2 strip as stress candidates. They are not
training-supported or automatically ranked: the pinned mapper searches only
through II 20, and tiny targets frequently have a Rec/Res lower bound above
that ceiling. Such attempts are censored, never converted into numeric labels.

The old `--samples` switch remains a legacy narrow random DAG and is not
implicitly folded into the motif corpus.  Motif source files are materialized
first, then `corpus-manifest.json` is atomically predeclared before any
compiler subprocess.  It records source/canonical hashes, root/base seeds,
operation counts, architecture identity, candidate IDs, and lineage
`generated/<generator_version>/<motif>/<base_id>`.  Each base DFG keeps the
same source hash across shapes and architecture variants.  Success records
point to cost/mapped artifacts; failed or timed-out candidates remain
censored and have no numeric label.  `--motif-jobs N` (default `1`) bounds
candidate-level parallelism, while each candidate's Rec/Res analysis and
mapper invocation remain serial.  The main coordinator alone updates the
manifest; terminal records are atomically checkpointed after completion
batches, while the manifest records remain in declaration order. For
generator-family holdout use
`--metadata-holdout-key generator_family`.

`--motif-generator-version motif-v4` selects `neura_motifs_v4.py` without
changing any v3 constants used by the existing MachSuite freezer. V4 has six
path contexts (`compute`, `recurrence`, `predicated`, `memory`, `pointer`, and
`mixed`) and crosses each with five topology-pressure profiles and three fixed
operation bands. Its shape design uses 4x4 plus one balanced secondary block;
transpose rectangles occur together on the same base DFG. Use
`--motif-predeclare-only` for the mandatory first phase: it materializes every
input and writes the complete manifest, then returns before any compiler or
mapper invocation. Resume the reviewed manifest separately to collect labels.

For v4, the report keeps analysis-valid, `LB <= 20`, mapper-attempted,
successful-label, and feasible-censored denominators separate. If `LB > 20`,
the mapper is not invoked and no numeric failure label is invented. Formal
acceptance additionally requires strict generator-family transfer improvement,
nonzero positive-residual recall in every family, improved positive-subset and
shape-balanced MAE, ranking non-degradation, distributed positive mechanisms,
and 80% complete coverage in every family-by-shape, family-by-profile, and
family-by-operation-band marginal cell.
The exact label-blind declaration is `../protocols/motif-v4.json`.

Use `--motif-resume` with the same output directory to continue a partial
collection; it is incompatible with `--clean`.  Resume validates the
manifest schema, generator configuration, timeout, compiler SHA-256, candidate
inputs, and source/architecture/cost/mapped artifact path/hash identities.
Cached successes are rebuilt from those artifacts without a compiler
subprocess, censored candidates are skipped without retrying, and a corrupt
cache fails before invocation.  A legacy `running` record is normalized to
`declared` before retry.  If every candidate is terminal, resume neither
requires nor probes the compiler. Omitted generator choices are recovered from
the manifest. V4 runs the required `generator_family` holdout automatically;
other invocation-local training/evaluation options must be repeated. SIGINT stops
scheduling, drains at most the configured in-flight candidates, checkpoints
atomically, exits 130, and does not fit a model.

This is a single-coordinator/single-writer manifest contract; no cross-process
lock is provided.  Do not run two fresh or resume processes concurrently
against the same output directory.

The recurrence, predicated-control, and pointer/load paths remain structural
generators rather than real workload semantics. V3 requests 250 bases per
family and admits a base only after both of its declared shape cells succeed.
V4 also requests 250 per family and admits a base only after every cell in its
two- or three-shape block succeeds. Partial labels stay auditable outside the
fit. Official benchmark suites remain necessary for real-program evidence.
The full paper protocol is in `../CORPUS_PROTOCOL.md`.

## Frozen MachSuite adapter

`machsuite_frozen.py` consumes the 19-entry inventory in
`../benchmarks/machsuite-v1.json`. `preflight` compiles, extracts, imports,
lowers, and runs only analysis-only RecMII/ResMII; `predict` accepts only a generated-only
model sealed by `freeze-model`; `predict` re-derives every feature from the
recorded artifacts, rejects any canonical DFG shared with the exact v2
training-hash set, and `reveal` verifies all hashes again before it can run the
heuristic mapper. Mutated preflight bytes are rejected, and unsupported
variants remain censored in the declared-suite denominator. Publish or
timestamp the seal externally before reveal when chronology must be proved.
