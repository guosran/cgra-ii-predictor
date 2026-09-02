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

## Deterministic motif corpus

The opt-in `--motif-samples-per-family N` mode uses
`neura_motifs.py` to generate nine lowered families: `chain`, `fanout`,
`reduction`, `diamond`, `mixed`, `random_dag`, `recurrence_chain`,
`predicated_diamond`, and `pointer_chase`. Repeat `--motif-shape
ROWSxCOLS` (default `3x3`, `3x4`, `4x4`) and select families with repeated or
comma-separated `--motif`/`--motifs` options.  `--motif-architecture-variant`
selects the deterministic homogeneous and split-domain attempts.

The old `--samples` switch remains a legacy narrow random DAG and is not
implicitly folded into the motif corpus.  Motif source files are materialized
first, then `corpus-manifest.json` is atomically predeclared before the first
mapper invocation.  It records source/canonical hashes, root/base seeds,
operation counts, architecture identity, candidate IDs, and lineage
`generated/<generator_version>/<motif>/<base_id>`.  Each base DFG keeps the
same source hash across shapes and architecture variants.  Success records
point to cost/mapped artifacts; failed or timed-out candidates remain
censored and have no numeric label.  For generator-family holdout use
`--metadata-holdout-key generator_family`.

The three v2 additions contain real lowered recurrence, predicated-control,
and pointer/load structure, but they remain structural generators rather than
real workload semantics. The formal protocol requests 250 bases per family
and admits a base to fitting only after all three shapes and both architecture
variants succeed; partial labels stay auditable outside the fit. Official
benchmark suites remain necessary for real-program evidence. The full paper
protocol is in `../CORPUS_PROTOCOL.md`.

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
