# II-predictor corpus protocol

This document defines the corpus that should support a paper claim about
predicting the initiation interval (`compiled_ii`) of a fixed CGRA mapper.  It
is intentionally separate from historical local exploratory artifacts. Those
files are ignored, are not part of a clean checkout, and are not frozen
benchmarks.

## Scope and labels

The prediction target is the integer `compiled_ii` returned by one explicitly
identified mapper revision, mapper configuration, compiler/lowering pipeline,
and architecture candidate.  A successful mapper invocation produces one
label.  A timeout, nonzero exit, invalid lowered IR, or missing `compiled_ii`
is a censored candidate and is recorded in `corpus-manifest.json`; it is never
converted to a numeric label.

Each sample records the source/canonical DFG hash, candidate ID, architecture
identity, mapper revision/configuration, generator family/version (if
generated), root/base seed, operation count, and lineage.  The target is
modelled as a residual above the fixed Model-1 lower-bound contract:

```text
lower_bound = max(RecMII, ResMII)
residual = compiled_ii - lower_bound
```

No placement, routing, memory, or learned estimate enters the floor. The
primary feature vector also excludes `lower_bound`, `rec_mii`, and `res_mii`,
so the fixed floor and learned residual have separate roles. Solver-branch
MemMII/RegMII/RouteMII-style fields are absent from the main-based Model 1
record.

The manifest keeps four non-interchangeable identities: leakage lineage for
splitting, base DFG/ranking query for comparing candidates, candidate ID for
one architecture/mapper setting within that DFG, and globally unique sample ID
for one recorded attempt. Candidate IDs may repeat across ranking queries. A
ranking query may not span leakage lineages.

## Strata and benchmark roles

The paper protocol has two roles with a hard boundary:

1. **Generated random-DAG and motif training/development.** Connected random
   DAGs provide broad topology variation; structured motifs provide controlled
   compute, recurrence, predication, and pointer/memory-path coverage. Shapes
   and FU variants at one explicitly
   selected register capacity, plus any future mask/register variants and
   mapper attempts of one base DFG, remain one lineage.
2. **MachSuite frozen test.** The complete 19-variant inventory is fixed before
   feature/model selection. Label-free lowering and feature compatibility may
   be checked, but mapper labels are revealed only after the generated-only
   model and every prediction are hash-sealed. The seal must be externally
   published/timestamped before label reveal for a chronology claim; local
   hashes alone provide tamper evidence, not trusted time.

Recommended benchmark roles are:

- **MachSuite:** application-oriented kernels with a different source and
  operation distribution. It is the primary frozen test at the pinned
  revision; it is not training or development data.
- **PolyBench/C:** a useful future secondary external validation suite for
  affine loops, stencils, linear algebra, and reductions. It must not influence
  the current MachSuite headline result.
- **CGRA-Bench (or an equivalent CGRA-specific suite):** mapping-relevant
  kernels and architecture stress cases. Use it as a real development set or
  external transfer test only after excluding algorithms already represented
  in training; matching algorithms such as GEMM/BiCG are not independent.

The exact suite version, commit/archive hash, compiler flags, unsupported
kernel list, and successful/censored counts must be reported.  If one suite
cannot be lowered by the pinned Neura frontend, it remains a declared but
unsupported candidate rather than being silently replaced by a generated
kernel.

The exact current local inventory and primary MachSuite/secondary future
PolyBench split are recorded in `BENCHMARK_COMPATIBILITY.md`. In particular,
the PolyBench-style MLIR files in the sibling experiment repository are
derived frontend fixtures, not an official version-pinned PolyBench corpus.

The current Neura checkout is not yet an official-suite corpus. Its e2e tests
cover roughly nine conservative algorithm lineages (`axpy`, `bicg`, `fft`,
`fir`, `gemm`, `gemv`, `histogram`, `relu`, `spmv`) and its nested CGRA-Bench
checkout adds candidate kernels, but overlapping algorithms are not independent
test samples. No complete, version-pinned PolyBench/C corpus is currently wired
to this adapter. The MachSuite inventory is now pinned and executable; its
label-free preflight records 11/19 ready variants and 8/19 lowering-censored
variants. All 19 remain in the reported denominator.

This allocation is inspired by LISA's use of generated weakly connected DFGs
for training followed by real applications for evaluation. It is not a LISA
reproduction: LISA learns node/edge mapping guidance with a GNN, whereas Model
1 learns one mapper's final-II residual with Ridge. The present MachSuite
preflight status and label-free structural covariates were visible during
protocol hardening; no MachSuite mapper labels were accessed. The correct claim
is therefore label-blind, not covariate-blind.

## Generated motif corpus

The current `motif-v2` implementation in `adapters/neura_motifs.py` emits
already-lowered DFGs for nine families:

```text
chain, fanout, reduction, diamond, mixed, random_dag,
recurrence_chain, predicated_diamond, pointer_chase
```

`chain` exercises dependence depth, `fanout` exercises broadcast pressure,
`reduction` is a binary reduction tree, `diamond` is split/join
reconvergence, `mixed` combines independent inputs, fanout, and reconvergence,
and `random_dag` samples operation kinds and dependency edges while guaranteeing
weak connectivity and acyclicity. `recurrence_chain` emits a real
reserve/phi/arithmetic/control backedge, `predicated_diamond` emits predicate
generation, complementary grants and reconvergence, and `pointer_chase` emits
argument-backed and indirect GEP/load paths plus a loop backedge. These are
lowered structural generators, not source-level workload semantics; real-suite
evaluation remains necessary.

Every invocation has an explicit root `--seed`.  A base is identified by
`(generator_version, motif, root_seed, base_index)` and its lineage is:

```text
generated/<generator_version>/<motif>/<base_id>
```

The same source text, source SHA-256, and canonical DFG hash are reused for
all current shape/FU variants of that base at the selected register capacity.
`generator_family` is motif specific (`generated/motif/chain`, for example), while `generator_type`
(`generated/motif`) may be used for aggregate reporting.  Thus a holdout by
`generator_family` can leave out one motif family, and a holdout by `lineage`
can leave out one base DFG.

Operation counts are stratified over `8--15`, `16--31`, and `32--48` for most
families; the formal recurrence bands end at 32 to keep the mapper-feasible
corpus from being dominated by very large RecMII. Larger direct-generator
limits are explicit stress tests and are not silently included in the frozen
distribution. Operation count is checked against emitted arithmetic structure.
Changing only a loop trip count, launch count, input size, or constant literal
is **not** a new DFG. Canonical hashes normalize constant literals; canonical
collisions are deterministically resampled and rejected across families by the
coverage contract.

Shapes and architectural variants are attempts on the same source DFG, not
new source groups.  A future expanded corpus should vary at least array shape,
valid-tile masks, FU domain/register capacities, links/bandwidth, and memory
placement while retaining architecture IDs and YAML hashes.

For source benchmarks, vary parameters only when they change the lowered DFG:
FIR tap count, reduction width/tree arity, stencil radius, GEMM micro-kernel
`MR x NR`, FFT radix/stage, unroll factor, vector width, or accumulator count
are meaningful examples. Merely changing matrix dimensions or loop trip counts
usually leaves an unrolled-off static DFG unchanged and is one workload, not a
new independent graph. All such variants remain in the same algorithm lineage
even when their canonical DFG hashes differ.

## Source-family and variant rules

All compiler/lowering variants of one source algorithm belong to one lineage
for generalization claims unless the paper explicitly studies compiler
transfer.  This includes scalar/vector forms, integer/float type variants,
unrolled forms, tile shapes, and valid-tile masks.  A source variant that is
semantically the same algorithm is not an independent training group.

Generated mutations follow the same rule: all candidates from one base DFG
share one lineage; distinct base DFGs have distinct canonical hashes and
lineages. Rows sharing one observation identity under the active model
projection are deduplicated for training weight and hyperparameter selection,
while the manifest retains every attempted candidate.

The motif materializer rejects two distinct bases anywhere in one generated
corpus when their canonical DFG hashes collide; this fail-fast check prevents
accidental duplicate bases from masquerading as independent training lineages.

## Manifest and reproducibility

For motif collection, all source and architecture files are materialized and
`output-dir/corpus-manifest.json` is atomically written in `predeclared` state
before any compiler subprocess.  Thus every candidate is in the manifest before
the first Rec/Res analysis or mapper invocation.  The manifest lists every
candidate's motif, seed, operation count, shape, architecture parameters,
hashes, and lineage.  `--motif-jobs N` bounds candidate-level parallelism
(default `1`); within each candidate, the shared Rec/Res analysis and heuristic
mapper stages remain serial.  Only the coordinator/main thread mutates the
manifest.

```text
manifest predeclared; candidate declared -> success
                                      \-> censored
```

The stable manifest contains terminal `success` or `censored` records and is
checkpointed atomically after completion batches in deterministic candidate
order.  Successful records point to hashed Rec/Res-analysis and mapped
artifacts and the corresponding `sample_id`; frozen-model validation re-parses
both and requires identical RecMII/ResMII. Censored records contain
stage/failure status and no label. There is no persistent `running` state in
the current collector. On `--motif-resume`, a legacy `running` record is
normalized back to `declared` with `stage=predeclared` before scheduling and
can be retried.

`--motif-resume` validates the manifest schema, generator configuration,
timeout, compiler SHA-256, candidate inputs, and recorded artifact
identities/path/hash values before invoking a subprocess. Cached successes are
rebuilt from their hashed source, architecture, Rec/Res-cost, and mapped
artifacts without invoking the compiler; censored candidates are skipped and
are never retried. A corrupt cached record fails before any subprocess. If a
resume has no non-terminal candidates, it neither requires nor probes the
compiler. `--clean` and `--motif-resume` are mutually exclusive.

On SIGINT, the coordinator stops scheduling new candidates, drains at most
`N` in-flight candidates, atomically checkpoints the stable manifest, and exits
with status 130 without fitting or emitting a training report. Resume can
continue the remaining `predeclared` candidates.

The manifest is a single-coordinator/single-writer contract; there is currently
no cross-process lock. Do not start two fresh or resume processes against the
same output directory concurrently.

## Evaluation and ablations

Do not use random row splits as the primary result. Within generated data, use nested grouped model
selection: each outer or inner invocation independently uses leave-one-lineage-
out for at most 20 current groups and up to 10 deterministic stratified group
folds above that threshold. Then report:

- generated-only training/validation by base lineage;
- whole-generator-family holdout as a topology-transfer diagnostic;
- architecture-family holdout (YAML/topology identity not seen during fit);
- the frozen MachSuite result whose labels were not inspected during feature,
  model, generator, dead-zone, or interval selection.

Generated samples provide fitting and structural coverage; they do not count
as real-program evidence. Report the number of distinct base DFGs, lineages,
candidates, successful labels, censored attempts, and exact/canonical
duplicates for every ablation.

Every primary-training sample declares `training_stratum=generated`. The
implementation balances lineages and distinct observations while keeping total
weight equal to the number of lineages. See `TRAINING.md` for the exact formula.

## Recommended scale and current boundary

The frozen study fixes **nine families × 250 requested bases × six candidates**
(three shapes times two FU layouts): 2,250 predeclared base DFGs and 13,500
attempts. A base is fit-eligible only if all six cells succeed. The gate requires
at least 200 complete bases in every family, or 1,800 lineages and 10,800
training rows. Partial successes remain visible in `labelled_samples`; every
failure remains in the manifest.

This complete-case design makes architecture comparisons balanced but selects
for graphs that the current mapper completes on all six candidates. Report
requested, successful, complete, and censored counts per family and failure
stage; do not describe the fitted subset as an unbiased sample of all generated
graphs. The old `--samples` option remains a legacy narrow random-DAG generator
and is outside the frozen corpus.

The frozen-model command additionally requires nested base-lineage selection,
a whole-generator-family holdout, and strictly lower generated nested-lineage
macro MAE for Ridge than for the Rec/Res floor. A small override produces an
artifact that the frozen MachSuite predictor refuses. A large row count from
one template does not substitute for distinct canonical DFGs or lineages.
