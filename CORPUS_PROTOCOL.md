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
so the analytical floor and learned residual have separate roles. Estimated
MemMII/RegMII/RouteMII-style values may be retained as diagnostics or training-
only ablations, but are not labels or pruning bounds.

The manifest keeps four non-interchangeable identities: leakage lineage for
splitting, base DFG/ranking query for comparing candidates, candidate ID for
one architecture/mapper setting within that DFG, and globally unique sample ID
for one recorded attempt. Candidate IDs may repeat across ranking queries. A
ranking query may not span leakage lineages.

## Strata and benchmark roles

The paper protocol has two roles with a hard boundary:

1. **Generated random-DAG and motif training/development.** Connected random
   DAGs provide broad topology variation; deterministic compute motifs provide
   controlled mechanism coverage. Shapes and FU variants at one explicitly
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

## Generated motif corpus

The current implementation in `adapters/neura_motifs.py` emits already-lowered
compute DFGs for six families:

```text
chain, fanout, reduction, diamond, mixed, random_dag
```

`chain` exercises dependence depth, `fanout` exercises broadcast pressure,
`reduction` is a binary reduction tree, `diamond` is split/join
reconvergence, `mixed` combines independent inputs, fanout, and reconvergence,
and `random_dag` samples operation kinds and dependency edges while guaranteeing
weak connectivity and acyclicity. They use only constant, data-movement, add,
and multiply operations. They do **not** claim coverage of memory or control
flow.
Memory/recurrence and control-heavy coverage must come from the frontend
generator and real suites (or a separately versioned generator family).

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

Operation counts are sampled cyclically from the current explicit bands
`8--15`, `16--31`, and `32--48`. The operation count is part of the source
generator input and is checked against the actual number of emitted binary
compute operations. Changing only a loop trip count, launch count, or benchmark
input size or constant literal is **not** a new DFG and must not receive a new
lineage. Canonical hashes normalize constant literals. A new DFG must change
the labelled operation/dependency graph (and receive a new base seed or source
hash); generated canonical collisions are deterministically resampled and
still rejected at materialization.

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
before the first mapper subprocess.  The manifest lists every candidate's
motif, seed, operation count, shape, architecture parameters, hashes, and
lineage.  It is updated atomically after each cost-model and mapper stage:

```text
declared -> running(cost-model) -> running(mapper) -> success
                                           \-> censored
```

Successful records point to cost/mapped artifacts and the corresponding
`sample_id`; censored records contain stage/failure status and no label.
Interrupted runs may leave `running` records and must be resumed or reported
as incomplete rather than treated as successful.

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

For a final study, target roughly **8 independent motif/generator families ×
200--300 base DFGs × 4--6 architectural attempts**, plus the separately
reserved MachSuite test. This is a recommendation, not evidence currently
available in this checkout. The present code implements six compute motif/
random-DAG families, three default shapes, and two deterministic architecture
variants;
it defaults to zero motif samples and does not run a large collection unless
`--motif-samples-per-family N` is explicitly supplied.  Its old `--samples`
option remains a legacy narrow random-DAG generator and is not automatically
mixed into the motif corpus.

The frozen-model command fails closed below 1,000 distinct generated base DFGs
or six generator families and also requires the training report's whole-
generator-family holdout gate. A small smoke override produces an artifact
that the frozen MachSuite predictor refuses.

Before claiming the recommended scale, add independently versioned memory,
recurrence, and control generators, validate their lowering success, collect
source/DFG provenance, and retain the untouched MachSuite test. A large
number of rows from one template does not substitute for independent base
DFGs or source lineages.
