# Benchmark compatibility and split plan

This file predeclares how benchmark sources should be used before a
paper-scale corpus is labelled.  It distinguishes an upstream benchmark from
a local compiler fixture and prevents repeated shapes or source translations
from being counted as independent workloads.

Upstream references: [PolyBench/C](https://www.cs.colostate.edu/~pouchet/software/polybench/polybench.html),
[MachSuite](https://github.com/breagen/MachSuite), and
[CGRA-Bench](https://github.com/tancheng/CGRA-Bench).

## Decision

Use generated DFGs for all fitting and model development, and the pinned
MachSuite revision only for the final real-program generalization claim:

```text
generated topology corpus -> fit/select/freeze model
MachSuite label-free pass  -> compatibility and prediction inputs
sealed predictions         -> one final mapper-label reveal
```

The primary frozen target is the version-pinned MachSuite checkout: it was
designed for accelerator research and differs materially from the generated
compute graphs. PolyBench/C and CGRA-Bench remain possible future secondary
external validations; their labels must not drive the current model.

Before collecting any frozen labels, record the upstream revision/archive
hash, source hashes, compiler flags, mapper revision/configuration, and the
complete requested kernel list.  Determine frontend compatibility without
reading mapper II labels.  Unsupported, failed, and timed-out entries remain
censored entries in the frozen manifest; they are not silently removed or
assigned an II.

## Locally available assets

| Asset | Exact local identity | Contents | Allowed role |
| --- | --- | --- | --- |
| Neura e2e fixtures | `third_party/neura` / `47b7e3a68c321075293e6fcb45fb3b1cabb93b88` | `axpy`, `bicg`, `fft`, `fir`, `gemm`, `gemv`, `histogram`, `relu`, `spmv` | compiler smoke tests and real development only |
| CGRA-Bench submodule | Neura `test/benchmark/CGRA-Bench` / `6729aaf225d0320e4e0d3b419e20483069a5a69b` | 15 kernel directories: `adpcm_coder`, `adpcm_decoder`, `bicg`, `blowfish`, `conv`, `dtw`, `fft`, `fir`, `gemm`, `histogram`, `latnrm`, `mvt`, `relu`, `spmv`, `susan` | mapping stress/development; overlap aliases required |
| Streaming-Bench nested checkout | CGRA-Bench `Streaming-Bench` / `333782f78d5475c8b33b11ff2b9ba9d75c93ca49` | `gcn`, `lu`, and `raytracing` applications with multiple subkernels | external challenge/development, grouped by parent application |
| Zeonica_Testbench | Neura `test/benchmark/Zeonica_Testbench` / `62389ec9f8e4e0f7f4988294213e71c6d2eebc85` | 13 generated-output/testbench directories synchronized from Neura | fixture validation only; never an independent benchmark set |
| Derived affine MLIR fixtures | sibling repository `neura-cgra-cost-model` / `82138932378a2b3cdd714d342c4751a5b7f8e47c`, `harness/front_affine` | 14 PolyBench-style affine functions: `adi`, `atax`, `bicg`, `doitgen`, `gemm`, `gemver`, `gesummv`, `jacobi_1d`, `mvt`, `nussinov`, `syrk`, `three_mm`, `trisolv`, `two_mm` | frontend development only; not an official PolyBench corpus |
| Official PolyBench/C | not vendored or version-pinned here | affine linear algebra, stencil, solver, data-mining, and medley kernels | future real transfer/secondary frozen set |
| Official MachSuite | `third_party/machsuite` / `6236e593012cb86b0d2f08d9fb9ba0411ff989b4` | 19 predeclared accelerator-oriented variants in `benchmarks/machsuite-v1.json` | primary frozen test only |

The local affine MLIR files are hand/generated compiler inputs inspired by
PolyBench algorithms.  Their presence does not establish that an unmodified,
official PolyBench release can pass the current frontend, and results on them
must not be labelled “PolyBench suite” results.

### Pinned MachSuite label-free preflight

The executable preflight uses Clang/LLVM 20 at `-O3` with vectorization,
unrolling, and lifetime markers disabled, extracts the inventory's named top
function, imports LLVM MLIR, lowers to Neura, and runs RecMII/ResMII analysis on the
fixed 4×4 architecture. The analysis-only pass directly calls the same C++ RecMII/
ResMII functions as the mapper; it invokes no mapper and records no
`compiled_ii`.

| Status | Variants |
| --- | --- |
| Ready (11/19) | `bfs/{bulk,queue}`, `gemm/{blocked,ncubed}`, `kmp/kmp`, `md/knn`, `spmv/{crs,ellpack}`, `stencil/{stencil2d,stencil3d}`, `viterbi/viterbi` |
| Lowering-censored (8/19) | `aes/aes` (`xor`), `backprop/backprop` (`sqrt/exp` calls), `fft/strided` (`xor`), `fft/transpose` (`uitofp`), `md/grid` (`umin`), `nw/nw` (canonicalize-live-in assertion), `sort/merge` (lifetime intrinsic), `sort/radix` (`ashr`) |

The headline accuracy denominator must state both the scored count and all 19
declared variants. Adding frontend support later defines a new protocol/tool
revision; it must not silently change this frozen result.

A local preflight confirmed that all 14 derived affine files reach Neura's
RecMII/ResMII analysis pass. With a deliberately short five-second mapper cap, six
completed (`adi`, `bicg`, `gemm`, `jacobi_1d`, `syrk`, `two_mm`) and eight were
censored by that time limit. This is only a compatibility diagnostic, not a
supported-kernel declaration or a model result. The sibling repository's nine
`front_real/e2e_*` files are identical empty-module stubs and supply zero usable
workloads.

Algorithms repeated across assets are one ancestry class for leakage
purposes.  For example, all `gemm` occurrences across Neura, CGRA-Bench,
derived affine inputs, and an official suite share a conservative `gemm`
lineage.  The same rule applies to `bicg`, `fft`, `fir`, `histogram`, `mvt`,
`relu`, and `spmv` where applicable.

## Generated training corpus

The implemented `motif-v3` stratum contains nine independently named
structural families:

```text
chain, fanout, reduction, diamond, random_dag, recurrence_chain,
predicated_diamond, memory_stream, pointer_chase
```

The fixed paper protocol requests 250 base DFGs per family and two target-shape
candidates per base, requiring at least 200 complete bases in every family.
The recurrence, predication, and pointer-chase families add real lowered
mechanisms but do not make the generated distribution equivalent to real
programs. The generation grid varies:

- actual graph size, with explicit operation-count bands;
- depth/width, fanout, reconvergence, and reduction arity;
- operation kinds and compatibility domains;
- recurrence, predication, pointer/load paths, and compute topology;
- rectangular active shape on one exact pinned Neura 4x4 architecture.

Every generated base has a distinct canonical labelled-graph hash.  Exact
canonical collisions are rejected.  A large row count from one fixed topology
with different constants is not a large DFG corpus.

Only bases with one successful label in both declared shape cells enter
CV and fitting. Partial successes and failures remain in the predeclared
denominator. The secondary shape is balanced across all eight non-4x4
rectangles, so completion fractions and per-shape counts are part of the
reported result.

## What counts as a new DFG

The canonical lowered graph, not a source-level knob name, decides whether the
exact DFG changed:

- Changing operation count, dependency edges, operation kinds, unroll factor,
  reduction arity, stencil radius, FIR tap count, FFT radix/stages, or a GEMM
  micro-kernel's accumulator structure may create a new exact DFG.
- Changing a literal value, dataset size, loop trip count, tensor extent, or
  launch count often leaves the static lowered DFG unchanged.  It is then the
  same DFG and must not be counted again.
- Changing CGRA rows/columns, masks, FU placement, register count, or mapper
  settings creates a new candidate for the same DFG, not a new DFG.
- Even when source transformations produce different exact DFG hashes, all
  variants of the same real algorithm remain in one leakage lineage.

Therefore “change op count or shape” has two different answers: changing the
DFG's operation count is useful topology augmentation; changing only the
hardware shape is useful candidate coverage but supplies no additional source
lineage.  Both are needed, and they must be counted separately.

## Predeclared evaluation matrix

Collect and report these experiments independently:

1. **Generated training/validation:** grouped by base DFG, plus leave-one-
   generator-family-out; all feature and hyperparameter decisions stop here.
2. **Architecture transfer within generated data:** hold out complete architecture families while
   retaining source-lineage grouping.
3. **Frozen MachSuite:** train once on the finalized generated corpus,
   serialize and hash the model, predict without mapper labels, then reveal
   labels once and preserve all censored attempts.
4. **Future PolyBench/C transfer:** report a separately predeclared compatible
   subset and do not use it to revise the MachSuite-tested model.

For every row of this matrix, publish requested/success/censored counts,
distinct canonical DFGs, leakage lineages, ranking queries, architecture
candidates, exact duplicates, and the mapper/source revisions.  Do not pool
generated held-out rows into the headline real-generalization metric.
