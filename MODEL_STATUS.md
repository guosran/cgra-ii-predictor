# Model 1 status

Last updated: 2026-09-03

Model 1 is an offline predictor of the `compiled_ii` produced by one recorded
Neura heuristic mapper. It neither changes the mapper nor proves that its point
estimate is feasible.

## Finalized mathematical contract

The only hard lower bound is:

```text
LB = max(RecMII, ResMII)
target_residual = compiled_ii - LB
predicted_compiled_ii = LB + max(0, predicted_residual)
```

Every newly collected row records `lower_bound_source=rec_res_max_v1`.
Explicit `lower_bound` or `baseline_lb` aliases must equal the Rec/Res maximum.
Each RecMII/ResMII component is a non-negative integer; the derived `LB` must
remain a positive integer, so `(0,3)` and `(3,0)` are valid while `(0,0)` is
rejected.
Portable training and prediction loaders require both integer components and
reject `LB`, `rec_mii`, or `res_mii` if any model artifact selects them as
features; this is a global contract, not only a Neura-adapter convention.
The former solver-branch MemMII, RegMII, RouteMII, and `analytical_ii` fields
have been removed from the Model 1 record. Placement and mandatory-route
lower-bound extensions are not part of Model 1.

This choice is intentionally narrower than the strongest bound that could be
implemented. It matches Neura `main` and keeps the proof contract small. A
minimal analysis-only compiler pass calls the same C++ recurrence/resource
functions as the mapper; the predictor does not reimplement those formulas.
An experimental local implementation of
placement-Hall and mandatory-route-cut bounds was found to be genuinely
stronger than Rec/Res on focused examples, but it was not retained: the
uncommitted work is recoverable from Neura `stash@{0}` and is absent from every
commit in this work.

## Why the lower bound and learned features do not conflict

The primary Ridge model does not receive `LB`, `rec_mii`, or `res_mii` as a
feature. The floor is applied exactly once after Ridge predicts the remaining
mapper gap. The 13 primary features are all available before mapping:

```text
semantic_depth
semantic_width
sources
semantic_branch_density
semantic_cut_fraction
multi_input_density
memory_op_density
pointer_path_fraction
memory_path_fraction
compute_fu_peak_pressure
memory_fu_pressure
routing_cut_pressure
register_pressure
```

This separation avoids label leakage, double counting, and the interpretability
problem of claiming an analytical term both as a theorem and as a learned
coefficient. Cost-model diagnostics may be evaluated as generated-training
ablations, but a MachSuite result may not be used to select them.
Raw graph-size counts were deliberately not kept in the fitted vector: a
label-free MachSuite preflight showed that they extrapolated beyond the
generated range. The retained densities, path fractions, and resource
pressures keep all 11 preflight-ready 4x4 DFGs inside every univariate training
min/max. This is only a support diagnostic, not an accuracy result; those
covariates were visible during design, so the protocol is label-blind rather
than covariate-blind.

## Ridge training

For weighted standardized features `z_i`, Model 1 solves:

```text
min_beta sum_i w_i (target_residual_i - beta_0 - beta z_i)^2
         + lambda ||beta||_2^2
```

The intercept is not penalized. Ridge strength and the small-residual dead zone
are selected inside nested generated-lineage validation. Rows are weighted so
each generated base-DFG lineage contributes equal total mass; exact or model-
indistinguishable duplicates share one observation's mass. Random row splits
are diagnostics only.

The exact standardization, weighting, nested split, and post-processing rules
are in `TRAINING.md` and `II_PREDICTION.md`.

## Training corpus decision

All primary fit and selection data are generated DFGs. The implemented
`motif-v3` generator contains nine families:

```text
chain, fanout, reduction, diamond, random_dag, recurrence_chain,
predicated_diamond, memory_stream, pointer_chase
```

It varies real operation count, dependency edges, depth/width, fanout,
reconvergence, recurrence cycles, predicated control, streaming memory,
pointer/load paths, and target rectangles. All target shapes of one base DFG share one
leakage lineage and canonical DFG identity. A manifest is written before any
mapper invocation; failures and timeouts are censored rather than fabricated
as numeric II labels.

The frozen protocol requests exactly 250 bases per family. Every attempt uses
the exact pinned Neura 4x4 YAML; target resources are selected only with the
existing `x-tiles`/`y-tiles` pass options. Supported target rectangles are all
nine shapes from 2x2 through 4x4. Each base is assigned 4x4 plus one secondary
shape in balanced round-robin order, giving 2,250 bases and 4,500 candidates.
A base contributes its paired rows only when both attempts succeed. At least
200 complete bases per family are required, hence at least 1,800 fitted
lineages and 3,600 rows.
Each family/shape cell must separately retain 80% of its declarations (25 or
26 complete bases for secondary cells), preventing aggregate coverage from
hiding an unsupported target size.
Successful rows from incomplete lineages remain in `labelled_samples`, and all
attempts remain in the manifest denominator. This complete-case rule avoids an
unbalanced architecture grid but conditions training on DFGs that map across
both declared candidates; per-family completion fractions and failure stages must be
reported as a possible selection bias.

`valid-tiles` masks are deliberately excluded: Neura main currently applies
false entries before true overrides can restore them. Rectangular dimension
overrides are shared by the analysis and mapping passes and are the reusable,
verified path. At inference the predictor records 1x1, a canonical 1x2 strip,
and all nine trained rectangles. The tiny shapes are marked as untrained
stress candidates and excluded from the automatic Pareto frontier; they do not
retroactively change the frozen v3 corpus or artifact. It also records whether
the Rec/Res floor fits below the pinned mapper's II ceiling of 20. The
predictor does not declare a predicted candidate feasible.

The same direct, clean-checkout run must use the predeclared Ridge/dead-zone
grid and whole-generator-family holdout. The generated nested-lineage Ridge
macro MAE must strictly improve on the Rec/Res-only baseline or no final model
can be frozen; whole-generator-family Ridge may tie but not degrade the floor,
and the 13 features plus intercept must be full rank. A small override produces
only a smoke artifact that frozen `predict` rejects.

The completed formal run predeclared 2,250 distinct bases and 4,500 candidates.
It retained 4,482 successful labels and 18 mapper-censored attempts; 2,232
complete paired lineages (4,464 rows) entered selection and the final fit. Every
family and family/shape coverage cell passed its 80% threshold. The selected
model is Ridge with `lambda=3` and residual dead zone `0.75`; its intercept plus
13-feature design is full rank 14/14 with condition number 48.01. In nested
unseen-base-lineage validation, macro MAE fell from 0.17249 for the Rec/Res
floor to 0.13030 for Ridge, a 24.46% relative reduction. Leave-one-generator-
family-out macro MAE tied the floor at 0.17393. That tie passes the declared
non-degradation gate but is not evidence that the residual transfers to an
unseen topology family.

## Why the current improvement is limited

The 24.46% headline reduction is real under the declared nested lineage split,
but it is narrow rather than broadly portable:

- 3,888 of 4,464 fitted rows (87.10%) have exactly zero residual. Of the 576
  positive rows, 334 (58.0%) are `random_dag`; recurrence has no positive row.
- Almost all gain comes from `random_dag` (macro MAE 1.0776 to 0.6965). Chain,
  diamond, memory, pointer, predicated, recurrence, and reduction are ties;
  fanout improves only from 0.1169 to 0.1137.
- `random_dag` accounts for 99.14% of the total absolute-error reduction. In
  leave-one-generator-family-out validation, all 4,464 Ridge predictions are
  reduced to the Rec/Res floor by the selected dead zones, missing every one
  of the 576 positive residuals.
- Ridge improves the larger 3x3-through-4x4 shapes, but degrades 2x2, 2x3,
  2x4, and 3x2. For example 2x2 MAE rises from 0.0253 to 0.0446, whereas 4x4
  falls from 0.2599 to 0.1846.
- Tie-aware within-DFG shape-ranking accuracy changes from 0.97413 for the
  Rec/Res floor to 0.97350 for Ridge. Point-MAE selection therefore did not
  improve the actual shape-ranking objective.
- Raw Ridge residuals are negative on 37.86% of rows and must be clipped. This
  is a symptom of fitting one linear mean to a zero-inflated, heterogeneous
  target, not additional lower-bound information.

The immediate cause is the dataset/model interaction. Current motif templates
make congestion residuals rare and strongly correlated with generator family;
the 13 scalar summaries lose the local reconvergence, placement conflict, and
route competition that create the mapper gap. Half the fitted rows are 4x4,
while each secondary shape is sparse, so the loss is not shape-balanced.
Nested lineage validation prevents duplicate leakage, but it cannot create
topology diversity that the generator does not contain. The leave-one-family-
out tie confirms this limitation. The current predeclared transfer gate allows
a tie, so it certifies non-degradation only; the next protocol should require
strict transfer improvement and reject a model that predicts the floor for
every held-out-family row.

Adding tiny shapes directly to fitting would worsen the problem. A 54-base
pilot across nine families and three operation-count strata produced no
positive residual on any successful 1x1/1x2/2x1 mapping, while 71 of 162
attempts had no mapper label. For the full 250-base distribution, only 0/250
pointer-chase bases on 1x1 and 122/250 on a two-tile strip even have
`LB <= 20`; other families are also non-randomly censored. These shapes are
therefore prediction/stress records, not new training evidence.

The next model iteration should first create a predeclared mapping-free v4
generator that varies cutwidth, reconvergence, fanout, live ranges, control,
and memory pressure within every family at comparable Rec/Res floors. Selection
must use shape-balanced metrics and require ranking not to regress. Only after
positive residuals are distributed across families is it meaningful to compare
a hurdle model (feasibility/zero-gap classification plus conditional residual
regression) or a graph model against Ridge. Increasing model complexity before
fixing the missing signal would mostly learn generator identity more sharply.

Freezing verifies the pinned clean Neura checkout, mapper-binary hash,
deterministically regenerated source, mapped-artifact hash and embedded
`compiled_ii`/RecMII/ResMII, plus the label-free Rec/Res artifact hash and its
`rec_res_mii_info` marker. It then requires analyzer and mapper Rec/Res facts
to match exactly before reproducing the selected Ridge model. It does not rerun
every expensive training mapping a second time; the archived analysis/mapped
artifacts plus their provenance are the training-label evidence.
The frozen container is now v2: it records the exact sorted set of canonical
training DFG hashes and its digest. Both prediction sealing and label reveal
reject overlap between that set and every ready MachSuite DFG. Generic v1
loading remains available for historical/non-frozen use, but v1 cannot enter
the frozen MachSuite path.

## Frozen MachSuite test

MachSuite is a Git submodule pinned to:

```text
6236e593012cb86b0d2f08d9fb9ba0411ff989b4
```

`benchmarks/machsuite-v1.json` predeclares all 19 source variants and folds
closely related variants into 12 conservative algorithm lineages. The primary
test uses one fixed 4×4 architecture candidate per variant; repeated shapes do
not inflate the headline sample count.

`adapters/machsuite_frozen.py` enforces four tamper-evident protocol boundaries:

1. `freeze-model` accepts only a generated-only training report using the
   structure-only feature contract and removes labelled rows from the artifact.
2. `preflight` compiles/lowers all 19 variants and runs only the compiler's
   analysis-only RecMII/ResMII pass. It contains no mapper invocation and no
   `compiled_ii`.
3. `predict` rejects exploratory or mismatched models, emits predictions, and
   seals the model, manifest, candidate-set, predictor, and prediction hashes.
4. `reveal` verifies every seal plus source, lowered DFG, architecture, Neura,
   and mapper-binary identity before it can invoke the heuristic mapper. Labels
   and evaluation are written to new files; any preflight mutation is rejected.

SHA-256 alone is not a trusted timestamp and cannot cryptographically prove
that the author did not recreate all files after seeing labels. For a paper
claim, publish the prediction seal in Git, an artifact registry, or an external
timestamp service before `reveal`; the local workflow then verifies that the
published bytes are exactly the bytes being evaluated.

The verified label-free preflight result is:

| Status | Count | Variants |
| --- | ---: | --- |
| Ready | 11 | BFS bulk/queue, GEMM blocked/ncubed, KMP, MD-KNN, SpMV CRS/ELLPACK, stencil 2D/3D, Viterbi |
| Lowering-censored | 8 | AES, backprop, FFT strided/transpose, MD-grid, NW, merge sort, radix sort |
| Declared | 19 | every inventory entry remains in the denominator |

The censored cases expose real frontend gaps (`xor`, calls, `uitofp`, `umin`,
`ashr`, lifetime intrinsics, and one control-flow assertion). Therefore a future
paper must say “MachSuite compatible-subset accuracy, 11/19 preflight-ready,”
not “full MachSuite coverage.” No MachSuite mapper labels were revealed during
this implementation and there is no final accuracy number yet. The label-free
preflight outcomes and structural covariates were visible while the protocol
was hardened, so this is accurately described as label-blind rather than fully
covariate-blind.

## Historical evidence

The earlier 54-row/15-name exploratory model used a different lower-bound and
feature contract and lacked complete source/candidate identities. Its reported
holdout numbers are historical diagnostics only. It must not be used in the
frozen MachSuite command or presented as the current model. Any local files
under `models/` or `evaluations/` with that contract are intentionally excluded
from the commits for this protocol.

LISA motivates generated graph training followed by evaluation on real
programs, and motivates a later graph/mapping-aware Model 2. It does not justify
using a GNN here: LISA predicts node/edge guidance labels and retrains per
accelerator, whereas this Model 1 predicts one mapper's final II residual.

## Remaining work before a paper result

The corrective generated-data protocol is now implemented and predeclared as
`motif-v4`, but no v4 mapper collection or model fit has been run. It keeps the
v3 generator/frozen artifact immutable, crosses six path contexts with five
pressure profiles and three operation bands, preserves transpose rectangles
on the same base lineage, and provides a manifest-only first phase. Its gates
require strict generator-family transfer improvement, positive-residual
recovery, shape-balanced improvement, ranking non-degradation, and versioned
coverage. Passing those gates remains future empirical work; code availability
is not evidence that the current Ridge limitation has been solved.

The following three steps apply only to the historical motif-v3 frozen
artifact. They do not authorize revealing labels for an unfitted v4 model.

1. Run MachSuite `predict` with the historical generated-only v3 artifact and
   publish or externally timestamp the resulting seal.
2. Only after publication, run `reveal` and report accuracy over scored
   candidates and coverage over all 19 declarations,
   with every mapper timeout/failure retained as censored.
3. Treat any frontend fix, model change, feature change, or generator change
   after reveal as a new protocol version, not a patch to the frozen result.

## Verification snapshot

Historical motif-v3 snapshot at this update:

```text
Neura source worktree: clean at 47b7e3a6 (main plus shared Rec/Res analysis pass)
MachSuite submodule: clean at 6236e593
MachSuite label-free preflight: 19 declared, 11 ready, 8 censored
formal generated corpus: 2250 bases, 4500 declared candidates, 4482 labels, 18 mapper-censored
complete paired training set: 2232 lineages, 4464 rows; all family/shape coverage cells passed
selected model: residual Ridge, lambda=3, dead-zone=0.75, design rank 14/14, condition 48.01
nested unseen-lineage macro MAE: LB 0.17249, Ridge 0.13030 (24.46% lower)
leave-one-generator-family-out macro MAE: LB 0.17393, Ridge 0.17393 (tie, not improvement)
MachSuite mapper labels revealed: 0
predictor unit tests: 130 passed
```

Motif-v4 protocol state is separate: the 1,500 bases and 3,900 candidates have
been materialized locally as a label-free manifest with SHA-256
`834278ae9ae40dc5aa2e0f09c834d19eeda17f63e10aab9f09d485231defb868`;
`protocols/motif-v4-predeclaration.json` records the associated implementation
and toolchain hashes. No v4 mapper label has been collected and no v4 model
has been fitted. The historical counts above are not v4 results, and the local
attestation is not an external trusted timestamp.
