# Model 1 status

Last updated: 2026-09-02

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
mapper gap. The 19 primary features are all available before mapping:

```text
semantic_edges
semantic_depth
semantic_width
semantic_max_fanout
semantic_branch_nodes
semantic_cutwidth
live_value_peak
pointer_path
memory_path
control_path
multi_input_nodes
memory_ops
gep_ops
indirect_geps
pointer_loads
tiles
links
rows
split_domain
```

This separation avoids label leakage, double counting, and the interpretability
problem of claiming an analytical term both as a theorem and as a learned
coefficient. Cost-model diagnostics may be evaluated as generated-training
ablations, but a MachSuite result may not be used to select them.

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
generator contains six families:

```text
chain, fanout, reduction, diamond, mixed, random_dag
```

It varies real operation count, dependency edges, depth/width, fanout,
reconvergence, and architecture candidates. All shapes/FU variants of one base
DFG share one leakage lineage and canonical DFG identity. A manifest is written
before mapper invocation; failures and timeouts are censored rather than
fabricated as numeric II labels.

The current generators are compute-focused. Memory, recurrence, pointer, and
control generator families are still needed before making a broad distribution
claim. The recommended paper-scale collection is roughly 200--300 base DFGs
per independent generator family and 4--6 architecture candidates per base.
`freeze-model` enforces at least 1,000 distinct base DFGs across all six current
families by default. A small override produces a smoke-only artifact that the
frozen `predict` phase rejects. No large training run or final frozen model is
claimed yet.

For the frozen protocol it also requires one direct, clean-checkout run with
at least 200 requested bases per family, shapes `3x3/3x4/4x4`, both
homogeneous and split-domain architectures, the predeclared Ridge/dead-zone
grid, and a whole-generator-family holdout. Raising the scale threshold is
allowed; lowering it can only produce a smoke artifact.

Freezing verifies the pinned clean Neura checkout, mapper-binary hash,
deterministically regenerated source, mapped-artifact hash and embedded
`compiled_ii`/RecMII/ResMII, plus the label-free Rec/Res artifact hash and its
`rec_res_mii_info` marker. It then requires analyzer and mapper Rec/Res facts
to match exactly before reproducing the selected Ridge model. It does not rerun
every expensive training mapping a second time; the archived analysis/mapped
artifacts plus their provenance are the training-label evidence.

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
this implementation and there is no final accuracy number yet.

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

1. Expand generated memory, recurrence, pointer, and predicated-control
   families, then collect the predeclared large generated corpus.
2. Run nested generated-lineage and whole-generator-family selection once and
   freeze the resulting model artifact.
3. Run MachSuite `predict`, publish the seal, then—and only then—run `reveal`.
4. Report accuracy over scored candidates and coverage over all 19 declarations,
   with every mapper timeout/failure retained as censored.
5. Treat any frontend fix, model change, feature change, or generator change
   after reveal as a new protocol version, not a patch to the frozen result.

## Verification snapshot

At this update:

```text
Neura source worktree: clean at 47b7e3a6 (main plus shared Rec/Res analysis pass)
MachSuite submodule: clean at 6236e593
MachSuite label-free preflight: 19 declared, 11 ready, 8 censored
generated end-to-end smoke: 18 declared, 16 labels, 2 mapper-censored
small-corpus scale gate: correctly rejected for a final frozen model
MachSuite mapper labels revealed: 0
predictor unit tests: 87 passed
```
