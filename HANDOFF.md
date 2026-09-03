# Fresh-agent handoff: CGRA compiled-II predictor

Last updated: 2026-09-03

## Objective and non-negotiable contract

This repository builds an offline predictor for the `compiled_ii` produced by
the pinned Neura heuristic mapper. It does not change the mapper and must not
claim that a point prediction proves mapping feasibility.

The mathematical contract is:

```text
LB = max(RecMII, ResMII)
target_residual = compiled_ii - LB
predicted_compiled_ii = LB + max(0, predicted_residual)
```

RecMII and ResMII come from Neura's shared C++ functions through the existing
analysis-only pass. Do not reimplement their formulas in this repository. Do
not add LB, RecMII, or ResMII to the learned feature vector. Do not resurrect
PlacementHallLB, MandatoryRouteCutLB, MemMII, RegMII, or RouteMII for Model 1.

Training uses generated DFGs. MachSuite is the held-out real-program test. No
MachSuite mapper labels have been revealed. Never use MachSuite labels for
feature, generator, hyperparameter, or model selection.

## Git and repository state

At handoff time:

```text
repository: /home/x/shiran/project/cgra-ii-predictor
branch: main
status: clean
relation to origin/main: ahead by 3 commits
HEAD: b46a807a45e9dcde5a2dbd12e0841aba5eaec992
author identity: guosran <sguoau@connect.ust.hk>
```

The three unpushed commits are:

```text
b46a807 feat: audit tiny Neura prediction shapes
5818601 feat: freeze normalized shape-aware II protocol
36242bb feat: train II predictor across Neura target shapes
```

Do not modify the Neura submodule for this work. The predictor already calls
the Rec/Res analysis pass available on Neura main. The pinned submodule is:

```text
third_party/neura = 47b7e3a68c321075293e6fcb45fb3b1cabb93b88
third_party/machsuite = 6236e593012cb86b0d2f08d9fb9ba0411ff989b4
```

## What is implemented

The frozen training population remains the nine rectangular prefix shapes from
2x2 through 4x4. Each generated base DFG has a 4x4 candidate plus one balanced
secondary candidate. This is the immutable `motif-v3` protocol.

Prediction now scans:

```text
1x1, 1x2, and all nine rectangles from 2x2 through 4x4
```

The 1x2 target is the canonical orientation for the 1x2/2x1 two-tile symmetry
of the pinned architecture and current feature set. The tiny targets are
reported as `stress_only_untrained_shape`; they are deliberately excluded from
the automatic Pareto frontier and mapper-verification order. They must not be
described as trained or frozen support.

For 1x1, network-link and bisection-link counts are zero and both routing
pressure ratios are defined as zero. Prediction records:

```text
mapper_ii_ceiling = 20
lower_bound_within_mapper_search_interval = (LB <= 20)
```

When `LB > 20`, the pinned mapper has no II value to try. The candidate stays
in the audit record but is excluded from automatic ranking. Never fabricate a
failure as `II=21`.

Relevant implementation:

- `adapters/neura_motifs.py`: `DEFAULT_SHAPES` versus `PREDICTION_SHAPES`.
- `adapters/neura_experiment.py`: 1x1 feature handling, prediction records,
  support classification, and shape selection.
- `II_PREDICTION.md`: inference semantics.
- `MODEL_STATUS.md`: current evidence and model limitations.

## Existing formal corpus and artifact

Local ignored artifacts are under:

```text
corpora/motif-v3-formal-seed-20260902/report.json
corpora/motif-v3-formal-seed-20260902/frozen-random-model.json
```

Formal v3 counts:

```text
2250 requested base DFGs
4500 declared candidates
4482 successful mapper labels
18 censored attempts
2232 complete paired lineages
4464 fitted rows
```

The selected residual Ridge model has `lambda=3`, dead zone `0.75`, and design
rank 14/14. Nested unseen-base-lineage macro MAE is:

```text
Rec/Res floor: 0.172491
Ridge:         0.130299
relative reduction: 24.46%
```

Leave-one-generator-family-out macro MAE is a tie at `0.1739305`.

The v3 artifact was frozen before the tiny-shape code change. Preserve it as a
historical, pre-MachSuite-reveal artifact, but do not claim that it was freshly
frozen from current HEAD. Any new feature, generator, shape-training, or gate
claim requires a new protocol/corpus/artifact version; never mutate the v3
corpus in place.

## Why the current model is not good enough

The apparent 24.46% improvement is almost entirely a known-generator-family
interpolation result:

```text
3888/4464 rows (87.10%) have residual 0
576/4464 rows have positive residual
334/576 positive rows are random_dag
all 149 rows with residual > 1 are random_dag
99.14% of total absolute-error reduction comes from random_dag
```

In leave-one-generator-family-out validation, the selected dead zones reduce
all 4464 predictions to the Rec/Res floor and miss all 576 positive residuals.
The existing transfer gate permits a tie, so it proves only non-degradation;
it does not demonstrate transfer.

Per-shape behavior is also uneven. Ridge improves 3x3 through 4x4 but is worse
than the floor on 2x2, 2x3, 2x4, and 3x2. Tie-aware within-DFG ranking changes
from 0.974127 for LB to 0.973504 for Ridge, so the model does not improve the
main shape-ranking objective.

Root causes, in priority order:

1. Difficult positive residuals are concentrated in `random_dag`; the model
   mainly learns generator identity.
2. The zero-inflated target, MAE, and selectable dead zone reward predicting
   zero residual.
3. Thirteen scalar summaries discard local reconvergence, placement conflict,
   route competition, and orientation.
4. Half the rows are 4x4 while each secondary shape has much less weight; the
   objective is not shape-balanced.
5. Increasing operation count often tightens ResMII for regular graphs, while
   random-DAG residuals grow with operation count. One global additive Ridge
   relation cannot express both mechanisms.

Do not respond by immediately replacing Ridge with a GNN. A more expressive
model cannot learn positive mechanisms absent from the data.

## Tiny-shape evidence

A pilot used 54 bases: six per family spanning low/mid/high operation bands,
each evaluated on 1x1, 1x2, and 2x1. Of 162 attempts:

```text
91 succeeded
71 had no mapper label
every successful attempt had compiled_ii == LB
```

For the full 250-base-per-family v3 distribution, the number with `LB <= 20`
is already too small for the current 80% coverage gate in several cells. The
most severe cases are:

```text
pointer_chase 1x1: 0/250
pointer_chase two-tile strip: 122/250
memory_stream 1x1: 9/250
memory_stream two-tile strip: 72/250
```

Therefore 1x1/1x2 are useful prediction/feasibility stress records, not useful
new Ridge training evidence under the current corpus.

## Recommended next iteration

Keep Ridge as the transparent Model 1 baseline. Before training another model,
predeclare a new `motif-v4` corpus that creates multiple independent positive-
residual mechanisms within every family, rather than merely adding more
`random_dag` instances. Vary at least:

- layered and sparse connected DAG structure;
- reconvergence and fork/join depth;
- long-range fanout and cutwidth;
- live-range/register pressure;
- mixed memory, pointer, and predicated-control paths;
- comparable Rec/Res floors across families;
- transpose pairs for the same base DFG: 2x3/3x2, 2x4/4x2, 3x4/4x3.

Use a new seed and write the manifest before collecting any mapper label.
Candidate generation may use mapping-free Rec/Res screening, but must not tune
structure after inspecting compiled-II labels.

The v4 acceptance gates should be predeclared and include:

1. Strict improvement in leave-one-generator-family-out evaluation, not a tie.
2. Rejection when every held-out prediction equals LB.
3. Positive-residual recall and positive-subset MAE.
4. Per-family, per-shape, and per-operation-band metrics.
5. Shape-balanced point error and non-degradation of tie-aware ranking.
6. Explicit censored/feasibility coverage, especially for tiny shapes.

Only after v4 distributes positive residuals across families should the next
agent compare Ridge with a predeclared hurdle model: L2 logistic prediction of
`residual > 0`, followed by Ridge on positive residuals. A graph model is a
later Model 2 candidate, not the first corrective action.

If formal tiny-shape support is required, it needs a separate feasibility or
censor-aware target. Numeric-II regression must remain conditional on a
successful mapper label.

## Verification and useful commands

The current complete test suite passed:

```sh
python3 -m pytest -q
# 130 passed
```

An end-to-end prediction smoke test at current HEAD produced 11 candidates and
correctly classified 1x1/1x2 as untrained stress shapes. The tested chain
instance produced `LB=40` on 1x1, which was correctly marked above the mapper
ceiling, and `LB=20` on 1x2.

Before doing new work, read these files in order:

1. `MODEL_STATUS.md`
2. `II_PREDICTION.md`
3. `CORPUS_PROTOCOL.md`
4. `TRAINING.md`
5. `EVALUATION_PROTOCOL.md`

Then verify `git status`, the pinned submodule revisions, and that MachSuite
mapper labels are still unrevealed. Do not push or rewrite the three existing
commits without explicit user direction.
