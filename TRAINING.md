# Residual Ridge training

This is the exact training contract for Model 1. It distinguishes the
statistical observation, leakage split unit, and DSE query; none may be inferred
from row order or filenames.

## Target and identities

For sample `i`, let `b_i = max(RecMII_i, ResMII_i)` and let `II_i` be the
completed heuristic-mapper label. Training predicts only the non-negative gap:

```text
y_i = II_i - b_i
```

All input features are computed before mapping. The primary feature contract
also excludes `b_i`, `RecMII_i`, and `ResMII_i`; the floor is applied once and
is not relearned by Ridge. Each row must carry both non-negative integer
components, and the loader verifies positive `b_i = max(RecMII_i, ResMII_i)`
before fitting. Model
artifacts naming any of these contract fields as features are rejected.
Mapper output, search time, kernel name, and
benchmark name are not features. Four IDs have different jobs:

1. `leakage_lineage_id`: the outer/inner split unit. All compiler variants of
   one algorithm and all candidates of one generated base DFG stay together.
2. `base_dfg_id` / `ranking_query_id`: one exact DFG whose architecture or
   mapper candidates may be ranked against each other. When both fields are
   present they must be equal.
3. `candidate_id`: one architecture and mapper configuration inside that
   query. It is unique only within its query.
4. `sample_id`: one recorded row/attempt and globally unique in a dataset.

A ranking query spanning two leakage lineages is invalid because its held-out
predictions came from different outer-fold models. Rows sharing the same
observation identity under the active model projection are collapsed;
contradictory rows for the same scoped candidate are rejected.

For the frozen generated corpus, every base DFG is observed under exactly two
predeclared target cells on the same pinned Neura 4x4 YAML: the full 4x4 domain
and one secondary rectangle assigned in round-robin order over the remaining
eight 2x2-through-4x4 shapes. A lineage enters weighting, CV, and fit only if
both cells have successful labels. Partial successes remain in the audit
report and manifest. The paired design estimates within-DFG shape effects at
one quarter of a nine-shape Cartesian scan, while completion fractions disclose
the remaining selection toward mapper-complete graphs.

## Weighting

Rows are not independent merely because a base DFG has many shapes. Let `G` be
the number of training lineages, `K_g` the number of distinct observations in
lineage `g`, and `D_gk` the number of exact copies of observation `k`. Without
corpus strata, each row has weight:

```text
w_i = 1 / (K_g * D_gk)
```

Thus each lineage contributes total mass one and all weights sum to `G`.

For mixed generated/real training, let there be `S` strata and `N_s` lineages
in stratum `s`. The implemented weight is:

```text
w_i = (G / S) / (N_s * K_g * D_gk)
```

Every stratum contributes mass `G/S`; thousands of generated lineages do not
numerically swamp a small real stratum, while total mass remains `G` so the
meaning of a fixed Ridge penalty does not change. With one declared stratum,
the coefficients are exactly the same as the unstratified calculation.

## Standardization and Ridge fit

For each feature `j`, compute its weighted training mean and standard deviation
and standardize it:

```text
z_ij = (x_ij - mean_j) / scale_j
```

The model solves:

```text
minimize_beta
    sum_i w_i * (y_i - beta_0 - sum_j beta_j * z_ij)^2
    + lambda * sum_j beta_j^2
```

`beta_0` is not penalized. `lambda` shrinks unstable correlated coefficients
toward zero; it does not clamp predictions and does not turn an estimated
feature into a lower bound. In matrix form:

```text
beta = (Z^T W Z + lambda * diag(0, 1, ..., 1))^-1 Z^T W y
```

At inference, the raw residual is projected onto the valid domain and an
optional validation-selected dead zone is applied:

```text
r_raw = beta_0 + sum_j beta_j * z_j
r = 0                 if max(0, r_raw) < dead_zone
    max(0, r_raw)     otherwise
predicted_II = max(RecMII, ResMII) + r
```

For example, if `max(RecMII, ResMII)` is 5 and Ridge predicts residual 2.4,
the point estimate is 7.4. If it predicts -0.6, the point estimate is 5. The
model returns a continuous value; integer rounding is a separate policy.

## Nested model selection

The outer split estimates transfer to an unseen leakage lineage. Inside each
outer training split, independent grouped validation selects `lambda` and the
dead zone. Outer test labels are never used for that choice.

- Each outer or inner validation invocation independently uses
  leave-one-lineage-out when its current group set has at most 20 groups.
  Otherwise it uses at most 10 deterministic grouped folds. Groups are
assigned by stable SHA-256 order and balanced inside declared strata. No
row-level random split is used. Thus an inner split may use LOGO
  even when its parent outer split used grouped K-fold.
- Hyperparameters minimize stratified macro-lineage MAE first, then ordinary
  macro-lineage MAE, row MAE, and deterministic numeric tie breakers.

After exploratory evaluation, the full development corpus selects one pair of
hyperparameters with grouped validation and fits one serialized model. This
does not make evaluation blind. A paper claim still requires the frozen
MachSuite manifest whose labels did not influence features, model class,
hyperparameters, or generators.

The final-freeze utility gate is also generated-only: the independently
recomputed outer nested-lineage Ridge macro MAE must be strictly less than the
corresponding prediction `max(RecMII, ResMII)`. Equality or degradation leaves
the artifact smoke-only. This gate uses neither random-row diagnostics nor
MachSuite labels and does not by itself establish cross-suite generalization.
The fitted 19-feature matrix plus intercept must be full rank; a redundant
feature therefore fails the gate instead of being hidden by Ridge regularization.

## Required reporting

Report generated training/validation by base and generator family; distinct
base-DFG/lineage/candidate counts; mapper successes and censored attempts;
requested and complete paired-shape lineages plus complete-case exclusions;
grouped split protocol; selected hyperparameters; negative raw-residual rate;
and frozen MachSuite results over both scored and all 19 declared variants.
Target-shape ranking is reported
only inside explicit base-DFG queries with at least two candidates and
non-constant target II.

Collection execution does not change this statistical contract.  The motif
collector predeclares every candidate before any compiler subprocess; bounded
candidate parallelism is controlled by `--motif-jobs` (default `1`), while
Rec/Res analysis and mapping are serial within a candidate.  A SIGINT leaves
only atomically checkpointed terminal results and declared work for resume,
returns status 130, and performs no fitting or training.
