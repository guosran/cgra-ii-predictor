# Amoeba frozen-candidate replay interface audit

This audit is read-only with respect to both Amoeba and Neura.  It records the
remaining upstream interface requirement; the predictor does not work around
it by silently changing a selected shape.

## Reproduction

The clean analytical-DSE build at commit
`2b7d75bd52afe845089238acb9b3f7e2499021f6` materialized
`candidate-5` from the 64-candidate `parallel-nested` manifest.  The resulting
Taskflow attributes were correct:

- `Task_0`: `cgra_count=1`, `cgra_shape="1x1"`
- `Task_1`: `cgra_count=4`, `cgra_shape="1x4"`

The frozen candidate was then passed to the current Amoeba compiled
resource-aware/heuristic-mapper pipeline.  Its log first mapped both original
tasks with `Overriding architecture dimensions to 4x4 tiles`, then fused the
tasks and emitted a different final allocation, `cgra_count=1`,
`cgra_shape="1x1"`.  It did not replay `Task_1` on the selected `4x16` mapper
array.  A later optimizer-created `cgra_count=2` probe did correctly invoke a
`4x8` array, so the lower-level physical-to-mapper conversion exists; what is
missing is a mode that preserves and maps the frozen materialized candidate
without re-running fusion/allocation search.

The read-only clean binary was
`/tmp/amoeba-static-dse/build/tools/mlir-amoeba-opt/mlir-amoeba-opt`, SHA-256
`b6dffbef35b5a5780b59c8a450b68a6d04c02a41f160008608947fbc64d78c81`.
Frozen `parallel-nested` candidate-manifest SHA-256 is
`29103f5dd1003a4d60140598a84738fa7cb8e2bacbfb90af56d647a155c84d4a`;
its 16-query analytical feature input is
`4e237ea3ec367538fc062f818751b35af22ada1c55e456c738ee8748bee7ba6b`,
and the independent real-mapper query oracle is
`aa96e2f51d1f79c1d68c1bc1a88802f42ac63738491eb0942331922c11160a31`.

Reproduction artifacts:

- `/tmp/amoeba-v8-candidate-5-materialized.mlir`
- `/tmp/amoeba-v8-candidate-5-replayed.log`
- `/tmp/amoeba-v8-candidate-5-replayed.mlir`

Required Amoeba-side interface: a frozen-candidate replay mode which treats
materialized `cgra_count`/`cgra_shape` as immutable and forwards each task's
exact `mapper_tile_rows/cols` as Neura `y_tiles/x_tiles`.  Until that exists,
the predictor can score every program candidate and independently validate
each task/shape against the real heuristic mapper, but cannot claim that a
shortlisted complete program was faithfully replayed by Amoeba.

## Independent frozen-candidate pipeline

`adapters/amoeba_frozen_pipeline.py` provides the independent path without
modifying Amoeba or Neura. It verifies the manifest, catalogue, task DFG,
architecture, and Neura executable hashes; scores every frozen candidate with
the same `startup + II * (trip_count - 1)` compute-bottleneck objective; then
maps each unique task/shape needed by the top-k exactly once. Mapper timeouts,
non-zero exits, missing output, and invalid labels remain censored and never
receive a numeric II. The emitted `scores.jsonl` is compatible with the Amoeba
v2 score schema, while `report.json` records commands, logs, artifact hashes,
shape checks, real shortlisted objectives, cache counts, and optional oracle
regret.

Before mapping, the driver runs Neura's `--insert-data-mov` pass. This is the
mapper's required IR normalization for extracted DFGs and is idempotent for
DFGs whose operands are already wrapped. It does not change the frozen shape
or the ML input catalogue.

The verified `parallel-nested` top-3 invocation is:

```sh
python3 adapters/amoeba_frozen_pipeline.py \
  --manifest /tmp/amoeba-static-dse-candidates-v2.jsonl \
  --cost-catalog evaluations/pointwise-v8-1478205/parallel-nested-costs.json \
  --task-dfg Task_0=/tmp/amoeba-task-0-neura.mlir \
  --task-dfg Task_1=/tmp/amoeba-task-1-neura.mlir \
  --neura-opt /home/x/shiran/project/neura/build/tools/mlir-neura-opt/mlir-neura-opt \
  --architecture corpora/motif-v8-pilot-seed-20260911/architecture/neura-main-4x4.yaml \
  --top-k 3 --mapper-timeout-seconds 300 \
  --oracle /tmp/amoeba-parallel-query-oracle-v1.json \
  --expected-neura-opt-sha256 7c8b0753609c9045fd4dabd3041f5a5311ce3922f431526c9a61ea4d794e6d49 \
  --expected-architecture-sha256 f244f15be30604eb32eb96e4837a4bf1ce5c34961c3a46299b90931505cc97e6 \
  --neura-source-revision 47b7e3a68c321075293e6fcb45fb3b1cabb93b88 \
  --output-dir evaluations/pointwise-v8-1478205/independent-parallel-top3-with-mov-20260905
```

This run scored all 64 candidates and selected `candidate-0`, `candidate-8`,
and `candidate-16`. Their six task references collapsed to four real mapper
calls; all four succeeded, preserved the requested shape, and matched the
independent oracle. `candidate-0` was selected at 130 cycles with zero oracle
regret. This is a 75% reduction from 16 exhaustive unique task/shape mappings;
the corresponding candidate-level top-3 reduction is 95.3125% from 64
candidates. The measured mapper replay time was 0.172 seconds on this host.
This result is exact for the current static shape-only compute-bottleneck
objective; it does not claim whole-program communication or temporal-placement
costs that are outside that objective.

For `parallel-nested`, all three shortlisted candidates were predicted at
106.432 cycles and measured at 130 cycles: an underprediction of 23.568 cycles,
or 18.129%. The independent scorer's complete 64-candidate objective vector
and shortlist are identical to Amoeba's ML-catalogue scorer. Amoeba's existing
materialized replay also reaches a 130-cycle compute bottleneck, but maps every
kernel as 4x4 even when the materialized candidate declares a larger shape;
the independent path is the one that verifies the requested mapper rectangle.

For `multi-nested`, all 32,768 candidate scores and the top-3
`candidate-8448/8449/8450` also match Amoeba's ML-catalogue scorer exactly.
Their 15 task references collapse to seven unique mapper calls. All seven
succeeded, but every candidate was predicted at 365.471 cycles and measured at
578 cycles, an underprediction of 212.529 cycles or 36.770%. The dominant error
is Task_2 on 12x4: predicted II 1.883 versus compiled II 3.

This case is also a negative DSE result. A stable RecMII/ResMII-lower-bound
ranking selects `candidate-0`; direct mapping measures 387 cycles, so the ML
selection is 191 cycles (49.354%) worse than that analytical selection. The ML
prediction for `candidate-0` itself is accurate (383.717 versus 387, -0.848%),
but its erroneous cross-shape ranking for Task_2 makes the DSE choose 12x4 over
4x4.

All 40 task/shape queries have since been mapped and shape-verified in
`evaluations/pointwise-v8-1478205/multi-nested-full-oracle-20260905/`. The
complete oracle confirms that 387 cycles is the global optimum for the current
compute-bottleneck objective, rather than only an analytical comparison point.

The program also has a hard resource constraint: every oriented task rectangle
must fit concurrently and without overlap on the architecture's 4x4 physical
CGRA grid. The independent scorer now uses exact bit-mask rectangle packing
before ranking. Of the 32,768 Cartesian-product candidates, 18,288 are legal
and 14,480 (44.2%) are rejected as `HARDWARE_GRID_UNPACKABLE`. There are 2,379
oracle-optimal legal candidates at 387 cycles. The old ML top-3
`candidate-8448/8449/8450` are themselves packable, however, so resource
filtering alone does not repair the Task_2 cross-shape error: their actual
objective remains 578 cycles. The audited constrained run is
`evaluations/pointwise-v8-1478205/independent-multi-grid-packable-top3-20260905/`.
The same packing predicate belongs in Amoeba's candidate generator so illegal
Cartesian-product members are never materialized; the independent scorer's
check remains a defensive validation boundary.

The remaining `test/multi-cgra/taskflow` cases need different treatment:

- `irregular-loop` now runs through a narrow extractor workaround for Neura's
  non-round-trippable zero-rank `store_indexed` custom syntax. Its full
  464-candidate exact-grid audit selects all-1x1 at 69 cycles, tied for the
  global optimum.
- `attention` has eight static tasks. The op-count heuristic plus incremental
  exact packing reduces `8^8 = 16,777,216` candidates to 3,643; this is a
  constrained search result, not a full-space oracle.
- `resnet` lowers to thirteen top-level affine tasks (ten after the earlier
  memory-access streaming fusion). Its current lit path retains resource-aware
  profiling and latency reporting, but disables resource-aware task fusion and
  imports a fixed all-1x1 allocation so balance/fission cannot change resources.
  If fusion/fission search is re-enabled, its mapper-backed search needs
  shortlist/beam-style pruning rather than repeated unrestricted mapper calls.
- `symbol-dynamic/*` has runtime trip counts and is outside the current static
  manifest contract.
- `pipeline-interval/*`, `allocation-with-resource-binding`, and `replica-set`
  are orchestration unit tests without standalone Neura kernels, so they are
  not II-prediction benchmarks.

Accordingly, this report does not claim that the complete
`test/multi-cgra/taskflow` directory passes. `parallel-nested` and
`multi-nested` have complete pointwise prediction, exact physical-grid
filtering, direct mapper replay, and oracle-regret results. `irregular-loop`
now has an equivalent global-optimum result despite one non-critical censored
query. `attention` has only the constrained result above.

The complete built-in taskflow lit directory was subsequently run. After the
user-directed ResNet no-fusion/no-fission configuration, its latest result is 9 pass,
6 fail, 0 timeout, and 1 unsupported. ResNet itself passes in 0.31 seconds.
Five failures are only the missing `torch_mlir` environment dependency. In the
user's current dirty Amoeba checkout, attention passes while irregular-loop
reaches the end of resource-aware optimization and then fails FileCheck because
the pass now chooses 1x1 for the fused task while the test expects 1x2. This
allocation expectation drift is separate from the extractor/parser workaround,
and the predictor work does not overwrite those in-progress Amoeba changes. Per
the user's current scope, the five missing-`torch_mlir` failures and this one
FileCheck drift are ignored; the remaining nine taskflow tests pass.

## Incremental candidate generation

`adapters/enumerate_amoeba_pruned_candidates.py` consumes a valid static v2
base manifest and one extracted Neura DFG per task. It counts mapper-relevant
Neura operations, then limits each task to
`ceil(operation_count / physical_CGRA_tile_count) + slack` physical CGRAs.
This is deliberately metadata-labelled as a heuristic search cap. At every
DFS prefix it separately runs exact oriented-rectangle packing against the
physical grid; packing is the hard architecture constraint. The output is a
normal v2 candidate manifest, so analytical feature generation, pointwise
catalogue inference, and frozen replay require no special scorer path.

The extractor accepts kernels with `inputs`, `iter_args_init`, or both. It
also normalizes only the empty-index spelling
`neura.store_indexed ... to [ : ]` to generic MLIR with explicit operand
segments. This preserves the zero-rank store and makes the standalone DFG
round-trip through Neura. It is not a general rewrite for indexed stores.

The attention lowering used for the audit came from an isolated build. Its
Neura control-flow pass moves the kernel yield back to the end of the flattened
entry block and its arithmetic conversion handles `arith.cmpf`. Those two
source fixes still need upstream tests before attention can be called a clean
checkout end-to-end pass. Full provenance is frozen in
`evaluations/taskflow-static-followup-2026-09-05.json`.

## Pairwise-ranking follow-up

Three otherwise identical residual pointwise models were trained with
within-DFG strict-II margin loss weights 0.1, 0.3, and 1.0.  Selection used
validation continuous-II MAE only.  Weight 0.1 was selected: its
validation/test MAE is `0.34900/0.39666`, compared with
`0.40025/0.43631` for the same-architecture zero-weight baseline.  The
selected checkpoint SHA-256 is
`ae9273e2bec0c864b4dce26369cefc8a20ecac4314f76bc0e2277b8b6238cd75`.

On `multi-nested`, this checkpoint reverses the former Task_2 mistake: 4x4
and 12x4 are predicted at `1.418` and `1.605`, respectively, while the mapper
labels are 2 and 3.  The exact-grid shortlist becomes
`candidate-0/1/2`; direct replay selects `candidate-0` at the global-optimal
387 cycles with zero regret.  The single model still underpredicts that
program at 275.836 cycles.

The deployment candidate adds this checkpoint to the preceding five-model
pool.  Validation assigns weights 0.91774 to the ranking model and 0.08226 to
the prior residual model, with every other model and the analytical lower
bound at zero.  Its validation/test MAE is `0.34095/0.38966`.  It preserves
the same zero-regret `multi-nested` choice while predicting 359.952 cycles,
only 6.99% below the mapped result.  The frozen reports are:

```text
evaluations/pointwise-v8-1479107/ensemble.json
evaluations/pointwise-v8-1479107/independent-multi-grid-packable-top3-20260905/report.json
evaluations/pointwise-v8-ranking-loss-2026-09-05.json
```

The small-DFG evidence does not support switching to the analytical lower
bound.  Across the 40 frozen `multi-nested` task/shape queries, the selected
single model has MAE 1.063 versus 1.675 for the lower bound and is closer on
38 of 40 queries.  The lower-bound scorer reaches `candidate-0` only through
large ties and deterministic enumeration order; it predicts 196 cycles for a
387-cycle program.  The lower bound remains a hard numeric floor and audit
baseline.  Improving absolute calibration requires diverse 3--7-operation
training-only DFGs, without fitting on these frozen program queries.

## Additional static-program audit

`multi-nested` lowers, extracts five standalone Neura task DFGs, enumerates all
`8^5 = 32,768` static rectangular programs, and scores all of them through 40
unique task/shape costs.  Its candidate manifest SHA-256 is
`04f50ec7198191209fb2eb6d8c4472d88ed5fe5d821757a8d733db73e2d8e9d4` and
analytical feature SHA-256 is
`74b8cf445f241f15e6bbe8e3f2fa8fc370c298ad731025e7848ea472be0f8171`.
The initial `irregular-loop` audit stopped here because both inspected Amoeba
binaries printed a scalar store with an empty index list,
`neura.store_indexed ... to [ : ]`, which Neura could not parse back. The
narrow generic-syntax workaround described above has since removed that
round-trip blocker; it was never an unsupported ML shape.
