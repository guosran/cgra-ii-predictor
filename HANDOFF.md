# Fresh-agent handoff: multi-CGRA pointwise II predictor

Last updated: 2026-09-05 17:35 (Asia/Hong_Kong)

## 0. 先读结论

用户的最终目标不是让模型直接选择 shape，而是：前端对大程序的 task 时空排列、
shape 等生成 DSE program candidate；本仓库对 candidate 内每个
`(task DFG, mapper shape)` 预测 II；前端再用
`startup + II * (trip_count - 1)` 聚合并给 program 排序。

原主要问题已经修复：旧模型把 `multi-nested` Task_2 的 12x4/4x4 预测为
II `1.883/1.946`，与真实 `3/2` 反序，导致合法 program top-1 实际 578 cycles，
而全局最优是 387。validation-only 选择的 pairwise ranking weight `0.1` 把两者预测
改为 `1.605/1.418`，program top-1 变为 oracle-optimal `candidate-0`，regret 为 0。

最终部署候选是 Job `1479107` 的六模型 uncertainty-gated ensemble；validation
权重实际只保留 ranking 模型 `0.91774` 和原 residual 模型 `0.08226`。它的
validation/test II MAE 为 `0.34095/0.38966`，`multi-nested` 预测/真实 top-1 为
`359.952/387` cycles，误差 `-6.99%`、regret 0。不要换成 analytical：冻结 40 个
小-DFG query 上它的 MAE 是 `1.675`，ranking 模型是 `1.063`；analytical 选中最优只
是大量 tie 后的枚举顺序。

硬件侧仍使用整机 4x4 物理 CGRA 网格精确矩形装箱。不要重新收集正式 v8 数据；
下一模型数据优先项是独立生成同协议 3--7 op 的 training-only DFG，而不是在冻结
`multi-nested` 上拟合或继续堆模型参数。

## 1. 用户明确的语义

- 模型直接目标是每一个候选 task-shape 的 `compiled_ii`，不是“这个 DFG 应选哪个
  shape”。
- program 的 task 排列、shape 组合、资源分配和最终候选排序属于前端。
- 保留连续小数 II 是允许且期望的，因为前端需要连续分数降低排序量化误差。
- mapper failure/timeout 是 censored 类别，绝不能伪造 `II=21`。
- 所有 task 的有向矩形必须能够同时、无重叠地装入当前整机的 4x4 物理 CGRA
  网格。这是硬件合法性约束，不由神经网络学习。
- Neura heuristic mapper 是真值来源；不替换 mapper。
- 用户要求少 commit；本轮 v8/ranking/replay/taskflow 工作已凝聚为一个
  `feat: add constrained multi-CGRA v8 replay` commit。
- 用户明确不喜欢高频轮询，只有用户询问或预计作业已经结束时才查一次。

## 2. 仓库与工作树

```text
repository: /home/x/shiran/project/cgra-ii-predictor
branch: main
HEAD subject: feat: add constrained multi-CGRA v8 replay
relation: main is ahead of origin/main by 10 commits
```

源码、精简 JSON 证据和文档已提交；工作树仍有意保留未跟踪的详细训练报告、mapper
logs/scores 和用户原有文件。不要 `git clean`、`git reset`、checkout 覆盖或批量删除；先运行
`git status --short`。不要修改 `/home/x/shiran/project/neura` 或
`/home/x/shiran/project/amoeba`，除非用户明确扩大范围。若以后要移动 Neura PR
内容，先读 `/home/x/shiran/project/neura-cgra-cost-model/PR_SPLIT.md`。

未提交内容主要是：

```text
adapters/analyze_pointwise_hybrid.py
evaluations/model2-pointwise-hybrid-analysis-2026-09-05.json
evaluations/pointwise-v8-14780*/ and pointwise-v8-14781*/ detailed reports
evaluations/**/mapper/ and evaluations/**/scores.jsonl raw replay artifacts
```

`adapters/analyze_pointwise_hybrid.py` 和
`evaluations/model2-pointwise-hybrid-analysis-2026-09-05.json` 在当前工作前已是
untracked；保留它们。不要为了本轮工作改写已有 test；最新完整回归是
`python3 -m pytest -q`，234 tests passed；相关 py_compile、`bash -n` 和
`git diff --check` 通过。

## 3. 正式 v8 数据协议

数据在 FactCluster：

```text
/fact_data/yibozhang/cgra-ii-model2/multi-cgra-v8/motif-v8-formal-seed-20260911/corpus-manifest.json
/fact_data/yibozhang/cgra-ii-model2/multi-cgra-v8/placement-supervision.json
```

固定身份：

```text
manifest SHA-256:
33ef4a2dc8385b8245234cb4cc95817ab321c35291a2663f0f09103f0888823f
placement supervision SHA-256:
71f00fe2d29895704ea6edbc352c9f3adf2caad04fc39749e07519b4b16bebc3
Neura revision:
47b7e3a68c321075293e6fcb45fb3b1cabb93b88
mlir-neura-opt SHA-256:
7c8b0753609c9045fd4dabd3041f5a5311ce3922f431526c9a61ea4d794e6d49
architecture SHA-256:
f244f15be30604eb32eb96e4837a4bf1ce5c34961c3a46299b90931505cc97e6
```

人口：1,500 个不同 DFG，每个交叉 8 个 Amoeba rectangle shape，共 12,000
candidate；4,442 mapper success，7,558 censored。训练 DFG 的生成 operation band
是 8--20。`multi-nested` 的真实 task 只有 3--7 个 semantic op（route-expanded
为 4--8），所以也存在小 DFG 的分布外问题；当前用户要求先用已有数据验证 ranking
loss，不要因此立即重采集。

8 个物理/mapper shape：

```text
1x1 -> 4x4
1x2 -> 4x8
2x1 -> 8x4
1x3 -> 4x12
3x1 -> 12x4
1x4 -> 4x16
2x2 -> 8x8
4x1 -> 16x4
```

数据 split 单元是完整 `canonical_dfg_sha256/ranking_query_id`，比例 70/15/15；
同一 DFG 的不同 shape 不跨 split。censored candidate 只训练 success classifier，
不参加数值 II、placement 或 pairwise-II 监督。

## 4. 当前冻结模型与指标

最终部署候选是六模型 validation-only uncertainty-gated ensemble；实际非零权重只有
ranking `0.917740` 和原 residual `0.082260`，analytical 及其余模型均为 0。报告：

```text
evaluations/pointwise-v8-1479107/ensemble.json
SHA-256 354cd9bf4f10fef84248b2307ba35ed289363a75b993a988fcfe44e4299e1521
```

主要指标：

```text
validation ensemble continuous-II MAE: 0.340953
test ensemble continuous-II MAE:       0.389665
validation/test macro-query MAE:       0.464413 / 0.484033
analytical lower-bound test MAE:       1.913713
test floor exact / within-one:          61.63% / 92.76%
test round exact / within-one:          71.03% / 93.84%
test success Brier / ECE:               0.09330 / 0.02717
test success recall at 0.5:             83.67%
```

相比旧五模型，II error、success precision/Brier/ECE 和真实 program regret 都改善；
固定阈值 0.5 的 test success recall 从 `87.67%` 降到 `83.67%`，是明确 tradeoff。
analytical hybrid 在新 ensemble 上由 validation 选择 `null`，即不切换。完整权重、
per-family/per-shape 指标和审计见 `protocols/motif-v8.json`、
`evaluations/pointwise-v8-ranking-loss-2026-09-05.json` 与
`FACTCLUSTER_RUNBOOK.md`。

## 5. 当前训练到底是什么

模型仍是 pointwise predictor。每个 forward 输入一个 task DFG 和一个 mapper shape，
输出 mapper success probability 与 II。训练 batch 会把同一 query 的 8 个 shape
放在一起，但网络本体不读取整个 program。

已完成的三组对照都使用：

```text
interaction_mode=residual_pointwise
DFG representation=route_expanded_v2
message aggregation=dual_mean
hidden dimension=128
message-passing layers=9
batch size=16 queries
optimizer=AdamW
learning rate=1e-3
weight decay=1e-4
epochs/patience=80/80
placement loss weight=0.1
training seed=20260912
GPU=one NVIDIA L20 through Slurm
```

旧 pointwise loss 对 8 个 shape 只做独立误差，代码明确把 listwise loss 置零；所以
“同一 DFG 的 shape 预测顺序反了”没有直接惩罚。新总 loss 是：

```text
success BCE
+ residual-II SmoothL1
+ discrete residual-II cross entropy
+ 0.1 * placement loss
+ lambda * pairwise ranking loss
```

若成功 shape `i` 的真实 II 小于成功 shape `j`，新项是：

```text
ReLU(0.5 - (predicted_ii_j - predicted_ii_i))
```

tie 和 censored pair 不产生排序标签。配置字段为
`pairwise_ranking_loss_weight` 和 `pairwise_ranking_margin`；weight 默认 0，因此旧
checkpoint 与旧训练语义不变。本地 smoke test：正确顺序 ranking loss 为 0，反序
为 1.5；完整测试 228 passed，相关文件 py_compile、`bash -n` 和
`git diff --check` 通过。

## 6. FactCluster 终态与选择

三个训练均 `COMPLETED (0:0)`：

```text
1478681  lambda=0.1  validation/test MAE=0.34900/0.39666
1478683  lambda=0.3  validation/test MAE=0.39145/0.42418
1478682  lambda=1.0  validation/test MAE=0.42967/0.45276
baseline lambda=0    validation/test MAE=0.40025/0.43631
```

只按 validation 选择 λ=0.1。三份本地报告：

```text
evaluations/pointwise-v8-residual128-rank01-s12-1478681.json
SHA bf109e79377959a7e145fe570003b61c06c80d2072230bfee663e3dcd27bd012
checkpoint ae9273e2bec0c864b4dce26369cefc8a20ecac4314f76bc0e2277b8b6238cd75

evaluations/pointwise-v8-residual128-rank03-s12-1478683.json
SHA 4ffcbe03040dd922aa574f142907738918042488ef4bc6e8f67560de2a1a8b7b

evaluations/pointwise-v8-residual128-rank10-s12-1478682.json
SHA 4fcf2ce7fa3bbb6bcc886fe8b1beadf3a8a5c98fd93b16f0d158df1e54251cfa
```

单 checkpoint catalog Job `1479105`、最终六模型 evaluation Job `1479107` 也均
`COMPLETED (0:0)`。最终报告目录：

```text
evaluations/pointwise-v8-rank01-s12-1479105/
evaluations/pointwise-v8-1479107/
evaluations/pointwise-v8-ranking-loss-2026-09-05.json
```

连接和集群规范见 `FACTCLUSTER_RUNBOOK.md`。使用已配置的 `ssh factcluster`；正式
训练必须通过 Slurm，Python 必须是：

```text
/fact_home/yibozhang/.conda/envs/awq-vggt/bin/python
```

不要在登录节点直接跑训练，也不要在文档/回答中输出密码。

## 7. 4x4 整机硬件约束

这里的 4x4 是整机 4 行 x 4 列物理 CGRA，而不是单 task 的 op-count 上限。一个
program candidate 中所有 task 选择的有向 rectangle 必须能同时无重叠放入该网格。
面积和不超过 16 只是必要条件，不充分；实现使用 bit-mask backtracking 做精确装箱。

实现位于 `adapters/amoeba_frozen_pipeline.py`：

```text
_grid_packable
_candidate_grid_packable
reject reason = HARDWARE_GRID_UNPACKABLE
score model = static-shape-grid-packable-compute-bottleneck-v3
```

`multi-nested`：

```text
raw Cartesian candidates:       32768
hardware-grid legal:            18288
rejected:                       14480 (44.2%)
legal oracle optimum:           387 cycles
number of tied legal optima:    2379
old ML top-1 actual:            578 cycles
absolute/relative regret:       191 cycles / 49.354%
new ensemble top-1:             candidate-0
new predicted / actual:         359.952 / 387 cycles
new absolute/relative regret:   0 cycles / 0%
```

完整 40-query oracle：

```text
evaluations/pointwise-v8-1478205/multi-nested-full-oracle-20260905/oracle.json
SHA-256 45fb62568e93f17ea4a7cd2e45e79763e04359be25432dec6e37c161267ce245
```

受约束独立运行：

```text
evaluations/pointwise-v8-1478205/independent-multi-grid-packable-top3-20260905/report.json
SHA-256 d8e001d2afdf9b3fc9edec58432f2edc5734e7888b3a136b45bb118e3bb98347

evaluations/pointwise-v8-1479107/independent-multi-grid-packable-top3-20260905/report.json
SHA-256 32ec1bdca46229c7457ea870c2d6a4d2812856a88055bafb463f5179896b1c25
```

当前独立 scorer 会读取 frozen manifest 后把不合法候选标 invalid；要真正避免前端
生成 14,480 个非法 candidate，还应把同一装箱 predicate 放入 Amoeba candidate
enumerator。当前范围尚未修改 Amoeba。注意坏的旧 ML top-1 本身可以合法装箱，所以
硬约束减少搜索空间但不能替代模型修复；ranking loss 已独立修复该反序。

## 8. 独立端到端 pipeline 与 latency

`adapters/amoeba_frozen_pipeline.py` 验证 manifest/catalog/DFG/Neura binary/
architecture SHA，按与 Amoeba 相同的 task duration 和 max-bottleneck 公式评分，
top-k 后才调用真实 heuristic mapper；每个唯一 task-shape 只调用一次。映射前调用
`--insert-data-mov`，并验证 `mapping_info` x/y 和 placement bounds。

已验证：

```text
parallel-nested: 64 raw / 62 grid-legal; top-1 candidate-0; actual 130;
                 oracle regret 0
old multi:       32768 raw / 18288 grid-legal; top-1 candidate-8448;
                 predicted 365.471; actual 578; regret 191
new multi:       top-1 candidate-0; predicted 359.952; actual/oracle 387;
                 regret 0; top-3 replay 7/7 mapper success and oracle match
irregular-loop:  generic zero-rank-store workaround; 464 grid-legal candidates;
                 23/24 mapper queries succeeded; all-1x1 actual/oracle 69;
                 one unrelated Task_0/16x4 query censored at 60 seconds
attention:       op cap 16,777,216 -> 3,645; exact packing -> 3,643;
                 constrained top-1 all-1x1; predicted/actual 17,402,649 /
                 29,360,129 cycles; two Task_5 larger shapes censored
```

推理计时（L20）：

```text
six-model ensemble, one uncached task-shape, warm mean: 75.972 ms
P95:                                                    114.443 ms
cold model load once:                                   about 0.67 s
40 unique multi-nested queries, batched steady state:    815.11 ms
cached program scoring:                                  about 0.15 ms/program
32768 cached program candidates:                         about 4.89 s total
```

正确部署方式是先批量预测所有唯一 `(task, shape)` 并缓存，然后每个 program 只查表
和取 max；不要为每个 program 重复跑 GNN。

不要声称 `test/multi-cgra/taskflow` 全跑了：`parallel-nested`、`multi-nested` 已完整
验收；`irregular-loop` 已用语义保持 workaround 完成全 shape 对照并证明剪枝候选为
全局最优；`attention` 只完成约束空间 top-3，不是全空间 oracle。`resnet` 仍需约束驱动
enumeration；`symbol-dynamic/*` 超出静态协议；`pipeline-interval/*`、
`allocation-with-resource-binding`、`replica-set` 没有当前 pointwise pipeline 所需的
独立 Neura kernel。

## 9. irregular-loop / attention follow-up

`adapters/extract_amoeba_task_dfgs.py` 现在同时提取 `inputs` 和 `iter_args_init`。对于
Neura 打印成 `neura.store_indexed %v to [ : ]` 的零秩 store，只把这一种不可回读的
custom syntax 改写成等价 generic form，并显式写
`operandSegmentSizes = [1, 0, 0]`；不改变操作或依赖。`irregular-loop` 三个 task 因此
都可分析和映射。严格 op cap 把 512 个笛卡尔候选降为全 1x1 的一个候选；另一次放开
cap 的对照枚举出 464 个精确可装箱候选，24 个唯一 mapper query 有 23 个成功。全
1x1 实测 69 cycles，并列已观测全局最优；唯一 timeout 是 Task_0/16x4，而每个候选都
包含的 Task_2 八个 shape 已全部映射且最短就是 69 cycles，因此该 timeout 不影响
最优性证明。这里的原 blocker 是 extractor/Neura assembly round-trip，不是模型。

`adapters/enumerate_amoeba_pruned_candidates.py` 在 DFS 每一级做 4x4 有向矩形精确装箱，
并使用显式 heuristic：每个 task 最多
`ceil(materialized_ops / (per_cgra_rows * per_cgra_cols)) + slack` 个物理 CGRA。
`attention` 的 8 个 op 数为 `28/28/28/30/24/35/10/28`，cap 为
`2/2/2/2/2/3/1/2`；原始 `8^8=16,777,216` 先降为 3,645，再由精确装箱降为
3,643（总剪枝 99.9783%）。最终 ensemble 的 top-3 为 candidate-0/3/6；真实 replay
选择全 1x1 的 candidate-0，objective 为 29,360,129 cycles。Task_5 的 4x8、8x4
都在 300 秒 timeout，不能伪造 II。对每个 task 抽取首个 cap 外 shape 的额外审计中，
7/8 在 30 秒 timeout；唯一成功的 Task_6/4x8 compiled II=2，与 4x4 相同。该证据
支持把 cap 用作搜索预算，但不把它升级为硬件合法性定理。

`attention` 还暴露了前端问题：控制流 RPO flatten 可把 `neura.yield` 放到 loop-body
操作之前，违反 terminator 约束；隔离 worktree 中测试的最小源修复是在 flatten 后把
yield 移回 entry block 末尾，并补 `arith.cmpf -> neura.fcmp` lowering。补丁没有写入
正式 Amoeba/Neura 工作树。完整证据和哈希在
`evaluations/taskflow-static-followup-2026-09-05.json`。

Amoeba 自带 lit 只补跑了这两个目录，不是整个 taskflow suite。隔离 commit
`2b7d75b` 上两项均通过。用户当前 dirty Amoeba checkout 中 attention 通过，
irregular-loop 的各 pipeline 实际执行完成，但最后 FileCheck 失败：当前
resource-aware pass 给融合 Task_0_Task_1 选择 1x1，而测试仍期待 1x2。这是该 checkout
的 allocation 期望漂移，不是零秩 store workaround 或模型失败；没有覆盖用户的
Amoeba 修改来“修 test”。

后续：

1. 把 attention 的两处 Neura 修复连同 verifier regression test 送到 Neura，再用正式
   可复现 binary 替换临时 lowering 证据。
2. 对 `resnet` 复用同一增量 op-cap + exact-packing enumerator；原始 `8^13` 不能先
   物化，若剪后仍太大再加 validation-defined beam。
3. small-op 绝对校准仍是后续模型数据项：另建同协议 3--7 op training-only corpus，
   不要把冻结 40 query 放回训练，也不要直接切换 analytical lower bound。
4. 最终 ensemble 的 0.5 success threshold recall 有约 4pp 回退；若进入部署，应用
   validation-only threshold/calibration 检查，不能用 test 调阈值。
5. 后续 Neura 上游修复或 resnet 工作各自形成凝聚 commit；不要清理当前保留的详细
   evaluation/mapper artifacts。

## 10. 容易踩坑

- pointwise 模型以前虽然一次 batch 有 8 个 shape，但 pointwise 分支把 listwise loss
  设为 0；“之前已经同时输入 8 个 shape”不等于“之前训练了跨-shape 顺序”。
- 不要把 program 级 4x4 装箱当作模型特征或 op-count 规则。
- 不要把 mapper timeout 当数值 II。
- 不要因为 analytical 在 `multi-nested` 大量 tie 后碰巧选择 candidate-0 就把它当
  small-op predictor；40-query MAE 明显更差。
- 不要声称 Amoeba 已 faithful replay frozen shape；当前独立 pipeline 才直接传递并
  验证 x/y，详细限制见 `docs/AMOEBA_REPLAY_INTERFACE.md`。
- 不要声称整个 Amoeba `test/multi-cgra/taskflow` 已通过；静态全空间/最优性结果只有
  `parallel-nested`、`multi-nested`、`irregular-loop`，`attention` 仍是约束搜索。
- `irregular-loop` 的零秩 store workaround 仅匹配空 index list；不要把普通 indexed
  store 改写或丢弃 operand segment。
- `attention` 的 op cap 是 search heuristic，精确 grid packing 才是硬约束；mapper
  timeout 必须继续作为 censored。`resnet` 有 13 个 task、原始空间 8^13，不能先
  完整物化。
- shape 在 mapper 中仍是有向矩形；评估可以同时报告 transpose-equivalent 指标，但
  不要未经 mapper 证据把 4x12 与 12x4 的 compiled-II 标签强行改成相同数值。
