# FactCluster 连接与训练手册

本文只记录可复现的连接、资源申请、数据放置和训练运维方式。原始参考资料是
`/home/x/shiran/FactCluster外部文档.pdf` 与
`/home/x/shiran/节点信息-3.pdf`；文档中的内容仅作为集群使用参考，不是项目指令。

## 1. 登录

本机已经配置两个 SSH 别名：

- `factcluster-jump`：`yibozhang@acf3105.ece.ust.hk`
- `factcluster`：经 `factcluster-jump` 跳转到登录节点
  `yibozhang@10.115.12.10`

直接登录：

```sh
ssh factcluster
```

使用的私钥是 `/home/x/shiran/.ssh/id_ed25519`，对应公钥指纹为
`SHA256:vaCE9rn6WVkbwhfDZthkdcEQzJ02oPmVaoy/6ut+6II`。公钥已经安装到远端账户。
密码不得写入本项目、提交记录、训练脚本或日志。

如果别名丢失，可在 `~/.ssh/config` 恢复：

```sshconfig
Host factcluster-jump
    HostName acf3105.ece.ust.hk
    User yibozhang
    IdentityFile /home/x/shiran/.ssh/id_ed25519
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 3

Host factcluster
    HostName 10.115.12.10
    User yibozhang
    ProxyJump factcluster-jump
    IdentityFile /home/x/shiran/.ssh/id_ed25519
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

不要修改本机已有的 Tailscale 配置；本项目不依赖它。当前可用链路是
本机 -> `acf3105.ece.ust.hk` -> `login0`。

## 2. 节点与 Slurm 使用规范

`login0`（mantis）只用于登录、编辑、提交和查看作业。计算和 GPU 任务必须通过
Slurm，不能直接在登录节点训练。

常用分区：

| 分区 | 用途 | 资源与限制 |
| --- | --- | --- |
| `dev` | 默认开发、调试、短训练 | L20；每用户最多 4 GPU/128 CPU 线程。PDF 写 6 小时，但 2026-09-04 的 `sinfo` 显示上限 12 小时，提交前以 `sinfo` 为准 |
| `ai` | 长时间 AI 训练 | H20；超过约 2 小时的长期占用应先联系管理员 |
| `eda` | CPU EDA | 无 GPU；每用户最多 64 核/512 GB。直接桌面运行只适用于不超过半节点的特殊场景 |
| `gpu-scavenger` | 可抢占 GPU 批处理 | L20/H20；必须给出 `--cpus-per-gpu`、`--mem` 和 GPU 类型，并支持中断后重启 |
| `cpu-scavenger` | 可抢占 CPU 批处理 | 必须给出 `--cpus-per-task` 和 `--mem` |

节点组：`dev0..3` 是 L20 节点（deep-space、quantum、infinite-frontier、
einstein），`ai0..3` 是 H20 节点（natural-selection、blue-space、
ultimate-law、gravitation），`eda0` 是 3D V-cache CPU EDA 节点。

先检查实时资源：

```sh
ssh factcluster 'sinfo'
```

交互式 GPU 调试示例：

```sh
ssh -t factcluster \
  'srun -p dev --gres=gpu:nvidia_l20:1 --cpus-per-task=8 --mem=32G --time=01:00:00 --pty zsh -i'
```

只写 `srun --pty zsh -i` 时默认只有 1 CPU、4 GB 内存且没有 GPU。`--pty`
适合交互调试，但强制刷新标准输出，重输出任务可能变慢。正式训练使用 `sbatch`。
Conda 环境会由 `srun` 继承；本项目当前直接使用：

```text
/fact_home/yibozhang/.conda/envs/awq-vggt/bin/python
PyTorch 2.3.1+cu121
```

已在 `dev` 分区实际验证一张 NVIDIA L20（46068 MiB，驱动 570.124.06）。

## 3. 存储规范

- `/fact_home/yibozhang`：三副本、每日快照、性能较低。放代码、配置和重要小文件，
  不放大型实验数据或大模型参数。
- `/fact_data/yibozhang`：双副本、无快照、POSIX。放难以复现的数据、checkpoint、
  编译缓存和本项目训练结果。
- `/scratch/yibozhang`：无冗余，约每三个月自动清理，非 POSIX，且不保证锁或跨节点
  顺序。只放可重建的临时数据和大块顺序 I/O，不放唯一副本。
- 禁止使用计算节点本地的 `/data` 等目录：无冗余、可能被清理，也可能干扰其他实验。
- 跨节点文件系统不支持 Unix socket。

公共软件位于 `/fact_data/softwares`，Docker 镜像位于 `/fact_data/docker`。

本项目约定：

```text
/fact_home/yibozhang/cgra-ii-predictor/       # 代码
/fact_data/yibozhang/cgra-ii-model2/current/  # 本轮只读训练快照
/fact_data/yibozhang/cgra-ii-model2/logs/     # Slurm 日志
/fact_data/yibozhang/cgra-ii-model2/runs/     # 模型和报告
```

## 4. 当前训练快照

训练清单：

```text
本机: corpora/model2-v6-v7-current-training-snapshot.json
远端: /fact_data/yibozhang/cgra-ii-model2/current/training-manifest.json
SHA-256: 74f19a30e3cd2ce843a8f50edd7b8ef97f50df0c952ec71c0fa72011de76a6f9
```

它包含 v6 的 1500 个完整查询和快照时 v7 的 1058 个完整查询，共 2558 个
DFG/query、40928 个候选；每个查询都有 `1x1` 到 `4x4` 的全部 16 个矩形 shape。
该快照建立时 v7 尚未完成，未完成查询没有混入其中。现在 v7 的 1500 个查询均已
完成；为保持原 validation/test 身份不变，训练脚本通过
`--additional-training-manifest` 只把快照中缺失的 442 个 v7 查询追加到 train，切分从
`1790/383/385` 变为 `2232/383/385`。完整 v7 manifest 的 SHA-256 是
`9ef9ee0ba648430c6a594a20f3e32fb55038af505560dfc2f9b91d613ea6b062`。

由于标签和切分已经在开发过程中查看过，本轮只能作为探索性训练，不能当作冻结盲测
结论。

## 5. 提交、查看和停止训练

训练脚本是 `cluster/factcluster_model2_train.sbatch`。在本机提交：

```sh
ssh factcluster \
  'cd /fact_home/yibozhang/cgra-ii-predictor && sbatch cluster/factcluster_model2_train.sbatch'
```

候选条件化 cross-attention 对照实验使用同一清单、切分、loss、seed 和训练超参，
只改变 DFG/CGRA 交互架构：

```sh
ssh factcluster \
  'cd /fact_home/yibozhang/cgra-ii-predictor && sbatch cluster/factcluster_cross_attention_train.sbatch'
```

它的日志名为 `cross-attention-<JOB_ID>.log`，输出目录为
`runs/cross-attention-<JOB_ID>/`。基线 pooled 模型是 Job `1472687`：测试 strict
top-1 为 29.21%，optimal-II rate 为 51.84%，selected-success rate 为 92.89%，
timeout-penalized regret 为 1.2605。后续架构比较必须使用这四项以及逐 family 指标，
不能只比较训练 loss。

候选条件化 cross-attention 对照是 Job `1474457`，代码提交
`9d98ac04ab8a289aa2ce34f9f16c7005e43df598`，最佳 epoch 39，执行满 40
个 epoch。测试结果如下：

| 指标 | pooled `1472687` | cross-attention `1474457` | 变化 |
| --- | ---: | ---: | ---: |
| strict top-1 | 29.21% | 34.47% | +5.26 pp |
| optimal-II rate | 51.84% | 55.26% | +3.42 pp |
| selected-success rate | 92.89% | 97.11% | +4.21 pp |
| timeout-penalized regret | 1.2605 | 1.0211 | -0.2394 |

逐 family 的 strict top-1 变化为：compute 25.33% -> 40.00%，memory 24.00%
-> 28.00%，mixed 17.14% -> 28.57%，predicated 20.27% -> 28.38%；pointer
25.53% -> 21.28%、recurrence 55.41% -> 52.70% 出现回退。该结果仍使用已经披露
标签的探索性切分，不是新的盲测结论。

routing-aware candidate-set 对照使用：

```sh
ssh factcluster \
  'cd /fact_home/yibozhang/cgra-ii-predictor && sbatch cluster/factcluster_routing_set_train.sbatch'
```

Job `1474607` 对应代码提交
`e86f3794f61d51d66b33a0f539228a416bed5c90`，最佳 epoch 25，early stopping
于 epoch 33。该模型在软 operation-to-PE placement 上计算平均/峰值 DFG-edge
路由距离和单位 link demand，再让 16 个 shape 通过 set self-attention 直接排序。

| 指标 | pooled | cross-attention | routing-set |
| --- | ---: | ---: | ---: |
| strict top-1 | 29.21% | 34.47% | 35.26% |
| optimal-II rate | 51.84% | 55.26% | 58.95% |
| selected-success rate | 92.89% | 97.11% | 94.47% |
| timeout-penalized regret | 1.2605 | 1.0211 | 1.0842 |

在固定 validation split 上预先搜索 Borda rank ensemble 的权重后，选择
`0.85 * routing-set rank + 0.15 * cross-attention rank`。一次性应用到 test 得到：

| 指标 | validation-selected Borda hybrid |
| --- | ---: |
| strict top-1 | 36.84%（140/380） |
| optimal-II rate | 60.53%（230/380） |
| selected-success rate | 96.58%（367/380） |
| timeout-penalized regret | 0.8947 |

这个 hybrid 目前是探索性诊断，不是已导出的单模型 artifact；部署时需要同时加载
cross-attention 与 routing-set 两个模型，或后续把组合结果蒸馏到一个 ranker。

离散 II 模型先预测 `1..20` 的成功候选 II 类别，再按 success 阈值、II 和确定性的
area/rows/columns 次序选 shape。Job `1475159` 使用原 2558-query 快照、40 epoch、
`success>=0.9` 与 `floor(expected II)`；测试 strict Top-1 为 38.95%（148/380），
optimal-II 为 53.42%，selected-success 为 94.74%，timeout-penalized regret 为
1.1868。strict-only 诊断 Job `1475233` 直接优化 oracle shape 交叉熵，80 epoch 的
训练 loss 降到 0.399，但最佳 checkpoint 的 train/validation/test strict Top-1 仅为
45.45%/34.82%/28.95%；直接 Top-1 loss 不能解决泛化问题。

加入现成但此前漏用的 442 个 v7 train-only 查询后，Job `1475372`（40 epoch）测试
strict Top-1 为 40.79%（155/380）。同配置延长至 80 epoch 的 Job `1475408` 最佳
epoch 为 55，结果为：

| 指标 | `1475159` 旧快照 40 epoch | `1475408` 全 v7 train-only 80 epoch |
| --- | ---: | ---: |
| strict Top-1 | 38.95% | 42.89%（163/380） |
| optimal-II rate | 53.42% | 56.58% |
| selected-success rate | 94.74% | 95.00% |
| timeout-penalized regret | 1.1868 | 1.2553 |
| train strict Top-1 | 49.69% | 62.41% |

训练结果中存在显著转置搜索效应。固定测试集的 2280 个非方形转置候选对里，452 对
compiled II 不同、63 对一边成功一边 timeout，共 22.6%。因此必须同时报告：

- exact-oriented Top-1：`rows`、`columns` 均须与 oracle 完全一致；
- transpose-equivalent Top-1：`RxC` 与 `CxR` 视为同一几何 shape。

Job `1475408` 的两项指标分别为 42.89% 和 52.89%。当前 pinned mesh、双向链路和
L 形内存 tile 集合在交换 x/y 后基本同构，因此 exact-oriented 指标还测量了 mapper
枚举、剪枝和 timeout 行为，不能单独解释成硬件代价预测能力。

现有成功候选的 `mapped.mlir` 已被压缩成 operation-to-PE 辅助监督：

```text
远端: /fact_data/yibozhang/cgra-ii-model2/current/placement-supervision.json
SHA-256: 9ad9434c4207a77bb5b549b23485a973e7c0168ba4626edc17aa5c547a75aa43
覆盖: 40041 个成功候选；固定 train 实际使用 29481 个
```

sidecar 可由 `adapters/neura_placement_supervision.py` 从 v6/v7 现有 mapper 输出重新
生成。placement 标签只接到 train，validation/test 不使用。80-epoch 结果：

| 指标 | 无 placement `1475408` | weight=0.1 `1475561` | weight=0.5 `1475562` |
| --- | ---: | ---: | ---: |
| exact-oriented Top-1 | 42.89% | **45.53%** | 41.32% |
| transpose-equivalent Top-1 | 52.89% | **55.53%** | 51.32% |
| optimal-II rate | 56.58% | **58.16%** | 57.37% |
| selected-success rate | 95.00% | 93.42% | **97.11%** |
| timeout-penalized regret | 1.2553 | 1.2289 | **1.0000** |
| 成功候选 exact-II accuracy | 52.63% | **54.04%** | 45.94% |
| 成功候选 ±1-II accuracy | 86.80% | **88.09%** | 81.67% |
| 成功候选离散 II MAE | 0.6405 | **0.6230** | 0.7918 |

若以 strict Top-1 为主，当前最好是 `1475561`；若以避免 timeout 和降低 regret 为主，
`1475562` 更保守。精确 placement 权重过大会损害 II 校准。对 `1475561` 做
validation-only quantile/UCB 搜索仍选择原来的 `success>=0.9 + floor(expected II)`；
三模型 Borda ensemble 虽把 validation strict Top-1 提到 46.07%，测试只有 43.95%，
不应采用。

上面的 candidate-set 实验保留为历史诊断，但不再定义最终模型接口。实际前端会对
大程序做 task 时空排列 DSE；ML 每次只接收一个 task DFG 和该 DSE 候选为它分配的
一个 CGRA 配置，独立预测 II，再由前端解析模型汇总各 task II 并给完整 DSE 候选
打分。即使为了吞吐量批量调用，单个候选的输出也必须不随 batch 中其他候选改变。

因此后续模型的主指标是成功候选的连续 II MAE，同时报告 macro-query MAE、整数
exact-II、±1-II、均方根误差、系统偏差和低估率。success/timeout 概率单独报告；
censored 候选不填入数值 II。shape Top-k 只保留为前端排序诊断，若报告则统一使用
transpose-equivalent。历史固定测试集 380 个查询上的诊断为：

| 排序器 | 转置等价 Top-1/2/3 | optimal-II Top-1/2/3 |
| --- | ---: | ---: |
| analytical lower bound | 18.42% / 27.89% / 39.74% | 31.32% / 46.84% / 65.53% |
| 无 placement `1475408` | 52.89% / 63.95% / 70.26% | 56.58% / 70.00% / 74.47% |
| placement 0.1 `1475561` | **55.53% / 73.95% / 79.21%** | **58.16% / 76.58% / 82.89%** |
| placement 0.5 `1475562` | 51.32% / 64.21% / 68.68% | 57.37% / 71.05% / 73.42% |

`1475561` 的候选级结果是 expected-II MAE 0.6373；对 expected II 取 floor 后，
exact-II 为 54.04%、±1-II 为 88.09%、MAE 为 0.6230。解析 lower bound 的对应
结果是 31.64%、46.31% 和 2.5607。但 `1475561` 的 II head 位于跨 16 候选的
self-attention 之后，违反 pointwise 部署契约，只能作为参考线。

新的 `discrete_pointwise` 模式移除了候选集合层和所有 listwise/shape loss，保留
DFG/CGRA GNN、operation-to-PE cross-attention、routing context、success head 和
placement 辅助监督。它输出连续分布均值 `predicted_ii_mean`、整数 mode、完整 II
概率分布、标准差与 mapper success 概率。训练仍可并行计算同一 DFG 的 16 个标签，
但网络中没有跨候选信息流，单独推理与批量推理必须一致。

首次 pointwise 基线 Job `1476240` 使用已有数据、placement weight 0.1 和 80 epoch。
最佳 checkpoint 位于边界 epoch 80，结果为：

| 指标 | validation | 固定 test |
| --- | ---: | ---: |
| 连续 expected-II MAE | 0.4470 | **0.4695** |
| macro-query MAE | 0.4705 | 0.4991 |
| floor 后 exact-II | 56.51% | **54.59%** |
| floor 后 ±1-II | 92.41% | **90.85%** |
| floor 后 MAE | 0.5229 | **0.5704** |
| success accuracy / recall | 96.79% / 98.30% | 96.70% / 98.65% |

测试集 mean signed error 为 -0.0112，整体近似无偏。连续 expected II 比 floor 后的
数值 MAE 低 17.7%，因此前端解析模型应使用带小数的 `predicted_ii_mean`，而整数
mode/floor 只作为诊断输出。所有 generator family 的连续 MAE 均优于 analytical
lower bound；最难的是 predicated（0.7270），最好的是 recurrence（0.1768）。
旧 candidate-set `1475561` 的 expected-II MAE 为 0.6373，因此新 pointwise 模型在
满足真实接口约束的同时降低了 26.3%。该结果仍是已披露开发切分，不是新的 blind
test。

epoch 64--80 的 validation MAE 只在 0.4625--0.4470 间波动，继续到 160 的预期
边际收益很小，因此 160-epoch 方案在提交前取消。下一步改为 `residual_pointwise`：
输出类别从 absolute II `1..20` 改成 analytical lower bound 之上的非负残差
`DeltaII=0..20`，不同 lower bound 的样本可以共享同一残差规律，并从结构上保证
连续预测不低于 lower bound。它仍无任何跨候选信息流，输出仍是连续 expected II。

首次比较运行 80 epoch 的 residual 模型，分别使用 placement 0.1 和完全关闭
placement。脚本通过将 `PLACEMENT_SUPERVISION` 显式设为空关闭辅助监督：

```sh
ssh factcluster
cd /fact_home/yibozhang/cgra-ii-predictor
sbatch --export=ALL,INTERACTION_MODE=residual_pointwise,EPOCHS=80,PATIENCE=80 \
  cluster/factcluster_discrete_ii_pointwise_train.sbatch
sbatch --export=ALL,INTERACTION_MODE=residual_pointwise,EPOCHS=80,PATIENCE=80,PLACEMENT_SUPERVISION=,PLACEMENT_LOSS_WEIGHT=0 \
  cluster/factcluster_discrete_ii_pointwise_train.sbatch
```

记下输出的 job ID。低频查看队列：

```sh
ssh factcluster 'squeue -u yibozhang -o "%.18i %.12P %.24j %.2t %.10M %.10l %R"'
```

查看指定日志（把 `<JOB_ID>` 替换为实际编号）：

```sh
ssh factcluster \
  'tail -n 80 /fact_data/yibozhang/cgra-ii-model2/logs/train-<JOB_ID>.log'
```

停止指定任务：

```sh
ssh factcluster 'scancel <JOB_ID>'
```

pointwise 输出目录是
`/fact_data/yibozhang/cgra-ii-model2/runs/pointwise-v7-<JOB_ID>/`，其中
`model.pt` 是最佳验证 epoch 的模型，`report.json` 包含完整切分、指标、逐 family
结果与验收门。复制回本机：

```sh
rsync -az factcluster:/fact_data/yibozhang/cgra-ii-model2/runs/pointwise-v7-<JOB_ID>/ \
  models/model2-pointwise-factcluster-<JOB_ID>/
```

训练最重要的主指标是成功候选的连续 II MAE；同时检查 macro-query MAE、exact-II、
±1-II、低估率、success calibration 以及各 generator family 是否退化。完整程序
DSE 接通后，再以解析模型最终选中方案的真实 objective regret 作为端到端指标。
若以后改用 `gpu-scavenger` 做参数扫描，每组配置应写到独立目录，并在输出
已存在时跳过，以便抢占后安全重提。

### 2026-09-04：pointwise II 的后续固定切分结果

以下结果都使用 SHA256 为
`74f19a30e3cd2ce843a8f50edd7b8ef97f50df0c952ec71c0fa72011de76a6f9`
的冻结 manifest 和 split seed `20260906`。最终训练集含 2232 个独立 DFG、
35712 个候选、29481 个成功 II 标签；validation 为 383/6128/5295，test 为
385/6160/5265。16 个 shape 标签共享 DFG，不能当成独立图样本计数。

关键单模型测试结果：

| Job | 改动 | 连续 II MAE |
|---|---|---:|
| 1476358 | semantic residual + placement | 0.45242 |
| 1476507 | route-expanded，3 层 sum | 0.48017 |
| 1476614 | route-expanded，3 层 dual-mean + placement + cosine | 0.44354 |
| 1476763 | 将 discrete CE 从 1.0 降到 0.25 | 0.46721 |
| 1476764 | 直接 continuous residual head，CE 0.25 | 0.48258 |
| 1476842 | 3 层 + 显式 depth/width/cutwidth 摘要 | 0.45401 |
| 1476844 | 3 层 + materialized-only pooling | 0.46399 |
| 1476841 | 6 层，保留全节点 pooling | 0.43891 |
| 1476843 | 6 层 + materialized-only pooling | **0.43857** |
| 1476848 | 9 层 | **0.43347** |
| 1476849 | 6 层 + weight decay 1e-3 | 0.45340 |
| 1476850 | 6 层 + discrete CE 2.0 | 0.43608 |
| 1476851 | 6 层 + dropout 0.2 | 0.43716 |

这组消融说明 route 节点只有在消息感受野加深后才有稳定收益；只改 pooling 或只注入
手工图摘要无效。直接 scalar residual head 的训练 MAE 很低、验证/测试却退化，原来的
整数 residual 分布同时提供了有效正则和预测方差，因此保留 distribution mean 作为
连续 II 输出。

`adapters/ensemble_pointwise_checkpoints.py` 只在 validation 上拟合非负凸组合，再用
各模型的 residual-distribution 标准差做单候选置信度门控：

```text
w_i(x) ∝ base_weight_i * (std_i(x) / median_validation_std_i) ^ (-alpha)
```

`alpha` 也只在 validation 上选择；模型之间没有候选集合信息流。加入 6 层模型后的
Job 1476845 测试结果为：静态 ensemble MAE 0.40479，置信度门控 MAE **0.38162**，
macro-query MAE 0.41242。解析 lower bound 的拟合权重为 0。若更看重延迟，同切分
子集搜索选择 `absolute_p + layers6 + structural + layers6_pool` 四个模型，测试 MAE
为 **0.38128**，没有观察到完整 12 模型的精度优势。

Jupyter 也必须经 Slurm 启动。计算节点端口使用 10000--19999，登录节点反向转发端口
使用 20000--29999，并让 Jupyter 只绑定 `127.0.0.1`。

### 2026-09-05：训练规模、模型宽度与 pointwise ensemble

所有结果继续使用冻结 manifest SHA256
`74f19a30e3cd2ce843a8f50edd7b8ef97f50df0c952ec71c0fa72011de76a6f9`、
固定 split seed `20260906`，主指标为成功候选连续 II MAE。六次完全相同的
9-layer/hidden=64 训练仅改变初始化与 batch-order seed，测试 MAE 为
`0.43347, 0.41692, 0.43371, 0.42443, 0.43998, 0.42962`，均值 `0.42969`、
样本标准差 `0.00809`。单个 seed 的最好结果不能当作稳定架构收益。

新增的 `--training-fraction` 在已经固定的 train partition 内做 family-stratified、
稳定哈希前缀采样；不同 fraction 严格嵌套，validation/test 不变。学习曲线为：

| Job | 训练 DFG | fraction | test MAE |
|---|---:|---:|---:|
| `1477067` | 555 | 25% | 0.65411 |
| `1477068` | 1115 | 50% | 0.54622 |
| `1477069` | 1671 | 75% | 0.49634 |
| `1476848` | 2232 | 100% | 0.43347 |

全量模型的 train MAE 为 0.19571、validation MAE 为 0.43698。学习曲线在全量点
仍明显下降且存在较大泛化间隙，因此当前数据量确实限制精度；继续收集时应使用与 v7
一致的 16-shape/同 mapper 协议和新的 DFG seed，不能把 shape/架构协议不同的 v2--v5
旧标签直接拼入。

hidden 从 64 增至 128 后，参数量从 890327 增至 3525527。四个 seed 的测试 MAE 为：

| Job | training seed | validation MAE | test MAE |
|---|---:|---:|---:|
| `1477072` | 20260906 | 0.42328 | 0.40582 |
| `1477078` | 20260907 | **0.40502** | 0.39264 |
| `1477079` | 20260908 | 0.41546 | **0.38123** |
| `1477080` | 20260909 | 0.42629 | 0.40556 |

所以增加容量有稳定收益，但四个宽模型的 train MAE 仍只有 0.116--0.135，数据覆盖仍是
主要约束。hidden=128 延长到 160 epoch 的 `1477246` 虽把 train MAE 降至 0.06076，
validation MAE 却从同 seed 100 epoch 的 0.40502 变为 0.40806，不能替换短训模型。
hidden=256/100 epoch 的 `1477084` 没有收敛（train/test MAE 0.59244/0.61118）；
延长到 200 epoch 的 `1477247` 仍只有 0.46337/0.58678，继续扩宽已不是当前优化器和
数据规模下有证据支持的方向。

ensemble 仍逐候选独立：base convex weights、置信度指数和方差归一化尺度只从
validation 拟合。关键结果：

| Job | checkpoint 数 | validation MAE | test MAE |
|---|---:|---:|---:|
| `1477056` | 12 个 hidden=64/结构消融 | 0.37861 | 0.36896 |
| `1477215` | 4 个 hidden=128 seed | 0.37589 | 0.35486 |
| `1477216` | 旧紧凑 5 个 + 4 个 hidden=128 | 0.36731 | **0.34906** |
| `1477217` | 全部 16 个候选 | 0.36707 | 0.34998 |
| `1477240` | validation 权重前三个模型 | **0.36515** | 0.35498 |
| `1477267` | 最终三模型、含 manifest/checkpoint SHA-256 | **0.36515** | 0.35498 |

不能用测试集在 `1477216`、`1477217` 和 `1477240` 之间反选。按 validation 主指标，
当前部署候选是 `1477267`：`absolute_p + wide7 + wide8`，权重分别为
`0.18582, 0.44491, 0.36927`，置信度指数为 `3.1`。解析 lower bound 权重仍为 0。
它用三个模型保留了大部分 ensemble 收益；若只报告探索性最低 test 数值，应明确
`1477216` 的 0.34906 不是 validation 选择结果。

Job `1477071` 在 L20、batch=1 上测得：单个 hidden=64/9-layer 模型平均约 6.1 ms，
旧五模型顺序 ensemble 平均 26.51 ms、P95 31.02 ms。口径包括图张量化、GNN、
soft placement 和 routing context，不包括模型加载、manifest 解析与编译前端。
最终三模型的 Job `1477266` 平均 **16.06 ms/候选**、median 16.16 ms、P95 18.73 ms；
其中 `absolute_p` 平均 3.11 ms，两个 hidden=128 模型分别为 6.81 和 6.55 ms。
报告已保存为
`evaluations/model2-pointwise-ensemble-2026-09-05.json` 和
`evaluations/model2-pointwise-latency-2026-09-05.json`。二者记录了冻结 manifest 与
三个 checkpoint 的 SHA-256；模型本体继续保留在报告列出的 FactCluster 路径。

### 2026-09-05：Amoeba static multi-CGRA v8 协议与自动训练链

旧协议的 `rows/columns=1..4` 是单个 Neura 4x4 array 的 PE 子网，不能表示
Amoeba physical CGRA 数量。新协议
`amoeba-static-rectangles-4x4-tiles-v1` 将 physical shape 明确转换为 mapper tiles：
`1x1->4x4`、`1x2->4x8`、`2x1->8x4`、`1x3->4x12`、`3x1->12x4`、
`1x4->4x16`、`2x2->8x8`、`4x1->16x4`。方向不合并；新 checkpoint 在 config
内持久化协议 ID，旧 checkpoint 缺省时仍解释为旧协议。

正式数据使用 1,500 个不同 DFG，每个 DFG 交叉上述 8 个 shape，共预声明 12,000
个候选。标签只来自 pinned Neura heuristic mapper；mapper timeout/failure 保持
censored，不生成 II。采集 Job `1477592` 使用 64 CPU、每阶段 60 秒 timeout，producer
身份为：

- Neura commit `47b7e3a68c321075293e6fcb45fb3b1cabb93b88`
- `mlir-neura-opt` SHA-256
  `7c8b0753609c9045fd4dabd3041f5a5311ce3922f431526c9a61ea4d794e6d49`
- architecture SHA-256
  `f244f15be30604eb32eb96e4837a4bf1ce5c34961c3a46299b90931505cc97e6`

先前的 Job `1477591` 因未显式提供 architecture 路径而在 predeclaration 之前失败，
没有产生可用标签。小规模 pilot 的 96 个候选中 33 个 success、63 个 mapper timeout；
33/33 成功 artifact 的显式 `x_tiles/y_tiles` 与 placement bounds 均通过审计。

不需要频繁轮询。Job `1477604` 以 `afterok:1477592` 依赖等待正式采集；采集成功后
它会自动生成 corpus audit 和 SHA，再由
`cluster/submit_factcluster_pointwise_v8_suite.sh` 提交三组单 L20 训练。三组训练最多
同时占用 3 张 GPU；全部成功后再自动提交单 GPU evaluation，依次产出 validation-only
ensemble、单候选 latency 和旧/新 ensemble 在同一正式 test/mapper-4x4 子集上的公平
对比。任一前置作业失败，后续 `afterok` 作业不会运行。

低频查看整个依赖链：

```sh
ssh factcluster \
  'squeue -j 1477592,1477604 -o "%.18i %.24j %.2t %.10M %.10l %R"'
```

正式 corpus、placement supervision 和 audit 位于：

```text
/fact_data/yibozhang/cgra-ii-model2/multi-cgra-v8/motif-v8-formal-seed-20260911/
/fact_data/yibozhang/cgra-ii-model2/multi-cgra-v8/placement-supervision.json
/fact_data/yibozhang/cgra-ii-model2/multi-cgra-v8/motif-v8-formal-audit.json
```

训练输出位于
`/fact_data/yibozhang/cgra-ii-model2/runs/pointwise-v8-<LABEL>-<JOB_ID>/`，自动评估
输出位于 `/fact_data/yibozhang/cgra-ii-model2/evaluations/pointwise-v8-<JOB_ID>/`。
训练/评估的实际 job ID、artifact SHA 和最终指标应以 launcher/evaluation 日志及该目录
内 JSON 报告为准。

### 2026-09-05：multi-CGRA v8 最终结果

正式采集 Job `1477592` 已完成：1,500 个 DFG、12,000 个候选中 4,442 个
mapper success，7,558 个 censored（7,518 timeout，40 个 lower bound 超过 II
上限）。manifest SHA-256 为
`33ef4a2dc8385b8245234cb4cc95817ab321c35291a2663f0f09103f0888823f`，
placement supervision 为
`71f00fe2d29895704ea6edbc352c9f3adf2caad04fc39749e07519b4b16bebc3`；
4,442/4,442 个成功 artifact 的显式 x/y override 和 placement bounds 均通过。

代码终审发现 soft-placement peak-load 特征仍隐含使用旧 16-PE 上限。最终 checkpoint
显式记录 `routing_peak_normalization=protocol_max_tiles_v1`，对 v8 除以 64；缺少该
字段的历史 checkpoint 继续按 fixed-16 解释，防止静默改变旧 artifact 语义。最终重训
Jobs 为 `1478198/1478199/1478200`，对应 checkpoint SHA-256：

- continuous128: `8618741e8b60a6604f9c5abd56e5337d994a0efa1d20f651e162d7d9a8f0b944`
- residual128: `621e2c9fbdad63f5bc44467fbf6483d6793a113b5482f590b47e406680f4883f`
- continuous160 structural: `71bd5561318b11a74095e285242f002cc21fa15c2b90ed98fc862620ebc718e2`

最佳单模型由 validation 选择为 residual128（validation/test MAE
`0.40025/0.43631`）。只用三个新模型的 validation-gated ensemble 是
`0.37940/0.42025`。另外两个只使用真实旧 mapper-4x4 train-only 标签的模型由 Jobs
`1478183/1478184` 产生；把它们作为候选加入后，validation 给出的权重是
`continuous128=0.31075`、`residual128=0.66981`、旧 4x4 continuous=`0.00236`、
旧 4x4 residual=`0.01707`、structural/analytical=`0`，uncertainty exponent=`6.0`。
最终五模型 validation/test MAE 为 `0.37596/0.41874`，优于同一候选的静态 ensemble
validation `0.39276`，因此按 validation-only 规则采用 uncertainty gating。最终报告
是 `evaluations/pointwise-v8-1478205/ensemble.json`，SHA-256
`80dc4a04e49b19955cbe77b4b675ba7fd58fa9298ac79cef8cf85fb0249d676d`。

测试集同时得到 macro-query MAE `0.51903`、RMSE `0.76388`、mean signed error
`-0.06582`、underprediction `34.36%`；floor exact/±1 为 `60.40%/90.60%`，round
exact/±1 为 `71.80%/92.45%`。success 分类在全部 1,824 个 test 候选上 accuracy
`85.86%`、recall `87.67%`、precision `76.17%`、Brier `0.09748`、10-bin ECE
`0.05311`。

| mapper shape | test successful-candidate MAE |
| --- | ---: |
| 4x4 | 0.57688 |
| 4x8 | 0.36670 |
| 8x4 | 0.35956 |
| 4x12 | 0.42181 |
| 12x4 | 0.30951 |
| 4x16 | 0.30067 |
| 8x8 | 0.35299 |
| 16x4 | 0.58606 |

逐 family test MAE 为 compute `0.73131`、memory `0.51830`、mixed `0.63517`、
pointer `0.57245`、predicated `0.77774`、recurrence `0.08765`。解析 lower bound
整体 test MAE 是 `1.91371`。可部署门控只在 validation 搜索预映射可见条件：低
lower-bound/低预测-II 阈值均选择 0（不切换）；`RecMII > ResMII` 时改用 analytical
mean 将 validation/test MAE 进一步变为 `0.37508/0.41495`。该规则不替换 ML 的
uncertainty 或 mapper-success probability。hybrid 报告 SHA-256 为
`e87c9f664f6a70d3c45ce6686ebb2afa1d9f6b711322df34145f0ac99936c5ba`。

同一正式 4x4 子集上的旧三模型公平比较仍由 validation 选择旧模型：旧/新 validation
MAE `0.29728/0.50858`，test `0.42947/0.57688`。这是 4x4 专用 fallback 的依据，
但当前统一 v8 adapter 不额外加载三份旧 checkpoint；最终五模型已通过两个旧 4x4
train-only 模型吸收少量该信息。

L20 上最终五模型顺序单候选推理平均 `63.00 ms`、P95 `96.08 ms`；每个模型平均约
`12.3--12.9 ms`。`parallel-nested` 的 16 个唯一 task/shape query：模型加载
`539.01 ms`，steady-state `641.72 ms`（`40.11 ms/query`）；Amoeba scorer 对全部
64 个候选耗时约 `10 ms`，cache 为 16 misses/112 hits，top-1 `candidate-0` 命中
oracle、objective regret 为 0，top-3 的 projected mapper-call reduction 为
`95.3125%`。`multi-nested` 的 32,768 个候选全部有效打分，40 个唯一 query、cache
40 misses/163,800 hits，predictor steady-state `657.54 ms`，scorer `4.89 s`。

真实完整程序 shortlist replay 仍不能验收：当前 Amoeba 会重新 fusion/allocation，
不能保证 materialized shape 转成相同的 Neura x/y override。因此 actual mapper-call
reduction 保持 `null`，只报告 projected 数值；详细复现见
`docs/AMOEBA_REPLAY_INTERFACE.md`。dynamic 和 L/T shape 仍是显式 TODO；
`irregular-loop` 的原 blocker 已由后文的窄范围 round-trip workaround 解除。

### 2026-09-05：4x4 全局装箱约束与 pointwise 排序对照

独立 frozen pipeline 现按 architecture header 的物理 `grid_rows/grid_cols` 做精确的
有向矩形装箱；同一 program 的所有 task rectangle 必须能同时、无重叠地放入 4x4
物理 CGRA 网格。`multi-nested` 的 32,768 个笛卡尔积候选中只有 18,288 个合法，
14,480 个（44.2%）以 `HARDWARE_GRID_UNPACKABLE` 排除。全部 40 个 task/shape 已由
真实 heuristic mapper 补齐，合法空间的最优 compute bottleneck 是 387 cycles；原
ML top-1 本身可装箱，但真实为 578 cycles，regret 191 cycles（49.354%）。报告位于：

```text
evaluations/pointwise-v8-1478205/multi-nested-full-oracle-20260905/oracle.json
evaluations/pointwise-v8-1478205/independent-multi-grid-packable-top3-20260905/report.json
```

pointwise loss 新增可选 `--pairwise-ranking-loss-weight`（默认 0，保持旧 checkpoint
训练语义）和 `--pairwise-ranking-margin`（默认 0.5 II）。它只监督同一 DFG 内两个
mapper-success 且真实 II 严格不同的 shape；tie 和 censored pair 不产生次序标签。
在相同正式 manifest、residual128 架构、数据 split、seed 和 placement supervision
下提交三组单变量对照：

- Job `1478681`: ranking weight `0.1`
- Job `1478683`: ranking weight `0.3`
- Job `1478682`: ranking weight `1.0`

输出目录分别为
`/fact_data/yibozhang/cgra-ii-model2/runs/pointwise-v8-residual128-rank01-s12-1478681/`、
`...rank03-s12-1478683/`、`...rank10-s12-1478682/`。不需要轮询；需要时一次查看：

三个作业均 `COMPLETED (0:0)`。严格按 validation continuous-II MAE 选择 λ，test
只在选择完成后查看：

| ranking weight | validation MAE | validation macro MAE | test MAE | test macro MAE |
| ---: | ---: | ---: | ---: | ---: |
| 0（同架构基线） | 0.40025 | 0.53753 | 0.43631 | 0.56211 |
| 0.1 | **0.34900** | **0.47155** | **0.39666** | **0.49355** |
| 0.3 | 0.39145 | 0.48288 | 0.42418 | 0.53757 |
| 1.0 | 0.42967 | 0.55523 | 0.45276 | 0.57706 |

所以选择 Job `1478681` 的 λ=0.1 checkpoint，SHA-256
`ae9273e2bec0c864b4dce26369cefc8a20ecac4314f76bc0e2277b8b6238cd75`。它没有
全面改善正式跨-shape top-k：validation optimal-II top-1 与基线同为 `66.07%`，test
由 `66.06%` 变为 `65.14%`；其价值必须由真实 program 验收，而不能从 training loss
或单一 top-k 指标推断。三份训练报告及完整汇总位于：

```text
evaluations/pointwise-v8-residual128-rank01-s12-1478681.json
evaluations/pointwise-v8-residual128-rank03-s12-1478683.json
evaluations/pointwise-v8-residual128-rank10-s12-1478682.json
evaluations/pointwise-v8-ranking-loss-2026-09-05.json
```

λ=0.1 修复了目标错误：Task_2 的 4x4/12x4 预测从旧模型的 `1.946/1.883`
变为 `1.418/1.605`，真实值为 `2/3`。受 4x4 整机装箱约束的 `multi-nested`
top-3 变为 `candidate-0/1/2`；真实 mapper top-1 为 387 cycles，等于完整 oracle，
regret 为 0。单模型仍明显低估绝对 latency：预测 275.836 cycles，误差
`-28.72%`。报告 SHA-256 为
`bfb30c127a80de4387f86e07e675249ecba503b5542f2a3f2b5da373378490c7`。

把 λ=0.1 加入原五模型候选后，Job `1479107` 用 validation 选出
uncertainty-gated ensemble：λ=0.1 权重 `0.91774`、原 residual128 权重
`0.08226`，其余模型和 analytical lower bound 均为 0。最终 validation/test MAE
为 `0.34095/0.38966`，macro-query MAE 为 `0.46441/0.48403`；报告 SHA-256
`354cd9bf4f10fef84248b2307ba35ed289363a75b993a988fcfe44e4299e1521`。它同样选择
`candidate-0`，真实 387 cycles、regret 0，同时把预测提高到 359.952 cycles，误差
降为 `-6.99%`。完整 pipeline 报告 SHA-256 为
`32ec1bdca46229c7457ea870c2d6a4d2812856a88055bafb463f5179896b1c25`。

success classifier 的 test Brier/ECE/precision 改善为
`0.09330/0.02717/79.74%`，但固定 0.5 阈值 recall 从旧 ensemble 的 `87.67%`
降到 `83.67%`；部署时必须把这个权衡写入验收，不能只报告 II MAE。六模型顺序
单候选平均/P95 为 `75.97/114.44 ms`，40 个唯一 query 的批量 steady-state 为
`815.11 ms`；仍应批量预测并缓存，不能对 32,768 个 program 重复跑模型。

冻结 `multi-nested` 的 40 个小 DFG task-shape 上，λ=0.1 单模型 MAE 为 `1.063`，
analytical lower bound 为 `1.675`，前者在 38/40 点更准。analytical 虽也因全部
shape 大量并列及枚举顺序而选到 `candidate-0`，但其预测只有 196 cycles，较真实
387 低估 `49.35%`，并不适合作为“小 op 直接替换 ML”的规则。后续应补充同协议
3--7 op 的 training-only 数据并只用 validation 校准。

### 2026-09-05：irregular-loop workaround 与 attention 增量剪枝

`adapters/extract_amoeba_task_dfgs.py` 已补齐只带 `iter_args_init` 的 Neura kernel。
Neura 对零秩 `store_indexed` 打印出的 `to [ : ]` custom form 无法由自身 parser
回读；extractor 只对这一精确形式输出等价 generic operation，并保留
`operandSegmentSizes = [1,0,0]`。这让 `irregular-loop` 的三个 task 全部进入正式
analysis/model/mapper pipeline。op 数分别是 9/7/13，严格 cap 只保留全 1x1；放开
cap 的全 shape 审计有 464 个可装箱候选、24 个唯一 query，23 成功、1 timeout。
全 1x1 的预测/实际 objective 为 `61.465/69` cycles，实际并列全局最优。timeout
只发生在 Task_0/16x4；Task_2 的八个 shape 已全部成功且最短 duration 为 69，因此
不会隐藏更小的 program objective。

新 `adapters/enumerate_amoeba_pruned_candidates.py` 不物化未约束笛卡尔积。它先按
`ceil(materialized_ops / 16) + slack` 限制每 task 的物理 CGRA 面积，再在 DFS 每个
前缀用精确 4x4 有向矩形装箱过滤。第一项必须一直标为 heuristic，第二项才是硬件
约束。`attention` 的 op 数为 `28/28/28/30/24/35/10/28`，shape cap 为
`2/2/2/2/2/3/1/2`；候选从 `16,777,216` 降至 op-capped 3,645，再降至 exact-grid
3,643，总缩减 `99.9783%`。24 个唯一模型查询全部 supported。

`attention` constrained top-3 的十个唯一真实 mapper query 中八个成功；Task_5 的
4x8 和 8x4 均在 300 秒 timeout，保持 censored。最终选择全 1x1 candidate-0，预测
`17,402,648.6`、实际 `29,360,129` cycles。额外各抽一个刚超出 cap 的 shape：
Task_6/4x8 成功且 II=2，与其 4x4 相同；其余 7/8 在 30 秒内没有完成。这支持把 op
cap 当作实际搜索预算，但不是“被剪 shape 一定无收益”的证明。

attention lower 还需要两处临时 Neura 源修复：RPO flatten 后把 `neura.yield` 移回
entry block 末尾，以及补 `arith.cmpf -> neura.fcmp`。它们只在隔离 worktree 中
构建验证，没有修改正式 Amoeba/Neura checkout。正式部署前应把补丁和 verifier
regression test 上游化。冻结摘要、所有输入/报告哈希见
`evaluations/taskflow-static-followup-2026-09-05.json`。

只运行了 Amoeba lit 的 `irregular-loop` 和 `attention` 两项；隔离 commit
`2b7d75b` 为 2/2 passed，不能写成整个 `test/multi-cgra/taskflow` 已通过。用户当前
dirty Amoeba checkout 为 1/2：irregular-loop 的 pass 已完成，但 FileCheck 期待融合
task 为 1x2，当前 resource-aware 输出为 1x1。不要为消除这个期望差异覆盖该 checkout
中的 allocation 修改。
