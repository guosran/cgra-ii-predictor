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

从后续实验开始，shape 主指标统一改为 transpose-equivalent，exact-oriented 只作为
诊断项保留。Top-k 表示按模型 selection cost 排名前 k 的候选中是否包含 oracle 的
转置等价 shape；optimal-II Top-k 表示前 k 个候选中是否至少有一个成功候选的
compiled II 等于该查询所有成功 shape 的最小值。censored 候选不填入数值 II。
固定测试集 380 个合格查询的结果为：

| 排序器 | 转置等价 Top-1/2/3 | optimal-II Top-1/2/3 |
| --- | ---: | ---: |
| analytical lower bound | 18.42% / 27.89% / 39.74% | 31.32% / 46.84% / 65.53% |
| 无 placement `1475408` | 52.89% / 63.95% / 70.26% | 56.58% / 70.00% / 74.47% |
| placement 0.1 `1475561` | **55.53% / 73.95% / 79.21%** | **58.16% / 76.58% / 82.89%** |
| placement 0.5 `1475562` | 51.32% / 64.21% / 68.68% | 57.37% / 71.05% / 73.42% |

`1475561` 的前 1/2/3 个候选中至少有一个 mapper 成功的比例分别为
93.42% / 98.68% / 99.47%。因此当前主要瓶颈不是候选召回，而是把已进入 Top-2/3
的最优候选排到第一。下一轮优先保持 placement weight 0.1，移除 exact-oriented
one-hot tie-break，改用 transpose-equivalent target，并按 validation
transpose-equivalent Top-1 选择 checkpoint；若允许少量 mapper 调用，则直接映射
模型 Top-2 后按真实 II 选择，是当前收益/开销最明确的 hybrid 路径。

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

输出目录是
`/fact_data/yibozhang/cgra-ii-model2/runs/retrain-<JOB_ID>/`，其中
`model.pt` 是最佳验证 epoch 的模型，`report.json` 包含完整切分、指标、逐 family
结果与验收门。复制回本机：

```sh
rsync -az factcluster:/fact_data/yibozhang/cgra-ii-model2/runs/retrain-<JOB_ID>/ \
  models/model2-factcluster-<JOB_ID>/
```

训练最重要的主指标是完整 shape 块上的 strict top-1 accuracy；同时检查 selected
success rate、optimal-II rate、timeout-penalized regret 以及各 generator family
是否退化。若以后改用 `gpu-scavenger` 做参数扫描，每组配置应写到独立目录，并在输出
已存在时跳过，以便抢占后安全重提。

Jupyter 也必须经 Slurm 启动。计算节点端口使用 10000--19999，登录节点反向转发端口
使用 20000--29999，并让 Jupyter 只绑定 `127.0.0.1`。
