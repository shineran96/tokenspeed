> 历史记录：保留当时的配置、指标和状态；当前实现状态以分支代码为准。完整证据位置见本目录上一级 README。

# DeepEP + Humming 容量 256/32 对比

2026-11-23。17 decode DP8/EP8，19 prefill TP4/QCP4/EP4。

算子测试、端到端各三轮 A/B 与候选 profile 已完成。固定全局 C32 下，仅将容量 256 调至 32，满载吞吐提升 **24.16%**，TPOT 降低 **19.46%**。

## 正式服务结果

统一使用 32 请求共同输出时间段的中间 80% 窗口，逐请求累计输出 token 差分统计，再取三轮均值。每个窗口的所有 batch 样本均为 `[4]*8`。Profile 轮不参与服务成绩。

| 配置 | 满载 TPOT ms | 满载输出 token/s | 整请求 TPOT P50 三轮均值 ms | 整轮流式输出 token/s |
|---|---:|---:|---:|---:|
| 原容量 256 | 6.116 | 5232.0 | 6.109 | 4795.7 |
| 优化容量 32 | 4.926 | 6496.0 | 4.914 | 5854.8 |

满载吞吐 +24.16%、TPOT −19.46%；包含进出场阶段的整轮输出吞吐 +22.09%。不同口径不能交叉相除。

| 容量 | 轮次 | 满载 TPOT ms | 满载 token/s | 满 batch 样本 | Host step 中位数 ms |
|---|---|---:|---:|---:|---:|
| 256 | 1 | 6.1164 | 5231.8 | 513/513 | 33.644 |
| 256 | 2 | 6.1096 | 5237.6 | 512/512 | 33.657 |
| 256 | 3 | 6.1227 | 5226.5 | 514/514 | 33.644 |
| 32 | 1 | 4.9215 | 6502.0 | 488/488 | 27.051 |
| 32 | 2 | 4.9321 | 6488.1 | 479/479 | 27.110 |
| 32 | 3 | 4.9246 | 6498.0 | 485/485 | 27.092 |

两组正式各 96 请求全部成功。连同预热与候选 profile，共核对容量 32 的 160 个、容量 256 的 128 个 4K 请求：输出均为 4096、cached_tokens 均为 130816、终态接受长度均为 5.50、接受率均为 0.8993，preemption 为 0。每组 8 rank 的 GPU NIC handler 日志均确认。

运行顺序为容量 32 后容量 256，保持 prefill/gateway 不重启以复用相同 prefix cache；不是交替随机 A/B。权重、运行时、后端和接受长度设置相同，各自完成预热。绝对性能仍受设备状态与运行顺序影响。

## 改动与适用范围

只调整启动参数 `--low-latency-max-num-tokens-per-gpu`，没有修改运行时计算源码。D 的 target/draft 均使用独立 Humming W4A8，DeepEP low_latency，GPU NIC handler，PDL 与 CUDA Graph 开启。全局 C32，每 rank C4，target verify 每请求 6 行，每 rank 24 行。输入 131072、输出 4096，模拟接受长度 5.5。

容量 32 是本轮固定负载的候选配置，不是通用默认值。某 rank 实际输入或 Graph padding 超过 32 行时必须使用更大容量。上下文 128K 不决定 MoE 输入行数。

## 独立进程算子测试

真实 DSV4.1 target 第 0 层、draft 第 0 层权重；8 GPU，Graph replay，7 个窗口，每窗口 100 次。每卡取窗口中位数，再取最慢 rank；容量 256、32 分别在独立进程建立共享 buffer。以下包含 dispatch、专家计算和 combine。

| 阶段 | 每 rank 行数 | 容量 256 μs | 容量 32 μs | 耗时下降 |
|---|---:|---:|---:|---:|
| Target | 24 | 549.088 | 408.950 | 25.5% |
| Target | 32 | 581.885 | 434.296 | 25.4% |
| Draft | 4 | 362.345 | 161.127 | 55.5% |
| Draft | 32 | 451.992 | 249.934 | 44.7% |

初始诊断在同一共享容量 256 buffer 中创建 dispatcher 256/32，两组耗时相同。源码确认 low_latency 的固定 wire layout 直接读取 `DeepEPBuffer._num_max_dispatch_tokens_per_rank`，所以该诊断没有实际缩小通信与计算张量。已改为上表的独立进程测试，初始结果不能当作容量 A/B。

## 数值与边界

每个形状均执行正常路由、集中路由、奇数 rank idle、全 idle、输入变更与恢复，以及 eager/Graph 对照。全 rank finite 检查通过，测试运行完成；集中路由覆盖每来源 rank 32 行边界。

Humming 在容量 256 下自身重复运行也非逐位一致。记录归一化 L2 `||a-b||/||b||`，跨容量误差与两组自身重复运行误差同量级；不能宣称 bitwise 等价，也不能由此推导完整模型质量无损。初次逐位断言失败的日志保留。

| 阶段/行数 | 跨容量最大 rel L2（8 rank、6 场景） | 原容量 eager 重复最大 rel L2 |
|---|---:|---:|
| Target/24 | 0.005115 | 0.006079 |
| Target/32 | 0.005829 | 0.005998 |
| Draft/4 | 0.011451 | 0.012943 |
| Draft/32 | 0.009683 | 0.009683 |

## 已完成的容量 32 服务与历史 CuTeDSL 对比

同为 17 D8 + 19 P4、全局 C32、输入 131072、输出 4096、模拟接受长度 5.5、PDL/Graph 开启。统一采用 32 请求共同输出时间段的中间 80% 窗口，三轮均值。本次各窗口所有 batch 样本均为 `[4]*8`。

| 服务方案 | 满载 TPOT ms | 输出 token/s | 单 D 卡 token/s |
|---|---:|---:|---:|
| DeepEP + Humming，容量 32 | 4.926 | 6496.0 | 812.0 |
| 历史 CuTeDSL MegaMoE，PDL on | 4.526 | 7069.8 | 883.7 |

CuTeDSL 相对本次 Humming 吞吐高 8.83%、TPOT 低 8.12%。历史组的 target 使用 CuTeDSL、draft 使用 FlashInfer CUTLASS W4A8；本次 target/draft 都为独立 Humming，因此是服务方案对比，并非隔离的 target kernel 对比。历史测试不是与本轮交替执行的配对实验。

容量 32 三轮满载 TPOT 为 4.9215/4.9321/4.9246 ms，吞吐为 6502.0/6488.1/6498.0 token/s；整请求 TPOT P50 为 4.9073/4.9202/4.9149 ms。满载窗口分别有 488/479/485 个 batch 样本，均为各 rank C4。

## Profile 验证

GPU0 完整 Graph 去除首样本后取 39 个样本中位数，其他 GPU 为 38 个；target 每 Graph 验证 40 次 FC1、40 次中间量化、40 次 FC2、80 次 dispatch 和 80 次 combine。与上一轮容量 256 profile 比较：

| 时间 ms/step | 容量 256 | 容量 32 |
|---|---:|---:|
| 整步 Graph | 34.455 | 27.604 |
| Target MoE 路径时间段 | 22.740 | 16.532 |
| Target 中间 SwiGLU/FP8 | 8.349 | 2.040 |
| Target FC1 | 5.729 | 5.766 |
| Target FC2 | 3.106 | 3.105 |
| Target dispatch | 0.920 | 0.906 |
| Target combine | 3.466 | 3.464 |
| Draft 尾段 | 2.593 | 1.968 |

整步降低 19.88%，中间处理累计时间降低 75.57%。主要收益集中在中间处理，专家 GEMM 与通信基本不变。GPU kernel 累计时长与路径跨度的定义不同，不能将表中所有行相加。

容量 32 的 padded 行数为 `48 × (32 × 8) = 12288`，但自动配置选择后实际量化网格是 36864 CTA、block 128 threads；原容量为 98304 CTA、block 160 threads。源码 launcher 支持每行多个 column task，不能将 padded 行数缩小 8 倍等同于实际 CTA 数或时间缩小 8 倍。8 GPU 所有已分析 target 层均观察到 36864 CTA。

## 不同并发的自动适配方案

优先方案是固定足够的通信容量，令 Humming 中间量化按 GPU 有效行计数调度，从而随并发变化减少空行工作并保持 Graph 地址稳定。另一方案是预建多档 buffer/Graph，要求全 rank 同档、生命周期和 counter 隔离；当前 singleton 封装不能直接每步切换。

[详细设计与验收范围](humming-capacity32-17-19/ADAPTIVE_CAPACITY_NOTES.md)。

## 资料

- [三轮性能与 batch 核对](humming-capacity32-17-19/capacity-comparison.json)
- [请求、接受长度与 GPU handler 核对](humming-capacity32-17-19/request-audit.json)
- 逐 GPU Graph 拆分（原始归档内路径：`humming-capacity32-17-19/decomposition.json`）
- [容量 256 算子结果](humming-capacity32-17-19/probe-cap256.json)、[容量 32 算子结果](humming-capacity32-17-19/probe-cap32.json)
- [配置与源码版本记录](humming-capacity32-17-19/manifest.json)
- 完整测试证据归档（原始归档内路径：`humming-capacity32-17-19/capacity-evidence.tar.gz`）

远端目录：`/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/wangchangran/ts_master_h20/local_work/pd17-37-c32-128k-20261122/humming-capacity32-17-19`。运行入口 `start.py`、`run_bench.sh`；分析入口 `summarize_capacity.py`、`audit_capacity.py`、`decompose_humming.py`。测试脚本、启动参数、日志、原始请求事件、batch trace 和 Nsight 报告均保留。
