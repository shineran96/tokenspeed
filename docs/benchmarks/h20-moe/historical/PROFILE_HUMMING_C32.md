> 历史记录：保留当时的配置、指标和状态；当前实现状态以分支代码为准。完整证据位置见本目录上一级 README。

# C32 DeepEP 与 Humming W4A8 性能分析

2026-11-23，worker17 decode，worker19 prefill。

## 结论

当前最明确的优化目标是 Humming FC1 与 FC2 之间的 SwiGLU/FP8 输入处理。它在每层按完整 padded 接收容量启动 98,304 个 CTA，40 层累计约 8.35 ms，占稳定 decode step 的 24.2%。这不是单纯的 DeepEP 网络问题，也不是 CPU launch 次数导致的开销：CUDA Graph 已开启，大量工作发生在一个中间处理 kernel 内。

GPU0 稳定 Graph 中位数为 34.455 ms，8 卡为 34.425–34.455 ms。Target MoE 路径时间段约 22.740 ms，占整步 66.0%；draft 尾段约 2.593 ms，占 7.5%。建议先验证接收容量收缩与只调度有效行的中间处理，再优化专家 GEMM，最后针对通信等待做细分。

本篇记录首次采集和分析。后续已完成容量 256→32 的算子验证、各三轮服务 A/B 与 profile：满载吞吐 +24.16%，TPOT −19.46%，详见[容量优化实测](HUMMING_CAPACITY_C32.md)。运行时计算源码未修改；容量 32 仅适用于实际输入与 Graph padding 不超过该上限的配置。

## 配置与样本

- D：17，8×H20-3e，DP8/EP8，TP1。P：19，4 卡，TP4/QCP4/EP4。
- DSV4.1，独立 `humming` 后端；target 与 draft 均继承该后端。不是 FlashInfer CUTLASS W4A8，也不是 CuTeDSL MegaMoE。
- DeepEP low-latency，FP8 dispatch、BF16 combine；GPU NIC handler，8 个 rank 均有确认日志。单机 EP8 允许 NVLink P2P，不应将 kernel 名称中的 internode 或 GPU handler 等同于本轮数据全部经过 NIC。
- 全局 C32，每个 DP rank 4 请求；target verify 每 rank 24 行。输入 131072 tokens，缓存命中 130816，输出 4096 tokens。
- 模拟接受长度 5.5；64 个 4K 请求（预热与 profile 各 32）终态均为 acc_len=5.50、acc_rate=0.8993，无 preemption。这不是自然模型接受率。
- CUDA Graph、PDL 开启。DeepEP `low_latency_max_num_tokens_per_gpu=256`。
- 使用原有虚拟环境，额外加载历史 Humming 源码与 `deepep-compat`；取消 `TORCH_USE_RTLD_GLOBAL`，沿用此前通过验证的局部符号加载方式。
- Humming HEAD：`d0c7af98d7f556132f86e671de7f397991e2df2b`。实际关键源码快照及 SHA256 在资料目录。

Nsight Systems 2025.3.2，trace=cuda,nvtx，cuda-graph-trace=node，关闭 CPU sampling/context switch。建立 prefix cache、完成输出预热后，rank0 在全局 `[4]*8` 条件下触发 cudaProfilerStart/Stop，采集约 40 个 step。

GPU0 有 40 个完整 Graph，其余各卡有 39 个；每卡去除第一个边界样本，分别分析 39/38 个 Graph。每个 target Graph 均验证 40 次 FC1、40 次中间处理、40 次 FC2、80 次 dispatch 和 80 次 combine；draft anchor 后单独统计。

## 整步与 MoE 拆分

下表以 GPU0 的逐 Graph 中位数为例。MoE 路径从每层 routing kernel 结束计至该层最后一个 DeepEP combine 结束，40 层累加。它包含局部间隙，并可能覆盖并行的 shared expert 工作，不能解释为严格排他的纯 routed MoE 运算时间。

| 时间段 | 每 step ms | 相对整步 |
|---|---:|---:|
| 完整 Graph 首 kernel 至末 kernel | 34.455 | 100% |
| Target 40 层 MoE 路径时间段 | 22.740 | 66.0% |
| Draft anchor 至 Graph 结束 | 2.593 | 7.5% |
| 按上述时间段扣除的剩余部分 | 9.121 | 26.5% |

Target MoE 相关 kernel 的累计 duration：

| 类别 | 每 step ms | 平均每层 μs | 相对整步 |
|---|---:|---:|---:|
| Humming FC1 | 5.729 | 143.2 | 16.6% |
| SwiGLU 与 FP8 中间处理 | 8.349 | 208.7 | 24.2% |
| Humming FC2 | 3.106 | 77.7 | 9.0% |
| DeepEP dispatch，含发送与接收阶段 | 0.920 | 23.0 | 2.7% |
| DeepEP combine，含发送与接收阶段 | 3.466 | 86.7 | 10.1% |
| 输入 scale 布局复制 | 0.127 | 3.2 | 0.4% |

上述类别各自取中位数，且不含所有间隙，不能要求其和精确等于路径时间段。PDL 或多 stream 下 kernel duration 也可能重叠；分析不能将全模型所有 kernel duration 相加后直接当作整步时间。

Target attention 主 kernel（sparse MLA 与 combine）累计约 1.769 ms，indexer/top-k 类约 0.678 ms。其他 dense GEMM 累计约 6.004 ms，包含 attention 投影、shared expert 等，不能全部归为 attention 或 shared expert。Draft 尾段包含 draft MoE、注意力、投影和采样收尾，不是纯 draft MoE 耗时。

## 中间处理为什么异常昂贵

实测 `process_input_kernel` 的网格为 `grid=(98304,1,1)`，block=(160,1,1)。全部已分析 target 层均使用该网格。

当前形状可对应为：

```text
本地专家数 = 384 / 8 = 48
每专家接收容量 = 每来源 rank 上限 256 × 8 ranks = 2048
padded 行数 = 48 × 2048 = 98304

实际全局 target 行数 = 8 × 24 = 192
全局专家路由数 = 192 × top-k 6 = 1152
平均每卡有效路由数 = 1152 / 8 = 144
平均每专家有效行数 = 1152 / 384 = 3
```

144 是跨卡平均值，不是每卡路由完全均匀的测量结果。容量与平均有效行数相差约 683 倍；这不是性能加速比，也不是说 GEMM 实际计算了全部 padding。

源码证据：

1. TS Humming wrapper 将接收张量展平，FC1 后把 padded 输出交给 `may_process_input(... layout="grouped_mask", expert_tokens=counts, activation_type="binary_split")`。
2. Humming `csrc/launcher/process_input.h` 在 grouped-mask 模式设置 `num_work_rows = num_output_rows = num_experts × max_tokens_per_expert`，再据此生成网格。
3. `context.cuh` 使用专家有效计数决定是否读取/输出某一行，但没有因此缩减 host 侧启动网格。`process_input.cuh` 对无效读取置零并继续经过相应处理逻辑，而非在网格层面只分配有效任务。
4. Nsight 在本轮观察到实际网格为 98,304 CTA，与完整容量一致；中间处理耗时在 8 卡上稳定为约 8.33–8.35 ms/step。

代表性层：FC1 约 153 μs，随后中间处理约 209 μs，再执行 FC2 约 84 μs。全样本中 FC1 与中间处理的最大时间交叠仅约 0.576 μs，不能用 PDL 重叠解释这段 200 μs 级时长。其浪费包含大量无效任务的调度和执行；缺少硬件计数器与消融时，不能把每一微秒都定量归因为某一种指令或访存。

## 通信与路由不均的判断

DeepEP dispatch + combine 在 GPU0 合计约 4.386 ms/step，显著小于中间处理加两次 GEMM的约 17.184 ms。当前不能优先将总瓶颈归为 NIC 带宽。

不同 rank 的专家 GEMM 与 combine 呈现补偿趋势。例如 GPU0 FC1+FC2 约 8.835 ms、combine 3.466 ms；GPU1 对应约 8.134 ms、4.094 ms。这与先完成计算的 rank 在 combine 中等待较慢 peer 的情况相符，但没有通信内部时间戳与路由计数分布，不能据此给出等待比例。

8 卡整步中位数仅相差约 0.030 ms，不存在明显单卡长期落后的整步现象；这不排除各层的路由倾斜和同步等待。通信 kernel 时长包含等待、同步和合并计算，不是纯网络传输时间。

## 优化顺序

### 1 验证接收容量能否由 256 收缩到 32

对本次固定 C32/DP8/每请求 6 行的满 batch，target 每 rank 24 行，32 是候选上界。若所有服务阶段与 Graph bucket 的实际 token 上界均不超过 32，每专家容量可从 2048 降至 256，完整行数从 98304 降至 12288。

这只是网格规模预计缩小 8 倍，不能声称 kernel 或服务一定加速 8 倍。需要验证 warmup、draft、batch 不均、Graph padding 和异常请求入场的最大行数，不能将针对 C32 的参数直接推广到更高并发。应先做同条件容量 A/B，并保留数值和空闲恢复检查。

### 2 让中间处理按有效行分配工作

可采用每专家读取 GPU counts 后循环，或 device 侧紧凑任务队列，避免每个 padded 行启动一个 CTA。保持固定地址、设备端计数与 Graph 兼容，不将 counts 拷回 CPU 决定网格。

若采用提前退出，必须保证 CTA/warp barrier 参与一致、PDL 依赖已满足，并检查 FC2 对 padding 的读取及零值契约，防止连续 replay 读取旧数据。随后再考虑把 SwiGLU 与中间量化融合进 FC1 epilogue。

中间处理当前占整步约 24.2%。假设这一项减少一半且其他时间不变，整步理论缩短约 12.1%，吞吐理论增加约 13.8%；这是敏感性估算，不是已实现收益。

### 3 再优化小 M 专家 GEMM与通信重叠

每专家平均 3 行，当前 FC1/FC2 合计约 8.835 ms。应结合真实 counts 和层间差异验证 tile、调度及权重读取效率，不能按大矩阵 FP8 峰值预测收益。

通信优化应先区分 combine 的传输、归并和等待，再决定是否调整调度或与 shared expert 重叠。本轮数据不足以判断 HBM 带宽、Tensor Core 利用率或 NIC 是否达到瓶颈；需要针对少量代表性 kernel 的硬件计数器采集。

## 服务指标与 profiler 扰动

缓存 prime 32 个请求全部成功；随后 warmup 和 profile 各 32 个 4K 请求全部成功，缓存命中均为 130816，无 preemption。

未启用采集的输出预热轮，整请求 TPOT P50 为 6.107 ms；连续满 `[4]*8` 窗口有 643 个 batch 样本、持续约 21.59 秒，host step 间隔中位数为 33.637 ms。它仅作为运行状态旁证，不是独立三轮正式性能成绩。

Profile 轮整请求 TPOT P50 为 12.318 ms，受到 profiler 启停扰动，不能用于与既有服务结果比较。Start/Stop 日志相隔约 23.39 秒，而完整 GPU Graph 的稳定跨度约 34.45 ms/step。这里报告的瓶颈来自 trace 内完整 Graph，而非启停日志墙钟跨度。

## 复现与资料

- [运行配置和关键源码 SHA256](profile-humming17-19/manifest.json)
- 逐 GPU 逐 Graph 拆分（原始归档内路径：`profile-humming17-19/decomposition.json`）
- [中间处理 kernel 网格](profile-humming17-19/activation-kernel-geometry.json)
- [单 Graph 时间线](profile-humming17-19/example-graph.json)
- [请求与接受长度核对](profile-humming17-19/request-audit.json)
- Nsight 报告（原始归档内路径：`profile-humming17-19/humming-trace.nsys-rep`）
- 完整数据归档（原始归档内路径：`profile-humming17-19/profile-evidence.tar.gz`）
- Humming 关键源码快照（原始归档内路径：`profile-humming17-19/source-evidence.tar.gz`）
- 分析脚本（原始归档内路径：`profile-humming17-19/decompose_humming.py`）

远端目录：`/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/wangchangran/ts_master_h20/local_work/pd17-37-c32-128k-20261122/profile-humming17-19`。

复现 decode 启动时必须在 `common.sh` 前设置 `TS_BENCH_EXTRA_PATH`，包含 `local_work/h20-humming-deepep-20261115/humming-src` 与同目录 `deepep-compat` 的绝对路径。启动参数为 `--moe-backend humming --all2all-backend deepep --deepep-mode low_latency --moe-mxfp4-fp8-activation --disaggregation-ib-device mlx5_0`。仅设置 W4A8 flag 并切到 `flashinfer_cutlass` 不会复现本测试。

P、D、gateway 已停止，17 的 8 卡、19 的 4 卡均确认恢复为 1 MiB 显存占用。权重与编译缓存保留。
