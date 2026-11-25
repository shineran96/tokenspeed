> 历史记录：保留当时的配置、指标和状态；当前实现状态以分支代码为准。完整证据位置见本目录上一级 README。

# 原生 FlashInfer MegaMoE 性能拆分

2026-11-22，worker17 D / worker37 P。沿用此前 C32、DP8/EP8、128K 输入缓存复用、1024 输出、固定接受长度 4、CUDA Graph、PDL off 的配置。Target 使用原生 SM90 MXFP4 MegaMoE，draft 使用普通 EP CUTLASS W4A8。

## 实测结论

MegaMoE 核心仍占约 59% 的执行时间，不能把端到端只提升 4.1% 简单解释成“MoE 占比很小”。外围量化和 staging 有明确的融合空间；draft 尾段约占 5%，不是当前主导项。此次只有原生 MegaMoE 的 trace，没有同形状 CUTLASS 服务 trace，尚不能给出两者 MoE 核心的直接加速比。

GPU0 稳定阶段 39 个完整 Graph 的中位数：

| 时间段 | 每步 ms | 相对单步跨度 |
|---|---:|---:|
| Target 40 层 MegaMoE 核心 | 19.386 | 58.92% |
| 核心前量化、路由准备与 staging | 2.207 | 6.71% |
| draft anchor 至 Graph 结束 | 1.734 | 5.27% |
| 剩余时间，按上述时间段扣除 | 9.575 | 29.10% |
| Graph 首 kernel 至末 kernel 结束 | 32.902 | 100% |

这是执行时间段的拆分，不是所有算子累计时间的简单相加。共享专家等操作会与准备阶段重叠，因此准备阶段的全部时间不一定可转化为优化收益。“剩余时间”包含 target attention、投影、其他模型操作、采样及间隙，不应称为纯 attention 时间。Draft 尾段含自身投影、注意力、MoE 和相关收尾，不等于纯 draft MoE。

各卡稳定 Graph 中位数为 32.894–32.912 ms，MegaMoE 核心累计中位数为 19.312–19.406 ms；没有明显单卡长期偏慢的迹象。这不排除 kernel 内部逐层通信等待和路由不均。

## 准备阶段的具体成本

对每个 MegaMoE 核心，在同一 stream 上定位 routing kernel 后至核心开始前的操作，所有被分析的层均有：

- 14 个 GPU kernel，包括清零、BF16→FP32、abs/reduce、scale 计算、FP8 转换和路由准备。
- 5 次 D2D memcpy，包括一次路由权重填入以及 stage 的四次复制。
- 每步 40 层，共 560 个准备 kernel 和 200 次复制。

GPU0 每步准备 kernel 累计约 1.199 ms，D2D 复制累计约 0.194 ms，准备时间段合计约 2.207 ms，即每层约 55.2 μs。两者的差额含操作间隙、依赖和其他并行工作的影响，不能全称为 CPU launch 开销。CUDA Graph 已开启，这些 GPU 操作仍然存在。

建议把量化、padding 和路由写入合并，直接写入 `MegaMoESymmBuffer` 的 staging views，再调用已有 `commit_staged()`，替代当前临时张量加 `stage()`。必须保持 block-128 power-of-two scale、FP8 舍入、有效/空闲行和路由语义一致，并验证 eager/Graph 输出。

2.207 ms 是该时间段当前规模，不是预计可全部节省的时间。即使完全消除，在其他时间不变且无重叠影响的理想模型下，吞吐改善上限约 7.2%；实际还需要一个生产输入 kernel，收益会更小。

## 核心为何仍值得重点优化

Trace 确认当前每 rank target verify 为 24 行，即 4 请求×6 行；EP8 全局 192 行，384 experts，top-k=6，平均每专家 3 个 token 分配。真实分布并非均匀，平均值不能代替路由直方图。

这种小 M 下，融合不消除专家权重读取、MXFP4 解码以及跨 rank 同步。W4A16 与 W4A8 的权重都已经是 4-bit，降低 activation 精度并不再减少一半权重存储。因此不能按大 GEMM 的算力比例预测服务收益。当前尚未采集 HBM 带宽、Tensor Core 利用率或内部等待计数，是否主要受权重带宽、解码或同步限制仍待验证。

40 层核心每次平均约 485 μs。不同层平均约 433–590 μs，说明只看一个合成均匀路由的 layer benchmark 不够；后续应采集真实激活和路由，覆盖不同层及 C1/C2/C4。

Core 的 Nsight duration 包含通信和同步等待，不是纯 GEMM。若它本身能缩短 20%，按当前占比估算整步可减少约 11.8%，吞吐约增加 13.4%；这是敏感性分析，不是已实现收益。

## 对先前调度判断的修正

先前依据 `_H20_FUSED_BUCKETS` 中缺少 hidden=5120，判断本次使用 generic BM16 schedule，这一判断不适用于实际 MXFP4 路径。

真实调用为 `mxfp4_mega_moe → select_mxfp4_fused_plan`，与上述 NVFP4 配置表分开。当前实际 kernel 模板和代码确认：H=5120、I=2560、E=384、K=6、M=24、BM8、BN256、48 experts/wave、8 stages。MXFP4 selector 使用 `push_dispatch=False`、`no_clean_barrier=False`。这些参数可作为调优候选，但不能声称启用其他策略一定更快或已兼容。

## 采集方法与限制

使用 Nsight Systems 2025.3.2，trace=cuda,nvtx，cuda-graph-trace=node，CPU sampling/context switch 关闭。首轮 prime 和输出 warmup 后，rank0 检查全局 `[4]*8` decode batch，在稳定 12 个 step 后启动 profiler，再记录 40 个 step。各请求成功，缓存命中 130816/131072。

GPU0 捕获 40 个完整 Graph，其他 GPU 捕获 39 个；每个完整 Graph 有 40 次 target MegaMoE、一次 draft anchor。分析统一去掉各 GPU 第一个 Graph，得到 GPU0 39 个、其他 GPU 38 个稳态样本。

Profiler 启停日志相隔约 28.86 秒，但 GPU0 实际 trace 跨度只有 1.325 秒，大部分额外墙钟时间发生在采集初始化/边界阶段。稳态每步约 32.90 ms，接近此前不带 profiler 的约 32.25 ms（TPOT 8.063 ms×固定接受长度 4），但并非严格配对；继续以此前三轮结果作为服务性能成绩。

没有采集足够 NVTX 模型区间，时间段通过实际 kernel 名称、stream、Graph correlation ID 和源代码顺序定位。无法据此精确分解核心内部 GEMM/dispatch/combine，也尚不能定量归因 MegaMoE 相比 CUTLASS 的全部差异。

## 后续优先级

1. 融合 quantize/pad/route staging，验证数值和 Graph，再做同条件配对服务测试。
2. 采集真实 target 层输入/路由，与 CUTLASS 做完整 MoE 路径对照，区分外围和核心收益。
3. 针对 MXFP4 核心分析权重带宽、解码和同步，随后调优 tile、stages、expert wave；不套用 NVFP4 调度表。

## 原始材料

远端目录：`/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/wangchangran/ts_master_h20/local_work/pd17-37-c32-128k-20261122/profile-native`。

本地保存 `profile-native/native-trace.nsys-rep`、`profile-native/decomposition.json` 和 `native-kernel-summary.json`。远端另有 SQLite、逐请求结果、batch trace、采集 observer 和分析脚本。

首个 P 启动遗漏显式后端/PDL 参数，已停止并修正，相关日志单独保留为 `prefill-attempt-missing-explicit-flags.log`，正式请求发生在修正后。

本轮 P、D 和 gateway 均已停止，17/37 各 8 GPU 已回到 1 MiB 显存占用；权重和编译缓存保留。
