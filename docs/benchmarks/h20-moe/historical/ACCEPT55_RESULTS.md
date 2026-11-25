> 历史记录：保留当时的配置、指标和状态；当前实现状态以分支代码为准。完整证据位置见本目录上一级 README。

# 模拟平均接受长度 4 → 5.5 性能对比

2026-11-22。使用最新优化后的 native FlashInfer MegaMoE：fused staging + intermediate 2304，draft 仍为普通 EP CUTLASS W4A8。仅将 D 的 `TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN` 从 `4` 改为 `5.5`，没有修改模型或 kernel 代码。

## 满 batch 结果

每组为三轮有效满 C32 窗口的中位数。长度 4 使用上一轮相同优化版本的数据；两组是独立测试，不是交错 A/B。

| 指标 | 接受长度 4 | 接受长度 5.5 | 变化 |
|---|---:|---:|---:|
| 满 batch TPOT | 7.3399 ms | 5.3415 ms | −27.23% |
| 总输出吞吐 | 4353.53 token/s | 5989.46 token/s | +37.58% |
| 单 D GPU 输出吞吐 | 544.19 token/s | 748.68 token/s | +37.58% |

若 verify step 耗时不变，吞吐比例为 5.5/4=1.375。实测 +37.58% 与这一比例基本一致；TPOT×配置接受长度约为 29.36 ms 和 29.38 ms，也与单步耗时近似不变相符。这不是 MoE kernel 本身又加速了 37.58%，而是每步平均产出更多 token。

5.5 的逐轮结果：

| 轮次 | TPOT ms | 输出 token/s | 连续满 batch 窗口 s |
|---|---:|---:|---:|
| formal1 | 5.3415 | 5989.46 | 3.321 |
| formal2 | 5.3527 | 5975.48 | 2.562 |
| formal3 | 5.3351 | 5994.76 | 2.260 |

全部满足连续 `[4,4,4,4,4,4,4,4]`、至少 1 秒且至少 30 个 batch 采样。所有正式请求成功，缓存命中 130816/131072。

## 接受长度核对

代码原生支持小数模拟长度。它根据实时 cache length 计算下一步保留宽度，稳定阶段在整数 5/6 之间切换，使平均值为 5.5；verify 宽度仍为 6。

满 batch 日志显示 `avg_accept_len: 5.50, accept_rate: 0.90`。96 个正式请求的终态日志均报告 `acc_len=5.47, acc_rate=0.8941`，这是整请求的统计口径，包含首尾边界，不应与稳定 decode batch 的 5.50 混用。

以上接受长度和接受率均由模拟配置控制，不是模型真实接受率。验收数据见 `acceptance-audit.json`，按原始请求 ID 与 D 端完成日志逐一匹配，共 96 个唯一请求。

## 其余条件与整请求指标

P=worker37，D=worker17；D DP8/EP8；C32 为每 rank 4 请求；输入 131072 tokens、输出 1024 tokens、ignore_eos。预热相同 prefix，再进行输出 warmup 和三轮正式测试。CUDA Graph 开启，PDL 关闭，P 继续 CUTLASS W4A16，PD 使用 Mooncake。权重、后端、verify 形状和采样种子不变。

采用相同 SSE 和 batch trace 统计方法：所有请求共同输出时间范围内最长连续满 batch 段，以实际输出 token 增量计算 TPOT 和吞吐。

整请求统计与满 batch 口径分开：

| 指标 | 长度 4 | 长度 5.5 |
|---|---:|---:|
| 整请求 TPOT P50 | 7.2955 ms | 5.3025 ms |
| 整轮输出窗口吞吐 | 3579.98 token/s | 4642.35 token/s |
| TTFT P50 | 2.0974 s | 2.2040 s |

整轮输出窗口包含请求入场/收尾，TTFT 还受 P 和 PD 传输影响，因此不应期待这些指标也严格按 5.5/4 改善。

## 文件与清理

远端：`/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/wangchangran/ts_master_h20/local_work/pd17-37-c32-128k-20261122/accept55`。

本地 `accept55/comparison.json`、`accept55/acceptance-audit.json`、`accept55/results.tar.gz` 保留逐轮结果、接受率核对、原始请求、batch trace 和复现脚本。长度 4 对照数据在 `core-tuning/comparison.json`。

P、D 和 gateway 已停止；权重、AOT 和调优缓存保留。
