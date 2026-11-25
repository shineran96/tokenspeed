> 历史记录：保留当时的配置、指标和状态；当前实现状态以分支代码为准。完整证据位置见本目录上一级 README。

# DeepEP CUTLASS W4A16：模拟平均接受长度 5.61

2026-11-22。将用户的“接受率 5.61”按前序测试口径解释为模拟平均接受长度，并设置 `TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN=5.61`。这不是 5.61% 接受率，也不是真实模型接受率。

## 满 C32 结果

三轮有效连续满 batch 窗口的中位数：

- TPOT：6.1623 ms。
- 总输出吞吐：5204.11 token/s。
- 单 D GPU 吞吐：650.51 token/s。

| 轮次 | TPOT ms | 总输出 token/s | 满 batch 窗口 s |
|---|---:|---:|---:|
| formal1 | 6.1662 | 5179.63 | 3.114 |
| formal2 | 6.1623 | 5204.11 | 2.939 |
| formal3 | 6.1301 | 5221.03 | 3.026 |

三轮均验证连续 `[4,4,4,4,4,4,4,4]`，至少 1 秒和 30 个 batch 采样。96 个正式请求全部成功，缓存命中均为 130816/131072 tokens。

## 对照

| 配置 | 模拟接受长度 | TPOT ms | 输出 token/s |
|---|---:|---:|---:|
| DeepEP CUTLASS W4A16，GPU NIC handler | 4 | 8.9457 | 3581.21 |
| DeepEP CUTLASS W4A16，GPU NIC handler | 5.61 | 6.1623 | 5204.11 |
| 最新优化版原生 MegaMoE 混合配置 | 5.5 | 5.3415 | 5989.46 |

本次相对同后端、GPU handler 的长度 4 测试：吞吐增加 45.32%，TPOT 降低 31.11%。这些是前后独立三轮测试，不是交错 A/B。

单纯按每步产出比例，5.61/4 对应 +40.25% 吞吐；实测差异还包含执行时间/路由和运行波动等因素，不能把超过该比例的部分认定为额外 kernel 优化。此次没有修改 CUTLASS 或 DeepEP 实现。

MegaMoE 对照行采用 target W4A8 native + draft CUTLASS W4A8，且接受长度是 5.5；本次 target/draft 都是 DeepEP CUTLASS W4A16。因此它是所列配置的实测对照，不是相同精度、相同接受长度的纯后端加速比。

## 配置与接受长度核对

P=worker37，D=worker17；D DP8/EP8；C32 即每 DP rank 4 请求。输入 131072 tokens，输出 1024 tokens，ignore_eos；先 prime，再输出 warmup，再 formal1～3。CUDA Graph 开启、PDL 关闭。

D 参数：`--moe-backend flashinfer_cutlass --all2all-backend deepep --deepep-mode low_latency`。`moe_mxfp4_fp8_activation=False`，draft 继承相同后端。沿用 deepep-compat 虚拟环境覆盖和局部动态符号加载。`NVSHMEM_IBGDA_NIC_HANDLER=gpu`，8 个 rank 均有 `NIC handler will be GPU` 日志；单机 EP8 仍允许 NVLink P2P，这不表示所有通信都走 NIC。

P 保持 CUTLASS W4A16、普通 EP、prefix cache；PD 传输使用 Mooncake。

模拟器使用实时 cache length 在整数接受宽度之间切换。单步每 rank 4 请求的日志均值会随步变化，不能要求每条日志恰好等于 5.61。按请求 ID 核对的 96 个终态日志全部为 `acc_len=5.59, acc_rate=0.918`，整请求统计包含输出上限及首尾边界。结果见 `acceptance-audit.json`。

## 整请求指标

与满 batch 窗口口径分开，三轮中位数：整请求 TPOT P50 6.0589 ms，整轮输出窗口吞吐 4215.30 token/s，TTFT P50 2.1658 s。整轮输出窗口包括入场和收尾。

## 文件与状态

远端目录：`/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/wangchangran/ts_master_h20/local_work/pd17-37-c32-128k-20261122/deepep-accept561`。

本地 `deepep-accept561/comparison.json`、`acceptance-audit.json`、`results.tar.gz` 保存逐轮指标、请求匹配、原始输出、batch trace 和脚本。长度 4 对照来自 `gpu-handler-comparison.json`；MegaMoE 5.5 来自 `accept55/comparison.json`。

测试 P、D、gateway 已停止；权重、AOT、调优缓存保留。
