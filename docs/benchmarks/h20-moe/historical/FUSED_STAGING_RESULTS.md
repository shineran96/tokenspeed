> 历史记录：保留当时的配置、指标和状态；当前实现状态以分支代码为准。完整证据位置见本目录上一级 README。

# MegaMoE 融合量化与 staging：实现和验证

2026-11-22。已在本次隔离 TS 工作树接入融合准备 kernel，未修改共享主工作树、MegaMoE 核心或权重转换。

## 服务结果

同条件三轮有效满 C32 窗口的中位数：

| 配置 | TPOT ms | 输出 token/s | 单 D GPU token/s |
|---|---:|---:|---:|
| 原生 MegaMoE 原准备路径 | 8.0630 | 3968.63 | 496.08 |
| 原生 MegaMoE 融合准备路径 | 7.8560 | 4077.90 | 509.74 |
| 最初普通 EP CUTLASS W4A16 | 8.3897 | 3813.07 | 476.63 |

相对原版 MegaMoE，TPOT 降低约 2.57%，吞吐增加 2.75%；相对最初普通 EP CUTLASS，累计吞吐增加 6.95%。服务原版数据来自前序同配置三轮，不是此次交错 A/B 重启测试。

本次逐轮数据：

| 轮次 | 满 batch TPOT ms | 输出 token/s | 满 batch 窗口 s |
|---|---:|---:|---:|
| formal1 | 7.8560 | 4071.07 | 6.819 |
| formal2 | 7.7916 | 4107.07 | 4.768 |
| formal3 | 7.8589 | 4077.90 | 4.905 |

P=worker37，D=worker17；D DP8/EP8，C32 即每 rank 4 请求；131072 token 输入、130816 token 缓存命中、1024 输出；固定 DSpark 接受长度 4，CUDA Graph 开启、PDL 关闭。Target 原生 MegaMoE W4A8，draft 普通 EP CUTLASS W4A8。保持原有统计方法：所有请求共同 SSE 范围内最长连续 `[4]*8` 窗口，至少 1 秒、30 个采样。三轮均有效，不带 profiler。

## 实现

新增 `tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/native_staging_triton.py`，通过项目 `_triton` 接口使用 Triton。`flashinfer_native.py` 的 apply 调用它，并使用已有的 `commit_staged()`。

原路径每层有 14 个准备 kernel 和 5 次 D2D 复制；新路径用一次 kernel 直接完成：

- BF16 输入按 128 元素分块量化到 FP8 E4M3，计算 power-of-two FP32 scale。
- 写入 symmetric buffer 的 activation/scale views。
- 写入专家 ID 和路由权重，处理 odd top-k padding。
- 短/空闲 rank 写零 activation、最小 scale `2**-126`、ID -1 和零路由权重。

输入行数、stride 使用运行时参数并禁止按其数值特化；行地址计算使用 int64。未使用 torch.compile。没有改动 native MegaMoE 核心、MXFP4 权重打包和 EP finalize。

## 正确性

8 张 GPU，每 rank 44 个 staging 用例，共 352 个用例；每个用例均检查 eager 和 CUDA Graph 重放。覆盖 M=1/6/12/24、full/short/idle、K=6 和带 padding 的 K=3、非连续输入、随机数、零块及大范围 2 的幂边界。

FP8 字节、FP32 scale、expert IDs、route weights 均与原 PyTorch 路径逐位一致。真实 target layer0 权重的完整 MoE 输出也逐位一致，包含部分 rank 空闲、全部空闲、短 batch、恢复与 Graph 重放。相对既有 dense reference 的最大 rank 相对 L2 保持 0.00358018。这里的单层测试使用真实权重和合成 activation/router 输入，不冒充真实服务激活采集或模型任务精度评测。

完整服务 C1～C4 Graph 捕获、prime、warmup 和三轮 C32 请求均成功。

## 单层配对性能

EP8，每 rank 24 行，真实 target layer0 权重；CUDA Graph，7 轮交替测试顺序，每轮每配置 100 次 replay，CUDA Event 计时取各 rank 最大值，最终取 7 轮中位数：

| 路径 | 原版 μs | 融合版 μs |
|---|---:|---:|
| 完整 MoE | 475.682 | 451.352 |
| 仅准备阶段 | 24.692 | 4.308 |

完整 MoE 耗时降低 5.11%，纯核心约 450.206 μs。仅准备阶段计时包含 Graph replay 节奏，不能等同于 Nsight 单 kernel duration。该实验与完整服务的并行执行、路由和缓存环境不同，不能直接把节省的微秒数乘以层数作为服务收益。

## 代码与复现材料

远端实验目录：

```
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/wangchangran/ts_master_h20/local_work/pd17-37-c32-128k-20261122/fused-staging
```

实际接入位于上述目录的 `runtime` 链接，即实验根目录的隔离 runtime。`original_native.py` 保存优化前适配代码；`candidate_native.py` 和 `native_staging_triton.py` 保存优化版本。

本地 `fused-staging/native-staging.patch` 提供相对优化前适配器的变更；`benchmark.json`、`comparison.json`、`results.tar.gz` 保存配对测试、服务逐轮结果、逐请求数据、batch trace、正确性结果和复现脚本。`source-sha256.json` 保存源文件校验和。

隔离工作树的 MoE README 已更新。未提交 commit 或创建 PR。初次尝试的 `--allow-reload-forward` 被 TS 明确拒绝（不支持 PD 分离），随后移除此参数完整启动；失败启动未产生测试数据，日志单独保留。

本轮 P、D 和 gateway 已停止，权重、AOT 和调优缓存保留。
