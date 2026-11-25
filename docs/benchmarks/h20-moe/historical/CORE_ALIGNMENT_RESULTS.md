> 历史记录：保留当时的配置、指标和状态；当前实现状态以分支代码为准。完整证据位置见本目录上一级 README。

# MegaMoE 第二轮优化：移除不必要的 intermediate padding

2026-11-22。已在隔离 TS runtime 接入并完成三轮服务测试。保留上一轮 fused staging，核心调度保持默认。

## 服务收益

同条件三轮有效满 C32 窗口中位数：

| 版本 | TPOT ms | 输出 token/s | 单 D GPU token/s |
|---|---:|---:|---:|
| 普通 EP CUTLASS W4A16 | 8.3897 | 3813.07 | 476.63 |
| 最初原生 MegaMoE 混合配置 | 8.0630 | 3968.63 | 496.08 |
| MegaMoE + fused staging | 7.8560 | 4077.90 | 509.74 |
| MegaMoE + fused staging + 正确对齐 | **7.3399** | **4353.53** | **544.19** |

本次增量：TPOT 降低 6.57%，吞吐增加 6.76%。相对最初 MegaMoE，累计吞吐增加 9.70%；相对普通 EP CUTLASS，增加 14.17%。这些是同配置的前后独立三轮测试，不是交错重启 A/B 实验。

本次逐轮：

| 轮次 | TPOT ms | 输出 token/s | 连续满 batch 窗口 s |
|---|---:|---:|---:|
| formal1 | 7.3306 | 4362.58 | 4.368 |
| formal2 | 7.3441 | 4348.63 | 4.494 |
| formal3 | 7.3399 | 4353.53 | 4.404 |

P=worker37，D=worker17，D 为 DP8/EP8；全服务 C32，每 rank 4 请求。输入 131072、缓存命中 130816、输出 1024 tokens；固定 DSpark 接受长度 4，CUDA Graph 开启，PDL 关闭。Target 原生 FlashInfer SM90 MegaMoE W4A8；draft 普通 EP CUTLASS W4A8。三轮均满足连续 `[4]*8`、至少 1 秒和 30 个 batch 采样；继续沿用原始 SSE/token 增量统计方法，不带 profiler。

## 原因与修改

DSV4.1 checkpoint 的专家 intermediate_size 是 2304。模型 `DeepseekV41MoE` 的公共 MegaMoE 分支原先默认按 512 对齐，因此 native SM90 路径也被补到了 2560。原生 SM90 MXFP4 的 registry trait 和 FlashInfer selector 实际只要求 intermediate 按 128 对齐；2304 已满足。

修改 `python/tokenspeed/runtime/models/deepseek_v41.py` 的构造期对齐选择，让 `flashinfer_native` 在 FP4 权重下使用 128，与它自己的 kernel 契约一致。其他后端保持原分支。这样不再为 target 专家多分配、搬运和计算额外的零通道；没有改动有效权重、activation 精度、路由、核心 schedule 或接受长度。

8 个服务 rank 的 `/proc/<pid>/maps` 均核对到 `i2304` 原生 kernel 模块，没有加载 `i2560` 的 native 实现。记录见 `loaded-native-modules.json`。

## 数值和单层对照

使用真实 target 第 0、20、39 层权重。为 2560 版本按 gate/up 分别补零并给第二矩阵补零列，与 2304 原始版本进行 EP8 对照。输入是合成 activation 经真实 router 权重生成路由，不是采集的服务 activation。

每 rank 6/12/24 行，9 组均通过 eager 输出逐位一致、各自 Graph/eager 逐位一致；另外比较部分 rank 空闲、全部空闲、短 batch、恢复场景，均逐位一致。服务 C1～C4 Graph 捕获及 prime/warmup/formal1～3 均成功。这不替代模型任务精度评测。

单层完整 MoE 的 CUDA Graph 配对性能，单位 μs。5 轮交替顺序，每轮 60 次 replay，CUDA Event 计时，取最大 rank 和轮次中位数：

| 层 | 每 rank 行数 | 2560 padded | 2304 native | 耗时降低 |
|---|---:|---:|---:|---:|
| 0 | 6 | 285.21 | 251.36 | 11.87% |
| 0 | 12 | 358.19 | 312.22 | 12.83% |
| 0 | 24 | 503.98 | 451.64 | 10.39% |
| 20 | 6 | 331.85 | 298.26 | 10.12% |
| 20 | 12 | 393.20 | 346.07 | 11.99% |
| 20 | 24 | 477.13 | 435.34 | 8.76% |
| 39 | 6 | 250.61 | 242.23 | 3.34% |
| 39 | 12 | 321.96 | 286.89 | 10.89% |
| 39 | 24 | 444.31 | 404.59 | 8.94% |

## 调度扫描结论

同时测试了 M=24、I=2304 的默认配置、stages=4/6、experts_per_wave=24/16，以及关闭 interleaving。覆盖真实 router 权重生成的路由、均匀路由和集中到少数专家的压力路由，结果均与默认逐位一致。

降低 stages 或改变 wave 没有稳定收益，不接入这些候选。关闭 interleaving 在 router 场景从约 452 μs 增至 476 μs，其他两类路由也变慢。最终保留默认 BM8、BN256、48 experts/wave、8 stages 和 interleaving（M=24）。具体逐轮结果见 `tuning.json`。

下一步若继续优化，应先采集真实服务路由和核心内部阶段时间，区分权重解码、GEMM、dispatch/combine 和同步等待，再针对性改核心；本轮小范围 knob 扫描没有证明进一步收益。

## 文件与状态

远端实验目录：

```
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/wangchangran/ts_master_h20/local_work/pd17-37-c32-128k-20261122/core-tuning
```

`runtime` 链接指向前序隔离工作树。`deepseek_v41-before.py`、`deepseek_v41-after.py` 保留修改前后源码。MoE README 已更新，未提交 commit 或创建 PR。

本地 `core-tuning/alignment.patch` 为本轮代码补丁；`comparison.json`、`padding.json`、`tuning.json` 保存服务/对齐/调度结果；`results.tar.gz` 包含原始请求、batch trace、脚本、模块审计与源码快照。

测试 P、D 和 gateway 已停止，权重、AOT、调优及新 kernel 编译缓存保留。
