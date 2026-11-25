> 历史记录：保留当时的配置、指标和状态；当前实现状态以分支代码为准。完整证据位置见本目录上一级 README。

# DSV4.1 Flash：H20 单机 DP8/EP8 MoE 后端对比

测试日期：2026-11-22。P 节点 worker37，D 节点 worker17，各使用 8 张 H20。

## 结论

在本次 C32、128K 输入缓存复用、1K 输出、固定接受长度 4 的负载下，原生 FlashInfer MegaMoE 混合配置最好，相比普通 EP CUTLASS W4A16，满 batch 输出吞吐提高 4.1%、TPOT 降低 3.9%。DeepEP Humming W4A8 与普通 EP CUTLASS 基本持平；DeepEP CUTLASS W4A16 反而略慢。

以下为每组 **三轮有效满 batch 样本的中位数**。吞吐单位均为输出 token/s，单卡指标只除以 8 张 D 卡，不包含 P 卡。

| D 后端 | 满 batch TPOT（ms） | 总输出吞吐 | 单 D 卡吞吐 | 吞吐相对普通 EP CUTLASS |
| --- | ---: | ---: | ---: | ---: |
| 普通 EP + FlashInfer CUTLASS W4A16 | 8.390 | 3813.1 | 476.6 | 基准 |
| DeepEP + FlashInfer CUTLASS W4A16 | 8.903 | 3593.9 | 449.2 | -5.75% |
| DeepEP + Humming W4A8 | 8.337 | 3834.1 | 479.3 | +0.55% |
| 原生 FlashInfer MegaMoE W4A8 target + CUTLASS W4A8 draft | **8.063** | **3968.6** | **496.1** | **+4.08%** |

第四组是用户确认的混合配置：target 使用 FlashInfer 原生单机 MegaMoE 及其单机通信，draft 使用普通 EP 的 FlashInfer CUTLASS W4A8。不是 GIN 多机 MegaMoE，也不使用 DeepEP。原因是 draft 权重的部分 scale 不能无损转换到原生 SM90 MegaMoE 所需表示；target 的 scale 检查通过。

W4A8 指 MXFP4 权重、FP8 激活，模型接口输出 BF16。TS 的 `humming` 与 `flashinfer_cutlass` 是不同后端入口，不能仅根据 W4A8 名称混为一组。

## 测试配置与验收

- 模型：DeepSeek-V4.1-Flash-1017，真实 target 和 DSpark draft 权重。
- P：TP8/QCP8/EP8，全程固定 FlashInfer CUTLASS W4A16，eager，prefix cache 开启。
- D：DP8/TP1/EP8，全服务并发 C32，即每 DP rank 4 请求；CUDA Graph 开启，实际捕获每 rank batch 1、2、3、4。
- 每请求输入 131072 tokens，输出 1024 tokens，ignore_eos；先用相同输入预热缓存，再执行一次输出 warmup 和正式轮次。
- DSpark 固定接受长度 4，所有组关闭 PDL，GPU memory utilization 0.85；PD 使用 Mooncake、mlx5_0。
- 各正式请求实际缓存命中均为 **130816/131072 tokens（99.8047%）**，每请求仍有 256 tokens 未命中。
- 每轮保存原始 SSE 时间戳、累计输出 token 数、返回元信息和同步 DP 调度记录。
- 满 batch 窗口：在全部 32 个请求共同 SSE 存活的范围内，选择实际调度记录中最长的连续 `[4,4,4,4,4,4,4,4]` 段，边界限定在已记录 step 的时间范围，至少 1 秒、30 个连续 step。选择只依据 batch，不依据性能。
- 窗口吞吐以 SSE 累计 token 增量除以窗口时长计算；窗口 TPOT 对每个请求使用窗口内首末事件的时间差除以 token 增量，再取 32 请求中位数，最终报告三轮中位数。
- 普通 EP 第一轮没有全部请求共同存活窗口，保留原始数据但不计为满 batch 样本，补测 formal4；其余组 formal1～3 均有效。没有挑选最快轮次。

## 逐轮满 batch 结果

| 后端 | 轮次 | 窗口（s） | TPOT（ms） | 总输出 token/s |
| --- | --- | ---: | ---: | ---: |
| 普通 EP CUTLASS | formal2 | 5.238 | 8.346 | 3825.0 |
| 普通 EP CUTLASS | formal3 | 5.166 | 8.390 | 3813.1 |
| 普通 EP CUTLASS | formal4 | 5.155 | 8.424 | 3791.0 |
| DeepEP CUTLASS | formal1 | 6.055 | 8.903 | 3593.9 |
| DeepEP CUTLASS | formal2 | 6.796 | 8.846 | 3622.7 |
| DeepEP CUTLASS | formal3 | 5.575 | 8.935 | 3581.7 |
| DeepEP Humming | formal1 | 5.767 | 8.346 | 3834.1 |
| DeepEP Humming | formal2 | 5.769 | 8.337 | 3833.0 |
| DeepEP Humming | formal3 | 6.035 | 8.335 | 3842.1 |
| MegaMoE 混合 | formal1 | 6.101 | 8.029 | 3983.4 |
| MegaMoE 混合 | formal2 | 4.999 | 8.063 | 3968.6 |
| MegaMoE 混合 | formal3 | 5.171 | 8.079 | 3959.2 |

这些是服务整体的实测差异，包含 attention、dense、draft 和调度等开销，不是 MoE 算子单独加速比。尚未做逐算子 profiling，因此不能把 DeepEP CUTLASS 的下降直接归因于通信本身。Humming 相对普通 EP 的差异很小，按基本持平解读。

## 整请求指标（包含进入、退出满 batch 的阶段）

下表采用同样三轮的中位数，与上面的满 batch 窗口口径分开。

| 后端 | 整请求 TPOT P50（ms） | 输出窗口吞吐（token/s） | TTFT P50（s） |
| --- | ---: | ---: | ---: |
| 普通 EP CUTLASS | 8.300 | 3219.9 | 2.100 |
| DeepEP CUTLASS | 8.802 | 3115.3 | 2.119 |
| DeepEP Humming | 8.292 | 3218.6 | 2.120 |
| MegaMoE 混合 | 8.009 | 3320.3 | 2.111 |

整请求 TPOT 沿用历史客户端定义 `(末输出事件时间−首输出事件时间)/(1024−1)`；DSpark 首个事件通常包含 5 tokens，因此它与使用实际 token 增量的满 batch TPOT 不能混用。输出窗口吞吐为一轮总输出除以最早首输出到最晚末输出的时间，包含请求入场和收尾。

## 与之前 1P1D 的比较

参考 2026-11-12 的 `pd_dp8_bench/1p1d_c1_c128/historical_results/RESULTS.json` 和 `BATCH_AND_CACHE_AUDIT.json`。

| C32 指标 | 历史 1P1D | 本次普通 EP CUTLASS |
| --- | ---: | ---: |
| 整请求 TPOT P50 | 7.957 ms | 8.300 ms |
| 输出窗口吞吐 | 2768.5 token/s | 3219.9 token/s |
| TTFT P50 | 3.056 s | 2.100 s |
| 实际全局 decode batch 峰值 | 28 | 32 |
| 满 C32 验证 | 0 个 step | 三轮连续窗口通过 |

数值上输出吞吐提高 16.3%，但不能解释成同负载后端加速：

- 历史是先预热 100Ki tokens，再执行十轮、每轮新增 10Ki，输入从 110Ki 增到 200Ki；本次固定 128Ki 输入，几乎全部命中缓存。
- 历史每轮要计算 10240 新输入 tokens，本次只剩 256；历史 C32 实际从未达到满 C32。
- 历史吞吐在十轮整体输出窗口上计算，包含轮间 prefill 等待；本次是独立轮次吞吐的中位数。
- 历史 P=worker42、D=worker11，本次 P=worker37、D=worker17；历史 PDL 未关闭，代码和环境也不是同一快照。

因此，本次满 batch 的 3813.1 token/s 没有对应的历史满 C32 数据可直接计算加速比。

## 接入、缓存与复现

实验根目录：

```
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/wangchangran/ts_master_h20/local_work/pd17-37-c32-128k-20261122
```

隔离代码在 `runtime/`，基线 commit 为 `83b1faec922470f7cca2a54ff0c911ec1f66929f`。未修改共享主工作树。原生适配入口：

```
runtime/tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/flashinfer_native.py
runtime/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/flashinfer_native.py
```

关键参数：

| 组 | moe-backend | all2all-backend | 附加参数 |
| --- | --- | --- | --- |
| 普通 EP CUTLASS | flashinfer_cutlass | none | 默认 W4A16 |
| DeepEP CUTLASS | flashinfer_cutlass | deepep | deepep-mode=low_latency |
| DeepEP Humming | humming | deepep | deepep-mode=low_latency，moe-mxfp4-fp8-activation |
| MegaMoE 混合 | flashinfer_native | none | draft-moe-backend=flashinfer_cutlass，moe-mxfp4-fp8-activation |

两机本地权重缓存均为 `/tmp/wcr-models/DeepSeek-V4.1-Flash-1017`，各 250 个文件、510311054842 bytes。源数据读取与本地重新读取的逐文件 SHA256 一致，两机清单一致。服务重启仍需重新分配、注册和装载 Engram 主机表。

本轮修复与验证：

1. TS Autotuner wrapper 透传新版 FlashInfer cache lookup 参数，真实 GPU 调优及缓存复用测试通过。
2. 隔离工作树复用现有 CUDA 编译产物；CUTLASS 复用已验证的 SM90 AOT。首次 autotuning 与 Graph 捕获不计入性能。
3. DeepEP 在 `TORCH_USE_RTLD_GLOBAL=1` 下出现 CUDA invalid resource handle，DeepEP 两组取消该设置。当前包和历史兼容包分别通过独立 8 卡 eager/Graph/idle/recovery 测试，每 rank 30 项检查无失败；完整服务测试通过。
4. 原生 MegaMoE target 的真实 layer 权重 8 rank 测试相对 L2 为 0.00358018，Graph 与 eager bit-exact；本轮完整模型 warmup、C1～C4 Graph 和 C32 服务测试通过。这不代替模型任务精度评测。

原始材料：`results/<backend>/<phase>/*.json`、`<backend>-batches.jsonl`、各服务日志、`comparison.json`、配置脚本和缓存清单均保留在实验根目录。完整精度与时间统计见 [comparison.json](comparison.json)。

测试结束后已停止本轮 P、D 和 gateway 服务；本地权重缓存、调优缓存及实验代码保留。
