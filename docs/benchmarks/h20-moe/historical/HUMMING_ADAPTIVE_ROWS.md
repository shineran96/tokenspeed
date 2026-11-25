> 历史记录：保留当时的配置、指标和状态；当前实现状态以分支代码为准。完整证据位置见本目录上一级 README。

# Humming 按有效行自适应调度

## 实现

通信容量保持 256，修改 Humming `process_input_kernel` 的 grouped-mask 调度。每个专家分配最多 4 个行 worker（按列分片时再乘 column tasks），worker 在 GPU 上读取有效行计数并循环处理。输入、输出与 scale 的 padded 地址保持不变，不把 counts 拷回 CPU，不按并发重建 Graph。

PDL 等待发生在读取 counts 之前；完成全部循环后才通知依赖者，零行 worker 也执行完成通知。循环间同步保护共享 scratch 重用。partial token block 被限制在当前专家的容量内，不能跨入下一专家。所有行地址使用 64 位计算。

`zero_invalid=True` 保留原 padded-row launch，以维持无效行清零契约；独立 group/token scale finalizer 保留原布局。该优化减少有效计算调度开销，不扩展通信容量，也不自动缩小 buffer。服务最大输入与 CUDA Graph padding 仍须落在容量以内。

代码库：17 上 `ts_master_h20/local_work/humming-adaptive`，Humming 分支 `perf/grouped-mask-active-rows`，基于 `d0c7af98d7f556132f86e671de7f397991e2df2b`。原 Humming checkout 保持不变。测试服务通过 PYTHONPATH 显式使用新 checkout。

## 验证范围

- 现有布局回归：normal/grouped-mask/scatter、activation、量化模式与无效行清零。
- 新增动态 Graph 测试：容量 7/2048、int32/int64 counts、PDL on/off；同一 Graph 连续改变输入和 counts，包含集中路由、全 idle、恢复和多个并发对应行数。
- 新增测试对有效行使用现有参考实现校验，Graph/eager 要求逐位一致，并检查无效输出与 scale sentinel 未被覆盖。
- 真实 DSV4.1 权重 EP8：target 每 rank 6/12/24/48/96 行，draft 每 rank 1/2/4/8/16 行。另检查集中路由、奇数 rank idle、全 idle、输入变更与恢复。真实 Humming MoE 自身并非逐位确定，跨版本误差需与原版重复运行误差并列记录；这不是模型质量评测。
- C32 服务：17 D8 + 19 P4，128K 输入、4K 输出、模拟接受长度 5.5，PDL/Graph on，容量 256，三轮正式输出与单独 profile。

最终版本完整布局测试：**90 passed**。CUDA Graph 有效行参考校验、Graph/eager 逐位比较、无效行 sentinel 检查均通过。C++ 使用仓库 clang-format 配置格式化修改行，Python Ruff format/check 通过，git diff --check 通过。

## 最终版本 EP8 算子测试

真实 target/draft 第 0 层权重，通信容量均为 256，PDL/Graph 开启。每形状 7 个计时窗口，每窗口 100 次 replay；每卡取窗口中位数，再取最慢 rank。原版和新版分别在独立进程运行，没有服务并发干扰。以下为正常路由下 dispatch、专家计算和 combine 总耗时。

| 每 DP rank 并发 | Target 行数 | 原版 Target μs | 自适应 Target μs | 耗时下降 | 原版 Draft μs | 自适应 Draft μs |
|---:|---:|---:|---:|---:|---:|---:|
| C1 | 6 | 402.71 | 211.31 | 47.5% | 342.72 | 279.27 |
| C2 | 12 | 451.70 | 263.16 | 41.7% | 337.09 | 274.72 |
| C4 | 24 | 547.83 | 373.02 | 31.9% | 362.35 | 299.63 |
| C8 | 48 | 703.58 | 543.75 | 22.7% | 378.17 | 315.91 |
| C16 | 96 | 828.98 | 664.40 | 19.9% | 409.12 | 350.75 |

Draft 行数等于每 rank 并发。集中路由、部分 idle、全 idle、输入变化与恢复场景均运行完成且输出有限。跨版本输出 rel L2 最大值：target 各形状 0.00332–0.00631、draft 0.01008–0.01556；原版自身重复运行最大误差在相同量级。不能把这些数值宣称为 bitwise 等价或完整模型质量无损。

当前验证聚焦 decode。固定 worker 调度在大 M prefill 或少专家高密度场景下的性能尚未评估，不应将本表收益推广到所有 grouped-mask 负载。

## 端到端结果与提交

已提交 Humming commit `219fc5061738dd5a0f7bfc92d3d3c9d36cd773ae`，分支 `perf/grouped-mask-active-rows`，尚未 push。包含调度实现、动态 Graph 测试与配置说明，共五个文件；没有修改 TokenSpeed 的运行时源码或默认容量。

同为全局 C32、128K 输入、4K 输出、模拟接受长度 5.5、PDL/Graph 开启。采用请求共同输出时间段中间 80% 的满载窗口，三轮平均；profile 不计入服务成绩。

| 方案 | 满载 TPOT ms | 输出 token/s |
|---|---:|---:|
| 原版 Humming，容量 256 | 6.116 | 5232.0 |
| 固定小容量 32 | 4.926 | 6496.0 |
| 自适应有效行，容量 256 | 5.000 | 6399.7 |

自适应版本相对原容量 256：吞吐 **+22.32%**，TPOT **−18.25%**。相对固定容量 32 的本次 C32 专用配置，吞吐低 1.48%，但保留容量 256 并已验证不同并发的有效行调度。不同版本顺序执行，不是随机交替 A/B。

三轮满载 TPOT 分别为 4.9828/5.0098/5.0083 ms，吞吐 6422.1/6387.5/6389.4 token/s；窗口内 480/486/474 个 batch 样本全部为 `[4]*8`。正式 96 请求全部成功；连同预热与 profile 的 160 个 4K 请求，缓存命中均为 130816，接受长度均为 5.5、接受率 0.8993，无 preemption，8 rank 的 GPU NIC handler 均确认。

## Profile

GPU0 去掉首个 Graph 后取 39 个样本中位数：整步 27.743 ms，target MoE 路径 16.099 ms，中间 SwiGLU/FP8 累计 0.985 ms。原版对应为 34.455、22.740、8.349 ms。中间处理时间降低约 88.2%。

原版中间 kernel 为 98304 CTA；新版本为 **192 CTA（48 专家 × 4 worker）**，每 CTA 160 threads，按设备 counts 循环。CTA 数减少不等于时间按相同比例下降。

相对小容量 32，新版 target 中间处理更短（0.985 vs 2.040 ms），但 GPU0 combine 累计更长（4.104 vs 3.464 ms），draft 尾段也更长（2.527 vs 1.968 ms），因此整服务略慢。通信 kernel 包含等待，缺少内部计数时不能将其全归因于网络传输。不同路径和累计 duration 可能重叠，不能简单相加。

17 和 19 的测试服务均已停止，GPU 已释放；权重与编译缓存保留。

## 证据

- [提交补丁](humming-adaptive17-19/humming-adaptive-219fc50.patch)
- [三轮服务结果](humming-adaptive17-19/adaptive-comparison.json)
- [请求核对](humming-adaptive17-19/request-audit.json)
- 逐 GPU Graph 分析（原始归档内路径：`humming-adaptive17-19/decomposition.json`）
- [测试日志](humming-adaptive17-19/layout-tests.log)
- [运行版本与源码哈希](humming-adaptive17-19/manifest.json)
- 完整运行证据（原始归档内路径：`humming-adaptive17-19/adaptive-evidence.tar.gz`）

远端证据目录：`/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/wangchangran/ts_master_h20/local_work/pd17-37-c32-128k-20261122/humming-adaptive17-19`。
