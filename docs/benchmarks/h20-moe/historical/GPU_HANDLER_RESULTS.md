> 历史记录：保留当时的配置、指标和状态；当前实现状态以分支代码为准。完整证据位置见本目录上一级 README。

# DeepEP GPU NIC handler 复测（2026-11-22）

GPU NIC handler 在 worker17 的 8 张 H20 上可用。两组完整服务测试均通过，但本次单机 EP8 没有观察到吞吐提升。独立禁用 NVLink P2P 的 RDMA 正确性检查也通过；这不是跨机 EP16 性能测试。

## 服务结果

P=worker37，D=worker17，D 为 DP8/EP8。全服务 C32，每 DP rank 4 请求；输入 131072 token，缓存命中 130816 token，输出 1024 token；CUDA Graph 开启，PDL 关闭。DSpark 固定接受长度 4，日志接受率约 60%，不是模型真实接受率。

表中为三轮有效满 batch 窗口指标的中位数。CPU 列是先前 cpu_host_memory handler 配置，GPU 列是本次配置；两者均允许 NVLink P2P。

| DeepEP 后端 | CPU TPOT ms | GPU TPOT ms | CPU 输出 token/s | GPU 输出 token/s | 吞吐变化 |
|---|---:|---:|---:|---:|---:|
| CUTLASS W4A16 | 8.9027 | 8.9457 | 3593.89 | 3581.21 | −0.35% |
| Humming W4A8 | 8.3371 | 8.3566 | 3834.09 | 3820.22 | −0.36% |

GPU handler 三轮：

| 后端 | 轮次 | TPOT ms | 输出 token/s | 满 batch 窗口 s |
|---|---|---:|---:|---:|
| CUTLASS W4A16 | 1 | 8.9457 | 3581.21 | 5.836 |
| CUTLASS W4A16 | 2 | 8.9510 | 3569.46 | 5.800 |
| CUTLASS W4A16 | 3 | 8.8675 | 3607.92 | 5.996 |
| Humming W4A8 | 1 | 8.3512 | 3831.78 | 5.913 |
| Humming W4A8 | 2 | 8.3833 | 3813.41 | 6.036 |
| Humming W4A8 | 3 | 8.3566 | 3820.22 | 5.815 |

统计使用所有请求共同 SSE 时间范围内最长连续满 C32 段，要求 rank batch 为 [4,4,4,4,4,4,4,4]、至少 30 个采样且至少 1 秒。TPOT 使用窗口内 token 增量，吞吐使用窗口内总输出增量；不是包含 TTFT 的整请求吞吐。三轮不是交错配对实验，约 0.4% 的差异不足以确认回退。

## 配置与证据

```bash
export NVSHMEM_IBGDA_NIC_HANDLER=gpu
unset NVSHMEM_IBGDA_FORCE_NIC_BUF_MEMTYPE
export NVSHMEM_DEBUG=INFO
```

两组服务日志各有 8 条 `NIC handler will be GPU.`。继续使用原虚拟环境与 deepep-compat，保持 DeepEP 所需的局部动态符号加载（unset TORCH_USE_RTLD_GLOBAL）。本次没有重新编译 DeepEP。

单机 EP8 的 DeepEP low-latency 默认允许 NVLink P2P。可直接访问的 peer 走 GPU copy 与通知；无法 P2P 访问时才走 IBGDA。因此 GPU handler 已启用并不意味着每次专家通信都经过 NIC，也不能把先前 DeepEP 的性能差异归因于 CPU handler。P→D KV 传输为独立 Mooncake 路径。

## 强制 RDMA 正确性验证

独立 probe 将 Buffer 的 `allow_nvlink_for_low_latency_mode=False`，创建后断言 `NVSHMEM_DISABLE_P2P=1`、handler=gpu。只修改 probe 的 Buffer 子类，没有修改服务运行代码。

- worker17 单机 8 GPU，经 mlx5_0:1；不是 17/37 跨机测试。
- 384/128 experts，hidden=5120，24 行输入，容量 256；使用可计算参考输出的专家缩放操作。
- 包含 eager、CUDA Graph、多次重放、部分 rank 空闲、全部空闲及恢复。
- 每 rank 30 项，共 240 项全部通过；进程退出码 0。
- 最大相对 L2 误差 0.001692；逐元素判据为 abs_error ≤ 0.001 + 0.01×abs(reference)。
- 该检查证明 RDMA 路径功能可用，不测完整模型精度，也不提供 RDMA 性能结论。

## 文件与清理

远端目录：`/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/wangchangran/ts_master_h20/local_work/pd17-37-c32-128k-20261122/gpu-handler`。

本地 `gpu-handler-comparison.json` 保存逐轮指标；`gpu-handler-results.tar.gz` 保存请求结果、batch trace、脚本、probe 日志和逐 rank 检查结果。原始 CPU handler 数据见同目录 `comparison.json` 与 `RESULTS.md`。

测试 D、P 和 gateway 已停止；worker17/37 各 8 GPU 均回到 1 MiB 显存占用。保留权重、AOT 与 autotune 缓存。
