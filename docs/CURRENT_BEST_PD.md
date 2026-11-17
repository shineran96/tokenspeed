# 当前已测最佳 PD 参数：RTX 6000D 单/双 P / DP16 Decode

本页保存 2026-11-10 实际运行配置，适用于 DeepSeek-V4.1-Flash-1017：
P 为单机 8 卡 QCP8/EP8；D 为两机共 16 卡 DP16/EP16；SMG 协调、Mooncake 直传 KV。
“最佳”仅指当前这套硬件和 100K prime、10 轮递增前缀、每轮 1K 输出的已测配置，
不是所有模型、网络和精度场景的通用最优值。本页验证单 P C16/C32/C64 和双 P C16/C32/C64/C128，不外推更高并发容量或模型精度。

## 源码、镜像与组件

- P/D runtime：`d620ec2abf7ec9f7602e56a4b9612589decb76fc`，本次核验时为 `feature/6000d-dsv41/main` 分支头。
- 使用 [IMAGE.md](IMAGE.md) 的镜像环境；镜像 TS manifest 为 `9cd447eea08140de08395ad095ceb9a53277df03`。
  runtime 通过当前 checkout 的 `python` 加载；镜像内编译依赖不会因 checkout 更新自动重建。
- Python 3.12 / Torch 2.14.0+cu130 / CUDA 13 / FlashInfer 0.7.2；依赖安装见
  [DEPLOYMENT.md](DEPLOYMENT.md)、[DEEPEP_NVFP4_FP8.md](DEEPEP_NVFP4_FP8.md)。
- P 使用 Elastic FP8 dispatch / BF16 combine；D 使用单独构建的 NVFP4 dispatch / FP8 combine wire1。
- 本轮使用 SMG，不使用 Dynamo。SMG 已验证修复版动态库 SHA256：
  `8d3580a4e86ca5b95f05bf73f21ff5b28080b0353b21931dd68a706cebac41ac`。
  镜像公开 wheel 的 SHA256 `8f3b9b997e80d2f2b3ebbb06659a30b8eae5291e3782589de973518410379e0d`
  不是本轮均衡修复版；相同 `tokenspeed-smg` 版本号不能证明代码相同。
  从仓库固定 Gateway submodule 构建，按真实 rank 分布重新验收；不能声称新构建必然与上述旧制品逐字节一致。

## 可迁移目录与地址

下列变量均由部署者提供，不引用个人共享目录：`REPO_ROOT`、`MODEL_PATH`、`VENV`、
`GATEWAY_VENV`、`P_STATE_DIR`、`D_STATE_DIR`、`CUDA_HOME`、`COMPILER_ROOT`、
`DEEPEP_WIRE1_SOURCE`、`NVSHMEM_DIR`、`NCCL_ROOT_DIR`、`P_HOST`、`GATEWAY_D0_HOST`、`D1_HOST`。
本次 Gateway 与 D0 同机；若分开部署，拆分两者地址。
`NVSHMEM_DIR`/`NCCL_ROOT_DIR` 指向实际安装包位置，不能只照抄路径。
缓存目录需本机可写、按角色隔离。共享环境表记录的是已解析值；路径占位符并非可原样执行的环境文件。
所有网卡名、GID、SL、线程数必须按目标拓扑检查。实测 NIC 顺序保留用于复现，不是通用自动映射。

## 启动参数

P 的关键差异是 `P_INDEXER_SOLUTION=native_fp4`，它对应下方两个原生环境选项；
仅 `sm120` 会选择另一条打分路径。本轮 `max_num_seqs=32`、chunk **16640**，不是 16384。
D 的 `max_num_seqs=256` 与 local CUDA Graph 上限 16 分开；capture BS 为 1/2/4/8/16。
D 未显式启用 LM-head TP4，不应将历史 LM-head TP4 实验混入本配置。
`--all2all-backend none` 是 P 的实际 CLI；SM120 DeepEP 由显式实验开关选择，不能据此认定没有 EP 通信。

### P

```bash
"${VENV}/bin/python" -m tokenspeed.cli serve \
  --headless \
  --model "${MODEL_PATH}" \
  --served-model-name deepseek-v41-flash \
  --data-parallel-size 1 \
  --disaggregation-mode prefill \
  --data-parallel-address "${GATEWAY_D0_HOST}" \
  --data-parallel-rpc-port 18510 \
  --zmq-engine-index 0 \
  --dense-gemm-backend deepgemm \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --query-context-parallel-size 8 \
  --kv-parallel-size 1 \
  --dense-tp-size 1 \
  --moe-tp-size 1 \
  --attn-head-tp-size 1 \
  --ep-size 8 \
  --dtype bfloat16 \
  --moe-backend flashinfer \
  --moe-mxfp4-fp8-activation \
  --all2all-backend none \
  --init-expert-location trivial \
  --max-model-len 1048576 \
  --chunked-prefill-size 16640 \
  --max-prefill-tokens 16640 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.85 \
  --engram-host-table \
  --engram-host-table-layout shared \
  --disable-prefill-graph \
  --disable-kvstore \
  --disable-pdl \
  --force-deterministic-rsag \
  --speculative-algorithm DSPARK \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-layerwise-interval 0 \
  --host "${P_HOST}" \
  --port 18501 \
  --dist-init-addr "${P_HOST}:19400" \
  --disable-autotune \
  --enable-prefix-caching \
  --enable-log-request-stats \
  --decode-log-interval 1 \
  --enable-metrics \
  --metrics-port 18521 \
  --disaggregation-bootstrap-port 18698
```

### D0

```bash
"${VENV}/bin/python" -m tokenspeed.cli serve \
  --headless \
  --model "${MODEL_PATH}" \
  --served-model-name deepseek-v41-flash \
  --data-parallel-size 16 \
  --disaggregation-mode decode \
  --data-parallel-address "${GATEWAY_D0_HOST}" \
  --data-parallel-rpc-port 18511 \
  --zmq-engine-index 0 \
  --dense-gemm-backend deepgemm \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --world-size 16 \
  --ep-size 16 \
  --nnodes 2 \
  --node-rank 0 \
  --moe-backend flashinfer \
  --moe-mxfp4-fp8-activation \
  --dtype bfloat16 \
  --seed 968340402 \
  --max-model-len 1048576 \
  --max-num-seqs 256 \
  --chunked-prefill-size 8192 \
  --decode-log-interval 1 \
  --max-cudagraph-capture-size 16 \
  --cudagraph-capture-sizes 1 2 4 8 16 \
  --speculative-algorithm DSPARK \
  --disable-prefill-graph \
  --disable-kvstore \
  --engram-host-table \
  --engram-host-table-layout shared \
  --disable-pdl \
  --force-deterministic-rsag \
  --gpu-memory-utilization 0.85 \
  --disable-weight-loader-prefetch-checkpoints \
  --attention-backend deepseek_v41 \
  --comm-fusion-max-num-tokens 4096 \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-layerwise-interval 0 \
  --disable-prefix-caching \
  --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 \
  --host "${GATEWAY_D0_HOST}" \
  --port 18347 \
  --dist-init-addr "${GATEWAY_D0_HOST}:13880"
```

### D1

```bash
"${VENV}/bin/python" -m tokenspeed.cli serve \
  --headless \
  --model "${MODEL_PATH}" \
  --served-model-name deepseek-v41-flash \
  --data-parallel-size 16 \
  --disaggregation-mode decode \
  --data-parallel-address "${GATEWAY_D0_HOST}" \
  --data-parallel-rpc-port 18511 \
  --zmq-engine-index 0 \
  --dense-gemm-backend deepgemm \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --world-size 16 \
  --ep-size 16 \
  --nnodes 2 \
  --node-rank 1 \
  --moe-backend flashinfer \
  --moe-mxfp4-fp8-activation \
  --dtype bfloat16 \
  --seed 968340402 \
  --max-model-len 1048576 \
  --max-num-seqs 256 \
  --chunked-prefill-size 8192 \
  --decode-log-interval 1 \
  --max-cudagraph-capture-size 16 \
  --cudagraph-capture-sizes 1 2 4 8 16 \
  --speculative-algorithm DSPARK \
  --disable-prefill-graph \
  --disable-kvstore \
  --engram-host-table \
  --engram-host-table-layout shared \
  --disable-pdl \
  --force-deterministic-rsag \
  --gpu-memory-utilization 0.85 \
  --disable-weight-loader-prefetch-checkpoints \
  --attention-backend deepseek_v41 \
  --comm-fusion-max-num-tokens 4096 \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-layerwise-interval 0 \
  --disable-prefix-caching \
  --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 \
  --host "${D1_HOST}" \
  --port 18347 \
  --dist-init-addr "${GATEWAY_D0_HOST}:13880"
```

### SMG

```bash
"${GATEWAY_VENV}/bin/python" -m smg launch \
  --pd-disaggregation \
  --backend tokenspeed \
  --prefill "tcp://${GATEWAY_D0_HOST}:18510" \
  --prefill "tcp://${GATEWAY_D0_HOST}:18512" \
  --decode "tcp://${GATEWAY_D0_HOST}:18511" \
  --model-path "${MODEL_PATH}" \
  --tokenizer-path "${MODEL_PATH}" \
  --model-alias "deepseek-v41-flash=${MODEL_PATH}" \
  --host "${GATEWAY_D0_HOST}" \
  --port 18500 \
  --prefill-policy consistent_hashing \
  --decode-policy round_robin \
  --max-concurrent-requests 256 \
  --queue-size 512 \
  --queue-timeout-secs 7200 \
  --request-timeout-secs 7200 \
  --disable-retries \
  --disable-circuit-breaker \
  --prometheus-port 18520 \
  --worker-stall-secs 1800 \
  --worker-wedge-secs 1800 \
  --worker-stale-secs 1800 \
  --max-payload-size 33554432 \
  --max-buffered-request-bytes 33554432 \
  --disable-worker-overload-protection
```

实际 SMG 注册了两个 P 地址，但本轮单 P 客户端明确选中 P08。复现单 P 时仅注册一个 P，
或先通过请求 ID 与 P 日志验证索引映射再显式选中；worker 列表顺序会变化，不能硬编码“索引 0 永远是某机器”。
单 P 测试不使用第二个 P；双 P 测试按下节的稳定会话亲和规则分配。
`decode-policy=round_robin` 是 D worker 级策略，不等于 DP rank 均衡；rank 选择需修复版的负载预占/释放逻辑。
`disable-worker-overload-protection` 是本轮显式设置；不应凭健康状态认定引擎已无在途请求。

## 双 P 配置与结果记录

双 P 对照沿用上面的 D 与 SMG，增加另一台独立的 8 卡 P；每个 P 均使用本页 P 的完整环境与参数。
第二个 P 的 `--host`、`--dist-init-addr` 改为自身地址，
`--data-parallel-rpc-port` 改为 `18512`，仍连接同一个 `GATEWAY_D0_HOST`。
两台 P 位于不同主机，本机 HTTP/metrics/bootstrap 端口可相同；缓存目录各自可写且独立。
两台 P 都开启 native FP4 Indexer，chunk 16640、max-num-seqs 32、显存比例 0.85；
不要仅给第一台 P 加速而让第二台保留另一条 Indexer 路径。

本次客户端按会话奇偶均分到两个 P，并在全部轮次保持该会话的 P 亲和性。
实际 worker 索引先用短请求和两台 P 日志核验，不能依赖注册顺序。
这是显式会话分配的对照实验，不是对默认 consistent-hashing 分布的验收。
prime **全体最多并发 8**，不是每 P 各 8；两台 P 的全部 prime 完成后才释放同一个全局屏障。
每档清理两个 P 和逻辑 D 三个 worker 的 KV，并核验没有 skipped。
总并发 C16/C32/C64/C128，D 始终为 16 卡；单卡吞吐只除 D 卡数，不含 P 卡。

双 P 四档均已完整验证，结果、限制和文件校验值见 [10 轮性能结果](results/performance/smg-single-dual-p-10round-20261110/INDEX.md)。
C16/C128 首次在线 JIT 的记录保留，最终表使用清 KV 后完整热运行；旧 Dynamo 8 轮数据未混入。

## 环境变量

以下为实际启动环境中 TS、CUDA、通信、算子、线程和库搜索路径相关项的完整白名单快照；
不复制登录环境、凭据、代理、shell 或任务调度平台注入项。
P 与 D 需隔离环境，避免跨角色残留 Indexer、wire1 或模拟接受率配置。

### P/D 相同

| 环境变量 | 实测值 / 可迁移占位符 |
|---|---|
| `CUDA_HOME` | `${CUDA_HOME}` |
| `CUDA_TOOLKIT_MODE` | `system` |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` |
| `DEEPEP_WIRE1_SOURCE` | `${DEEPEP_WIRE1_SOURCE}` |
| `NCCL_IB_GID_INDEX` | `7` |
| `NCCL_IB_HCA` | `mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9` |
| `NCCL_IB_SL` | `1` |
| `NCCL_P2P_DISABLE` | `1` |
| `NCCL_ROOT_DIR` | `${NCCL_ROOT_DIR}` |
| `NCCL_SHM_DISABLE` | `1` |
| `NCCL_SOCKET_IFNAME` | `eth0` |
| `NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME` | `eth0` |
| `NVSHMEM_DIR` | `${NVSHMEM_DIR}` |
| `NVSHMEM_IB_GID_INDEX` | `7` |
| `OMP_NUM_THREADS` | `200` |
| `TOKENSPEED_CUDA_ARCH` | `120a` |
| `TOKENSPEED_DISABLE_PLUGINS` | `qwen_sm120` |
| `TOKENSPEED_DISAGGREGATION_BOOTSTRAP_TIMEOUT` | `300` |
| `TOKENSPEED_EXPERIMENT_SM120_DEEPEP` | `1` |
| `TOKENSPEED_HOME` | `/opt/tokenspeed` |
| `TOKENSPEED_KERNEL_BACKEND` | `cuda` |
| `TOKENSPEED_KERNEL_DISABLE_PLUGINS` | `sm120` |
| `TOKENSPEED_KERNEL_OVERRIDE_RESIDUAL_MHC_MIXES` | `triton_mhc_mixes` |
| `TOKENSPEED_KERNEL_OVERRIDE_TRANSFORM_HADAMARD_TRANSFORM` | `triton_hadamard_transform_128` |
| `TOKENSPEED_KERNEL_VERSION_DATE` | `20261021` |
| `TOKENSPEED_WIRE1_NICS` | `mlx5_4,mlx5_5,mlx5_2,mlx5_3,mlx5_8,mlx5_9,mlx5_6,mlx5_7` |
| `TORCH_CUDA_ARCH_LIST` | `12.0a` |
| `TRITON_PTXAS_BLACKWELL_PATH` | `${CUDA_HOME}/bin/ptxas` |
| `TRITON_PTXAS_PATH` | `${CUDA_HOME}/bin/ptxas` |

### P 专有或与 D 不同

| 环境变量 | 实测值 / 可迁移占位符 |
|---|---|
| `CUDA_CACHE_PATH` | `${P_STATE_DIR}/cache/cuda` |
| `LD_LIBRARY_PATH` | `/usr/local/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${COMPILER_ROOT}/lib` |
| `PYTHONPATH` | `${REPO_ROOT}/python` |
| `TOKENSPEED_ENABLE_SM120_MHC_MIXES` | `0` |
| `TOKENSPEED_ENABLE_SM120_NATIVE_FP4_INDEXER` | `1` |
| `TOKENSPEED_ENABLE_SM120_PAGED_PREFILL` | `1` |
| `TOKENSPEED_ENABLE_SM120_PREFILL_INDEXER` | `1` |
| `TOKENSPEED_EXPERIMENT_SM120_DG_INDEXER` | `0` |
| `TOKENSPEED_FLASHINFER_AUTOTUNE_CACHE_DIR` | `${P_STATE_DIR}/cache/autotune` |
| `TOKENSPEED_KERNEL_CACHE_ROOT` | `${P_STATE_DIR}/cache/kernel` |
| `TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_DSV41_INDEX_TOPK` | `deep_gemm_dsv41_index_topk` |
| `TOKENSPEED_KERNEL_OVERRIDE_RESIDUAL_MHC_PRE` | `triton_mhc_pre` |
| `TOKENSPEED_SM120_DEEPEP_CAPACITY` | `2048` |
| `TOKENSPEED_SM120_DEEPEP_WIRE` | `elastic_fp8_bf16` |
| `TRITON_CACHE_DIR` | `${P_STATE_DIR}/cache/triton` |

### D0/D1 相同的专有配置

| 环境变量 | 实测值 / 可迁移占位符 |
|---|---|
| `CUDA_CACHE_PATH` | `${D_STATE_DIR}/cache/cuda` |
| `DEEPEP_WIRE1_AH_FIX` | `1` |
| `LD_LIBRARY_PATH` | `${NVSHMEM_DIR}/lib:/usr/local/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${COMPILER_ROOT}/lib` |
| `LD_PRELOAD` | `${DEEPEP_WIRE1_SOURCE}/tokenspeed_nvshmem_ah_fix.so` |
| `NVSHMEM_DISABLE_P2P` | `1` |
| `NVSHMEM_ENABLE_NIC_PE_MAPPING` | `1` |
| `NVSHMEM_IBGDA_NIC_HANDLER` | `gpu` |
| `NVSHMEM_IB_ENABLE_IBGDA` | `1` |
| `NVSHMEM_ROOT_DIR` | `${NVSHMEM_DIR}` |
| `PYTHONPATH` | `${DECODE_OBSERVER_DIR}:${DEEPEP_WIRE1_SOURCE}:${REPO_ROOT}/python` |
| `TOKENSPEED_EXPERIMENT_FUSED_INDEX_PACK` | `1` |
| `TOKENSPEED_EXPERIMENT_SM120_DEEPSELECT` | `1` |
| `TOKENSPEED_EXPERIMENT_SM120_DG_INDEXER` | `1` |
| `TOKENSPEED_EXPERIMENT_SM120_FUSED_SCALE_PACK` | `1` |
| `TOKENSPEED_EXPERIMENT_SM120_SELECTED_ATTENTION` | `gather` |
| `TOKENSPEED_EXPERIMENT_SM120_SHARED_OVERLAP` | `1` |
| `TOKENSPEED_EXPERIMENT_SM120_WO_A` | `mxfp8` |
| `TOKENSPEED_FLASHINFER_AUTOTUNE_CACHE_DIR` | `${D_STATE_DIR}/cache/autotune` |
| `TOKENSPEED_KERNEL_CACHE_ROOT` | `${D_STATE_DIR}/cache/kernel` |
| `TOKENSPEED_SM120_DEEPEP_CACHED_CLEAN` | `1` |
| `TOKENSPEED_SM120_DEEPEP_CAPACITY` | `128` |
| `TOKENSPEED_SM120_DEEPEP_COMPACT` | `1` |
| `TOKENSPEED_SM120_DEEPEP_EXPERT_IMPL` | `web` |
| `TOKENSPEED_SM120_DEEPEP_WIRE` | `nvfp4_fp8` |
| `TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN` | `4` |
| `TRITON_CACHE_DIR` | `${D_STATE_DIR}/cache/triton` |

### 环境使用说明

- `CUDA_LAUNCH_BLOCKING` 未设置（关闭）；不要为性能压测开启。
- `TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN=4` **仅用于性能测试**，本轮实测 draft 接受率约 59.921875%。
  精度或正常推理必须取消它，使用真实验证结果；不能将本性能配置直接当作精度配置。
- `DECODE_OBSERVER_DIR` 是本次测量 hook 的位置，只记录 CPU 侧同步 DP metadata 的 `global_num_decodes`。
  它不属于推理依赖。普通部署从 D `PYTHONPATH` 移除这一项；若复现精确全局 BS，需部署已审查的观测器，
  设置 `WCR_DECODE_BATCH_LOG` 为本机文件，并保留观测开销。各 rank 独立峰值相加不是真实全局 BS。
- `TOKENSPEED_HOME`、`TOKENSPEED_KERNEL_VERSION_DATE`、插件禁用项是当前镜像继承值，不表示这些就是源码 commit。
  P 虽继承 wire1 源目录与 NIC 变量，实际 wire 为 Elastic；不要把 D 的 `LD_PRELOAD` 和 transport 选择传给 P。
- D 的 AH workaround 为本轮 NVSHMEM/网卡组合保留，详见 wire1 安装文档；不能泛化为所有 RDMA 错误的修复。
- SMG 使用自己的 venv，运行时不继承 TS 的 `PYTHONPATH`。实际测试未使用 Dynamo 路由环境变量。

## 压测协议和统计

1. 同一批 LongBench 高接受率候选，按并发 C 取前 C 个独立会话；精确 token IDs，不能重新 tokenize 后仍宣称相同输入。
2. 每档清 KV，保留 JIT 缓存。必须确认全部已注册 worker 清理成功且无 skipped；HTTP 200 或 `/workers` 空闲还不够。
   若清理返回 engine not idle，等待实际请求排空后重试，不能继续发测量请求。
3. 每个会话 prime 102400 token，prime 最多并发 8；全部 prime 完成后一次全局屏障。
4. 10 轮分别输入 112640…204800 token，每轮新增 10240；每轮输出 1024。
   各会话结束后立即发下一轮，不等其他会话；不拼接上轮输出。prime 和正式请求都使用 ignore_eos、temperature=0、seed=37。
5. 每轮预期命中上轮输入前缀，输入输出长度、缓存、错误、DSpark 接受率与 JIT 均需核验。
6. TTFT 为正式请求平均；TPOT 为每请求 `(最后输出时刻−首次输出时刻)/(输出token−1)` 后求平均，排除该请求 TTFT。
   单请求 TPS 为 `1000/平均TPOT毫秒`。
7. 全 batch 实测输出吞吐为全部正式输出 token 总和除以“最晚末 token−最早首 token”；
   包含后续轮次 prefill、排队、并发爬升和尾部。单卡吞吐再除以 **16 张 D 卡**，不计独立 P 的 8 张卡。
   这不是 `并发数×单请求TPS`，也不是满 BS 稳态窗口吞吐。
8. 首次新形状可能在线 JIT；同配置再次清 KV 复测并单独记录 JIT，不能混淆热 kernel 与 prefix 命中。
   profiler 请求使用 `activities=["CPU","GPU"]`。此镜像实测同进程先 CPU-only 再启 GPU 可能没有 kernel 事件，
   新进程首次直接 CPU/GPU 可采到；必须检查 trace 内容，不能仅凭 HTTP success。

## 当前性能依据

完整协议、P 处理/排队线索及性能图见 [结果归档](results/performance/smg-single-dual-p-10round-20261110/INDEX.md)。

| 配置 | 并发 | TTFT (s) | TPOT (ms) | 实测输出 (token/s) | 每 D 卡 (token/s/GPU) | 最大 Decode BS |
|---|---:|---:|---:|---:|---:|---:|
| 1P1D | 16 | 0.963 | 6.536 | 2080.68 | 130.04 | 16 |
| 1P1D | 32 | 2.598 | 7.160 | 3172.47 | 198.28 | 28 |
| 1P1D | 64 | 11.471 | 7.475 | 3280.69 | 205.04 | 31 |
| 2P1D | 16 | 0.841 | 6.549 | 2145.95 | 134.12 | 16 |
| 2P1D | 32 | 0.964 | 7.484 | 3700.09 | 231.26 | 32 |
| 2P1D | 64 | 1.864 | 8.553 | 5949.89 | 371.87 | 62 |
| 2P1D | 128 | 8.499 | 10.551 | 6514.61 | 407.16 | 81 |

七档全部完整成功、前缀命中、正式区间无在线 JIT，draft 接受率 59.921875%。
单 P C32 首次 native FP4 测试最高 248 ms JIT，双 P C16/C128 首次分别最高 3925/264 ms；均保留首次记录并使用完整热运行复测作为最终值。
单 P 非 native FP4 的 C32 对照为 TTFT 6.003 s、TPOT 6.787 ms、输出 2426.95 token/s；仅修改 P Indexer 两个环境变量后恢复约 3172.47 token/s。
不能把模拟接受率的性能成绩作为模型精度验收。
