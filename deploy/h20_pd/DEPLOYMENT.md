# H20 DeepSeek-V4.1 PD deployment and DeepEP tests

This guide covers the committed H20 branch. All paths are configurable.
The baseline uses one or two independent 8-GPU P instances and one 8-GPU
D instance. The experimental DeepEP D can span two 8-GPU nodes.

## Environment

Use the same compatible TS revision and model on every node. Install the
repository following its root installation instructions, including the matching
tokenspeed-kernel build. Select that environment with PYTHON and this checkout
with TS_ROOT. SMG is a separately installed dependency; set SMG_RUNTIME to
the directory containing its smg package. These scripts do not install packages.

Required components include CUDA-compatible PyTorch/TS kernels, FlashInfer,
Moонcake for KV transfer, and SMG with TokenSpeed headless/ZMQ support.
DeepEP tests additionally require a working Legacy DeepEP/NVSHMEM installation.
Humming is optional; the integration was developed against
d0c7af98d7f556132f86e671de7f397991e2df2b.

Do not assume any DeepEP/NVSHMEM versions are interchangeable. In particular,
the committed TS code does not include the external CPU-notification compatibility
patches used in dependency experiments. Verify the installed dependency pair on
both nodes before attempting cross-node service tests. Humming remains
experimental; successful launch is not proof of model accuracy.

Use addresses routable between containers, expose the service ports below and
the required RDMA devices. Check GPU visibility, peer access and RDMA device
mapping. Network-interface and HCA names must match the actual machines.

## Shared configuration

Run from deploy/h20_pd on each node. Either export variables or use CONFIG
pointing to an edited config.example.sh:

```bash
export MODEL=/models/DeepSeek-V4.1-Flash-1017
export TS_ROOT=/workspace/tokenspeed
export PYTHON=/workspace/venv/bin/python
export SMG_RUNTIME=/opt/gateway-runtime
export D_HOST=192.0.2.11
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
```

Replace example paths and addresses. Defaults: maximum context 1M, memory
fraction 0.85, host shared Engram, maximum requests 256, DSpark enabled.
P uses QCP8/EP8, 16384-token chunks, eager execution and prefix caching.
Baseline D uses DP8/EP8, 8192-token chunks, CUDA Graph sizes 1/2/4/8/16/32
and disables prefix caching.

CUDA_LAUNCH_BLOCKING is cleared. H20_CUDA_COMPAT=1 optionally enables the
compatibility libraries at H20_CUDA_COMPAT_DIR; use only if installed and
appropriate for the host driver.

## Baseline 1P1D with SMG

On the D host, start SMG and D in separate terminals/processes:

```bash
bash gateway.sh > gateway.log 2>&1 &
bash decode.sh > decode.log 2>&1 &
```

On the P host:

```bash
export P_HOST=192.0.2.42
export P_RPC_PORT=18510
bash prefill.sh > prefill.log 2>&1 &
```

SMG exposes HTTP on D_HOST:18500. It uses cache-aware P selection and
round-robin D worker selection. DP-rank selection is also dependent on the
installed SMG implementation; do not infer strict per-rank balance from the
worker policy flag. Verify actual rank loads during tests.

P and D are headless TS processes, not separate HTTP endpoints for clients.
SMG coordinates prefill, KV transfer metadata and decode responses. KV is
transferred through Mooncake; it is not relayed through the HTTP client.

## 2P1D

Start gateway.sh with P_COUNT=2. Use the same D startup.
Start two P instances with their respective P_HOST and P_RPC_PORT values
18510 and 18512. These scripts register P RPC endpoints at D_HOST, so P
processes must reach the D host. Each P uses its own eight GPUs.

## DeepEP D: single-node EP8

Keep P and SMG as above. Stop the baseline D before using its ports and GPUs.

```bash
# CUTLASS W4A16, BF16 dispatch/combine
DEEP_EP_MOE=flashinfer_cutlass bash decode_deepep.sh > decode-deepep.log 2>&1 &

# Alternative experimental Humming W4A8:
# FP8 dispatch, BF16 combine
# DEEP_EP_MOE=humming bash decode_deepep.sh > decode-humming.log 2>&1 &
```

Use only one D command at a time. The new entry point selects Legacy
low-latency DeepEP and dispatch capacity 256 token rows per rank.
This capacity includes speculative verify rows and Graph padding, not just
the number of requests. It is not an FP8 combine or ElasticBuffer launcher.
The P launcher remains non-DeepEP so that the D comparison is isolated.
A P DeepEP configuration needs separate normal-mode validation; do not apply
this low-latency-only adapter blindly to large prefill batches.

## DeepEP D: two-node DP16/EP16

Keep SMG on the first D host. On both D nodes, use identical software,
model, backend and network configuration. On the first D node:

```bash
export D_HOST=192.0.2.11
D_NNODES=2 NODE_RANK=0 D_LOCAL_HOST=192.0.2.11 \
  DEEP_EP_MOE=flashinfer_cutlass bash decode_deepep.sh > d0.log 2>&1 &
```

On the second D node:

```bash
export D_HOST=192.0.2.11
D_NNODES=2 NODE_RANK=1 D_LOCAL_HOST=192.0.2.12 \
  DEEP_EP_MOE=flashinfer_cutlass bash decode_deepep.sh > d1.log 2>&1 &
```

Use DEEP_EP_MOE=humming on both nodes to test Humming instead.
The two nodes form one logical D worker with 16 DP/EP ranks. SMG still points
to the first D host RPC endpoint. Do not launch a second SMG for rank1.

For an installation that explicitly requires the NVSHMEM 3.4.5 CPU-assisted
path, the tested environment knobs included:

```bash
export NVSHMEM_IBGDA_NIC_HANDLER=cpu_host_memory
export NVSHMEM_IBGDA_FORCE_NIC_BUF_MEMTYPE=hostmem
export NVSHMEM_QP_DEPTH=1024
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
export NVSHMEM_HCA_PE_MAPPING='mlx5_0:1:1,mlx5_1:1:1,mlx5_2:1:1,mlx5_3:1:1,mlx5_4:1:1,mlx5_5:1:1,mlx5_6:1:1,mlx5_7:1:1'
unset NVSHMEM_HCA_LIST
export NCCL_IB_DISABLE=0
```

These values are not universal defaults and do not fix an incompatible native
library. Validate HCA order, GID index and traffic class with the actual fabric;
do not copy a different cluster's GID/SL settings blindly.

## Readiness and smoke test

Wait for weight loading, tuning and Graph capture to finish, then inspect:

```bash
curl --fail "http://$D_HOST:18500/workers"
curl --fail "http://$D_HOST:18500/generate" \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v41-flash","text":"Explain what a GPU is.","sampling_params":{"temperature":0,"max_new_tokens":64},"stream":false}'
```

Require healthy/ready P and D workers with fresh engine-load reports. After
restarting D, stale SMG readiness alone is insufficient. For DP8/DP16, verify
that eight/sixteen rank load entries appear before starting a batch.

## Accuracy and performance checks

Default scripts use real acceptance. For accuracy, leave SIMULATED_ACCEPT_LEN
unset and verify it is absent from the D environment.

For fixed-acceptance performance only:

```bash
SIMULATED_ACCEPT_LEN=4 bash decode_deepep.sh
```

Length 4 means one anchor plus three accepted draft tokens in the tested
configuration. This synthetic mode is not an accuracy result.

Start with C1, then C32/C64. Reuse identical tokenized inputs, output limits,
cache state and acceptance settings across backends. Warm up before formal
rounds; retain failures and per-rank distributions. Record TTFT, per-request
TPOT excluding TTFT, actual output-window throughput and per-D-GPU throughput.
Do not replace actual throughput with concurrency divided by TPOT.

TRACE_BATCHES=1 enables the included scheduler observer; each D node writes a
separate trace by default in the DeepEP launcher. Inspect maximum and sustained
per-rank decode counts. Submitted concurrency is not proof of a full batch.

For DeepEP correctness, cover target/draft transitions, empty ranks, changing
routing and repeated Graph replay before reporting performance. Profile dispatch
send and receive separately: receive duration includes peer waiting. Repeat
timing without profiler to rule out startup skew.

## Ports and troubleshooting

| Purpose | Default |
|---|---|
| SMG HTTP / metrics | 18500 / 18520 |
| P RPC endpoints at D host | 18510, 18512 |
| D RPC | 18511 |
| P bootstrap | 18598 |
| P internal initialization | P_HOST:19400 |
| Baseline D initialization | loopback:13580 |
| DeepEP D distributed initialization | D_HOST:13580 |

A shared model mount does not imply shared process environments. Check the
actual Python and imported package locations on every host. Missing kernel
images require an SM90-compatible build. Native communication timeouts require
checking all ranks, RDMA mappings, dependency compatibility and stale processes,
not merely increasing the timeout. Check actual backend selection in startup
logs; auto selection is not equivalent to a forced backend.

These launchers have syntax/dry-run validation in this change. Fresh end-to-end
deployment of the committed scripts and a compatible dependency stack remains
a runtime acceptance step.
