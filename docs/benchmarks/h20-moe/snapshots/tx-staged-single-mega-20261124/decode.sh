#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
export H20_PROFILE_DIR="$HERE/$BENCH_BACKEND"
export NVSHMEM_IBGDA_NIC_HANDLER=gpu NVSHMEM_DEBUG=INFO
unset NVSHMEM_IBGDA_FORCE_NIC_BUF_MEMTYPE
export H20_PD_BATCH_TRACE="$HERE/${BENCH_BACKEND:-baseline}-batches.jsonl"
export PYTHONPATH="$HERE/observer:$PYTHONPATH"
export TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN=4.0 TOKENSPEED_HOST_IP="$D_LOCAL_HOST"
exec "$TS_PYTHON_DIR/python" -m tokenspeed.cli serve \
    --headless \
    --model "$MODEL" \
    --served-model-name deepseek-v41-flash \
    --trust-remote-code \
    --dtype bfloat16 \
    --seed 968340402 \
    --max-model-len 262144 \
    --max-num-seqs 128 \
    --speculative-algorithm DSPARK \
    --disable-prefill-graph \
    --disable-kvstore \
    --engram-host-table \
    --engram-host-table-layout shared \
    --gpu-memory-utilization 0.85 \
    --disable-weight-loader-prefetch-checkpoints \
    --decode-log-interval 1 \
    --enable-log-request-stats \
    --distributed-timeout-seconds 7200 \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-layerwise-interval 0 \
    --data-parallel-address "$D_HOST" \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --world-size 8 \
    --enable-expert-parallel \
    --deepseek-v4-mega-moe-max-num-tokens 96 \
    --chunked-prefill-size 8192 \
    --disable-prefix-caching \
    --max-cudagraph-capture-size 128 \
    --cudagraph-capture-sizes 1 2 4 8 16 \
    --disaggregation-mode decode \
    --data-parallel-rpc-port "${D_RPC_PORT:?}" \
    --host "$D_LOCAL_HOST" \
    --port 28347 \
    --dist-init-addr "$D_LOCAL_HOST:23580" \
    --all2all-backend none \
    --moe-backend mega_moe \
    "$@"
