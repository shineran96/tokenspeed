#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
export MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=16384
export MC_TCP_LANES_PER_PEER=4
export P_HOST="${P_HOST:?Set P_HOST}"
export P_RPC_PORT="${P_RPC_PORT:?Set P_RPC_PORT}"
unset TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN
export TOKENSPEED_HOST_IP="$P_HOST"
exec "$TS_PYTHON_DIR/python" -m tokenspeed.cli serve \
    --headless \
    --model "$MODEL" \
    --served-model-name deepseek-v41-flash \
    --trust-remote-code \
    --dtype bfloat16 \
    --seed 968340402 \
    --max-model-len 262144 \
    --max-num-seqs 256 \
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
    --tensor-parallel-size 4 \
    --world-size 4 \
    --enable-expert-parallel \
    --query-context-parallel-size 4 \
    --kv-parallel-size 1 \
    --chunked-prefill-size 8192 \
    --enforce-eager \
    --enable-prefix-caching \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port 28598 \
    --data-parallel-rpc-port "$P_RPC_PORT" \
    --host "$P_HOST" \
    --port 28501 \
    --dist-init-addr "$P_HOST:29400" \
    "$@"
