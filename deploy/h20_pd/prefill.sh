#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
export P_HOST="${P_HOST:?Set P_HOST}"
export P_RPC_PORT="${P_RPC_PORT:-18510}"
unset TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN
export TOKENSPEED_HOST_IP="$P_HOST"
run "$PYTHON" -m tokenspeed.cli serve \
    --headless \
    --model "$MODEL" \
    --served-model-name deepseek-v41-flash \
    --trust-remote-code \
    --dtype bfloat16 \
    --seed 968340402 \
    --max-model-len 1048576 \
    --max-num-seqs "$MAX_NUM_SEQS" \
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
    --tensor-parallel-size 8 \
    --world-size 8 \
    --enable-expert-parallel \
    --all2all-backend none \
    --moe-backend auto \
    --query-context-parallel-size 8 \
    --kv-parallel-size 1 \
    --chunked-prefill-size 16384 \
    --enforce-eager \
    --enable-prefix-caching \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port 18598 \
    --data-parallel-rpc-port "$P_RPC_PORT" \
    --host "$P_HOST" \
    --port 18501 \
    --dist-init-addr "$P_HOST:19400" \
    "$@"
