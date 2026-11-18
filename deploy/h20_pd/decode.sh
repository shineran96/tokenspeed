#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
if [[ "${TRACE_BATCHES:-1}" == 1 ]]; then
    export H20_PD_BATCH_TRACE="${H20_PD_BATCH_TRACE:-$HERE/decode-global-batches.jsonl}"
else
    unset H20_PD_BATCH_TRACE
fi
export PYTHONPATH="$HERE/observer:$PYTHONPATH"
if [[ -n "${SIMULATED_ACCEPT_LEN:-}" ]]; then
    export TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN="$SIMULATED_ACCEPT_LEN"
else
    unset TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN
fi
export TOKENSPEED_HOST_IP="$D_HOST"
GRAPH_MAX="${GRAPH_MAX:-32}"
read -r -a GRAPH_SIZES <<< "${GRAPH_CAPTURE_SIZES:-1 2 4 8 16 32}"
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
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --world-size 8 \
    --enable-expert-parallel \
    --all2all-backend none \
    --moe-backend auto \
    --chunked-prefill-size 8192 \
    --disable-prefix-caching \
    --max-cudagraph-capture-size "$GRAPH_MAX" \
    --cudagraph-capture-sizes "${GRAPH_SIZES[@]}" \
    --disaggregation-mode decode \
    --data-parallel-rpc-port 18511 \
    --host "$D_HOST" \
    --port 18347 \
    --dist-init-addr 127.0.0.1:13580 \
    "$@"
