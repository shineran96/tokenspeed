#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
D_NNODES="${D_NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
D_LOCAL_HOST="${D_LOCAL_HOST:-$D_HOST}"
DEEP_EP_MOE="${DEEP_EP_MOE:-flashinfer_cutlass}"
case "$D_NNODES" in 1|2) ;; *) echo "D_NNODES must be 1 or 2" >&2; exit 2;; esac
[[ "$NODE_RANK" =~ ^[0-9]+$ ]] && (( NODE_RANK < D_NNODES )) || exit 2
MOE_ARGS=()
case "$DEEP_EP_MOE" in
    flashinfer_cutlass) ;;
    humming) MOE_ARGS+=(--moe-mxfp4-fp8-activation) ;;
    *) echo "Unsupported DEEP_EP_MOE" >&2; exit 2 ;;
esac
WORLD_SIZE=$((8 * D_NNODES))
if [[ "${TRACE_BATCHES:-1}" == 1 ]]; then
    export H20_PD_BATCH_TRACE="${H20_PD_BATCH_TRACE:-$HERE/decode-node${NODE_RANK}-batches.jsonl}"
else
    unset H20_PD_BATCH_TRACE
fi
export PYTHONPATH="$HERE/observer:$PYTHONPATH"
if [[ -n "${SIMULATED_ACCEPT_LEN:-}" ]]; then
    export TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN="$SIMULATED_ACCEPT_LEN"
else
    unset TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN
fi
export TOKENSPEED_HOST_IP="$D_LOCAL_HOST"
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
    --data-parallel-size "$WORLD_SIZE" \
    --nnodes "$D_NNODES" \
    --node-rank "$NODE_RANK" \
    --world-size "$WORLD_SIZE" \
    --enable-expert-parallel \
    --all2all-backend deepep \
    --moe-backend "$DEEP_EP_MOE" \
    --deepep-mode low_latency \
    --low-latency-max-num-tokens-per-gpu 256 \
    "${MOE_ARGS[@]}" \
    --chunked-prefill-size 8192 \
    --disable-prefix-caching \
    --max-cudagraph-capture-size "$GRAPH_MAX" \
    --cudagraph-capture-sizes "${GRAPH_SIZES[@]}" \
    --disaggregation-mode decode \
    --data-parallel-rpc-port 18511 \
    --host "$D_LOCAL_HOST" \
    --port 18347 \
    --dist-init-addr "$D_HOST:13580" \
    "$@"
