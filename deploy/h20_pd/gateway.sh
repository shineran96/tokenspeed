#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
P_COUNT="${P_COUNT:-1}"
case "$P_COUNT" in
    1) PREFILL_ARGS=(--prefill "tcp://$D_HOST:${P_RPC_PORT:-18510}" 18598) ;;
    2) PREFILL_ARGS=(--prefill "tcp://$D_HOST:18510" 18598 --prefill "tcp://$D_HOST:18512" 18598) ;;
    *) echo "P_COUNT must be 1 or 2" >&2; exit 2 ;;
esac
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    test -d "$SMG_RUNTIME/smg"
fi
run "$PYTHON" -m smg launch \
    --pd-disaggregation \
    --backend tokenspeed \
    --zmq-startup-timeout-secs 7200 \
    "${PREFILL_ARGS[@]}" \
    --decode "tcp://$D_HOST:18511" \
    --model-path "$MODEL" \
    --tokenizer-path "$MODEL" \
    --model-alias "deepseek-v41-flash=$MODEL" \
    --host "$D_HOST" \
    --port 18500 \
    --prefill-policy cache_aware \
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
    "$@"
