#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
gateway_venv="$HOME/codes/gateway-ae2139e/.venv"
export PYTHONPATH="$gateway_venv/lib/python3.12/site-packages:$PYTHONPATH"
exec "$gateway_venv/bin/python" -m smg launch \
    --enable-rl \
    --pd-disaggregation \
    --backend tokenspeed \
    --zmq-startup-timeout-secs 7200 \
    --prefill "tcp://$D_HOST:28510" 28598 \
    --decode "tcp://$D_HOST:28511" \
    --decode "tcp://$D_HOST:28512" \
    --model-path "$MODEL" \
    --tokenizer-path "$MODEL" \
    --model-alias "deepseek-v41-flash=$MODEL" \
    --host "$D_HOST" \
    --port 28500 \
    --prefill-policy cache_aware \
    --decode-policy round_robin \
    --max-concurrent-requests 256 \
    --queue-size 512 \
    --queue-timeout-secs 7200 \
    --request-timeout-secs 7200 \
    --disable-retries \
    --disable-circuit-breaker \
    --prometheus-port 28520 \
    --worker-stall-secs 7200 \
    --worker-wedge-secs 7200 \
    --worker-stale-secs 7200 \
    --max-payload-size 33554432 \
    --max-buffered-request-bytes 33554432 \
    "$@"
