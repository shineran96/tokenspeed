#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${CONFIG:-}" ]]; then
    source "$CONFIG"
fi
: "${MODEL:?Set MODEL to the model directory mounted inside the container}"
: "${D_HOST:?Set D_HOST to the decode node address reachable from P}"
TS_ROOT="${TS_ROOT:-$(cd "$HERE/../.." && pwd)}"
SMG_RUNTIME="${SMG_RUNTIME:-/opt/gateway-runtime}"
PYTHON="${PYTHON:-python3}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"

case "${H20_CUDA_COMPAT:-0}" in
    0) ;;
    1)
        H20_CUDA_COMPAT_DIR="${H20_CUDA_COMPAT_DIR:-/opt/cuda-compat-13.0}"
        if [[ "${DRY_RUN:-0}" != 1 ]]; then
            test -r "$H20_CUDA_COMPAT_DIR/libcuda.so.1"
            test -r "$H20_CUDA_COMPAT_DIR/libnvidia-ptxjitcompiler.so.1"
        fi
        export LD_LIBRARY_PATH="$H20_CUDA_COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        ;;
    *) echo "H20_CUDA_COMPAT must be 0 or 1" >&2; exit 2 ;;
esac

export PYTHONPATH="$SMG_RUNTIME:$TS_ROOT/python:$TS_ROOT/tokenspeed-kernel/python:$TS_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NO_PROXY="*" no_proxy="*"
unset CUDA_LAUNCH_BLOCKING TOKENSPEED_HOST_BINARY NCCL_P2P_DISABLE NCCL_SHM_DISABLE
export TOKENSPEED_DISAGGREGATION_WAITING_TIMEOUT=7200
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    command -v "$PYTHON" >/dev/null
    test -d "$TS_ROOT/python/tokenspeed"
    test -d "$MODEL"
fi
run() {
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf '%q ' "$@"
        printf '\n'
    else
        exec "$@"
    fi
}
