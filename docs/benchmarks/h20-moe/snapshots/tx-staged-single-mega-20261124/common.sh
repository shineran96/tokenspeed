#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $(hostname) == set-tx-* ]]; then
    source "$HOME/codes/environment/activate.sh"
    export TS_ROOT="$HOME/codes/tokenspeed-ep16"
    export PYTHONPATH="/opt/gateway-runtime:$TS_ROOT/python:$TS_ROOT/tokenspeed-kernel/python:$TS_ROOT"
    export TS_PYTHON_DIR="$HOME/codes/tokenspeed/.venv/bin"
else
    source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/wangchangran/ts_master_h20/local_work/pd17-37-c32-128k-20261122/wxy-cutedsl17-19/common.sh
    HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
export D_HOST=33.253.229.168
export P_HOST=33.253.136.240
export P_RPC_PORT=28510
unset MC_FORCE_TCP
export MC_TRANSFER_TIMEOUT=3600
export TOKENSPEED_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export TOKENSPEED_DISAGGREGATION_WAITING_TIMEOUT=7200
export NO_PROXY='*' no_proxy='*'
export NVSHMEM_IBGDA_NIC_HANDLER=gpu
export NVSHMEM_QP_DEPTH=2048
unset NVSHMEM_IBGDA_FORCE_NIC_BUF_MEMTYPE CUDA_LAUNCH_BLOCKING
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
export PYTHONPATH="$HERE/observer:$PYTHONPATH"
