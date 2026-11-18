# Container paths and routable addresses; edit before launch.
export MODEL=/models/DeepSeek-V4.1-Flash-1017
export D_HOST=192.0.2.11
export P_HOST=192.0.2.42
export P_COUNT=1
# P1:18510, P2:18512
export P_RPC_PORT=18510
export TS_ROOT=/tokenspeed
export SMG_RUNTIME=/opt/gateway-runtime
export PYTHON=python3
export MAX_NUM_SEQS=256
export GRAPH_MAX=32
export GRAPH_CAPTURE_SIZES="1 2 4 8 16 32"

# Optional: use image-bundled CUDA 13 forward compatibility libraries.
export H20_CUDA_COMPAT=0
export H20_CUDA_COMPAT_DIR=/opt/cuda-compat-13.0
