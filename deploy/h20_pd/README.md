# H20 镜像内启动 1P1D / 2P1D（D DP8/EP8）

本目录仅包含启动文件，可整体复制到各节点容器。参数逐项换行。
使用此前 image.dockerfile 定义的环境：
- Python：镜像 PATH 中的 python3，不激活宿主机虚拟环境。
- TS 源码：默认使用当前仓库；镜像部署可设置 TS_ROOT=/tokenspeed。
- SMG：/opt/gateway-runtime，优先于其他安装版本。
- CUDA 和编译缓存：继承镜像环境，不覆盖为宿主机目录。
- 权重：通过 MODEL 指定容器内挂载路径。

P：QCP8/EP8，chunk=16384，prefix cache 开启，eager。
D：DP8/EP8，chunk=8192，prefix cache 关闭，DSpark + CUDA Graph。
P/D：max context=1M、显存比例0.85、host Engram。
使用普通 EP（all2all_backend=none），并非正在调试的 DeepEP/DP16。
D 默认真实接受率。复现固定接受长度4的性能测试时设置 SIMULATED_ACCEPT_LEN=4；精度测试不要设置。

## 1P1D

每个节点先进入本目录，设置 MODEL 和 D_HOST（示例地址请替换）：

```bash
export MODEL=/models/DeepSeek-V4.1-Flash-1017
export D_HOST=192.0.2.11
```

D 节点：

```bash
nohup bash gateway.sh > gateway.log 2>&1 &
nohup bash decode.sh > decode.log 2>&1 &
```

P 节点：

```bash
export P_HOST=192.0.2.42
export P_RPC_PORT=18510
nohup bash prefill.sh > prefill.log 2>&1 &
```

## 2P1D

D 节点用 P_COUNT=2 启动 gateway，decode 命令不变：

```bash
P_COUNT=2 nohup bash gateway.sh > gateway.log 2>&1 &
nohup bash decode.sh > decode.log 2>&1 &
```

P1 同上。P2：

```bash
export P_HOST=192.0.2.28
export P_RPC_PORT=18512
nohup bash prefill.sh > prefill.log 2>&1 &
```

P_RPC_PORT 是注册在 D 地址上的 headless RPC 端点，P1/P2 必须不同。
SMG 在 D 节点提供 HTTP 18500；两种拓扑不能同时占用同一组端口。
各节点需共享可连通的网络，部署容器时按实际环境开放相应端口及 RDMA 设备。

## 历史配置与检查

默认 max-num-seqs=256，D graph 档位1/2/4/8/16/32，对应后期高并发测试配置。
若复现较早 2P1D C1–C32 的服务上限，各 P/D 设置 MAX_NUM_SEQS=32；
D 另设置 GRAPH_MAX=16、GRAPH_CAPTURE_SIZES='1 2 4 8 16'。

运行前可打印命令，不加载模型、不使用 GPU：

```bash
DRY_RUN=1 bash decode.sh
DRY_RUN=1 P_COUNT=2 bash gateway.sh
DRY_RUN=1 P_HOST=192.0.2.42 bash prefill.sh
```

默认 D 加载 observer/sitecustomize.py，实际 batch 写入
decode-global-batches.jsonl。可通过 H20_PD_BATCH_TRACE 指定可写位置，
或 TRACE_BATCHES=0 关闭。日志及缓存目录应对容器运行用户可写。

已完成 shell 语法和命令展开检查；本次整理没有在目标镜像内重新启动服务。
镜像构建及 GPU 运行验收状态应以实际构建日志、启动日志为准。

## 独立迁移

将整个 h20_pd_launch 目录复制到各节点任意可写位置；无需原实验目录或宿主机虚拟环境。observer 随目录提供，无外部脚本依赖。使用已有 H20 镜像中的依赖；若镜像源码位置不同，设置 TS_ROOT、SMG_RUNTIME。编译缓存继承镜像环境。

复制 config.example.sh 为 config.sh，编辑模型挂载路径及节点地址。通过 CONFIG 指定配置，例如：

    CONFIG="$PWD/config.sh" DRY_RUN=1 bash decode.sh
    CONFIG="$PWD/config.sh" bash decode.sh > decode.log 2>&1

P 节点执行 prefill.sh；D 节点另行执行 gateway.sh。2P1D 的 gateway 设置 P_COUNT=2；P1/P2 分别设置 P_RPC_PORT=18510/18512 及本节点 P_HOST。D_HOST 是 headless RPC 所在的 D 地址。

本目录保留之前 DP8 普通 EP 测试配置，不包含实验数据、旧环境路径或双机 DP16 DeepEP 调试。固定接受长度4仅用于性能测试。端口沿用上文列出的默认值；部署时需开放这些端口，避免同机服务冲突。

## CUDA 13 旧驱动备选配置

镜像额外解压固定版本 cuda-compat-13-0 580.65.06 到 /opt/cuda-compat-13.0。
默认 H20_CUDA_COMPAT=0，使用宿主注入的驱动库。
对于已验证兼容的旧驱动 H20 部署，可在 config.sh 中设置 H20_CUDA_COMPAT=1，
P、D 和 gateway 均经 common.sh 将此目录加入 LD_LIBRARY_PATH 最前面。
不修改系统驱动、不使用外部虚拟环境、不改全局 ldconfig。

该开关必须在 Python 进程启动前设置。nvidia-smi 仍显示宿主驱动版本，
应通过进程实际加载的 libcuda 路径、cuDriverGetVersion 和 GPU 计算检查验证。
历史 worker23 的550.127.08上通过基础计算、Graph和NCCL验证；
这不等于当前04/37镜像已经通过完整PD验收。
新镜像构建仅执行下载、解压和文件检查，不运行GPU测试。
不要照搬历史 FLASHINFER_DISABLE_JIT=1；当前FI 0.7.3缓存命中需单独验证。

## H20 FlashInfer 专用二进制缓存

镜像将经过PD启动、Graph和9条请求冒烟验证的 fused_moe_90_h20_prebuilt_d.so
安装到 flashinfer/data/aot/fused_moe_90_h20_prebuilt_d/。
FI 0.7.3检测到H20后会选择此模块；通用SM90a wheel的 fused_moe_90.so 不能替代这个缓存键。
使用AOT目录可避免仅复制JIT .so后仍进入Ninja依赖检查。无需禁用JIT或修改FlashInfer源码。

归档压缩后约41.6 MiB，固定S3内容地址并校验归档及内部文件SHA256；
构建检查Torch、FlashInfer、TVM FFI及CUTLASS DSL版本，不执行GPU测试。
原有autotune参数缓存保持独立：配置或环境摘要不匹配时仍会重新调优，
本次新增二进制缓存不保证调优缓存命中，也不覆盖所有Triton/CuTe首次编译。

## 仅追加缓存的增量镜像

父目录 image-cache-incremental.dockerfile 使用当前已构建的H20 PD镜像作为
BASE_IMAGE，不使用最初的CUDA/Python基础镜像。只下载、校验并安装H20专用
FlashInfer AOT模块，不重新编译TS或安装依赖，继承已有CUDA兼容库、SMG和环境。

    docker build --build-arg BASE_IMAGE=<已有H20镜像:tag> -f image-cache-incremental.dockerfile -t <新镜像:tag> .

构建平台自动提供FROM时，选择已有H20镜像，仅粘贴该文件的RUN部分。
只进行CPU文件和版本检查，不执行GPU测试。尚未实际构建此增量镜像。

## Committed branch scope

This directory contains the historical non-DeepEP DP8 baseline launchers, not an EP16 Humming deployment. Use an installed Python environment compatible with this branch and an SMG version supporting the listed TokenSpeed options. TS_ROOT defaults to the repository containing this directory; set it explicitly after copying this folder elsewhere. CONFIG may point to an edited config.example.sh. For command inspection use DRY_RUN=1.

完整部署、SMG 与 DeepEP EP8/EP16 测试请参阅 [DEPLOYMENT.md](DEPLOYMENT.md)。
