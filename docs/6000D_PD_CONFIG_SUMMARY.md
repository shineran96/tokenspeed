# RTX 6000D：DeepSeek-V4.1 Prefill / Decode 配置简表

范围：DeepSeek-V4.1-Flash-1017，按模型内部模块整理。依据 2026-11-10 最后完成验证的 PD 配置，runtime commit `d620ec2a`；不代表 11 月 17 日远端最新 main 或仍在线的服务。GLM 历史实验不混入本表。

## 部署配置

| 项目 | Prefill（每个 P） | Decode（一个逻辑 D） |
|---|---|---|
| GPU / 并行 | 单机 8 卡；QCP8 / EP8，DP1、KVP1；启动 TP8，dense/moe/attention-head TP 均为 1 | 两机 16 卡；DP16 / EP16、TP1 |
| 2P1D 总资源 | 两个独立 P，共 16 卡 | 一个跨两机 D，共 16 卡；全套 32 卡 |
| 最大上下文 | 1,048,576 token | 1,048,576 token |
| Chunk prefill | 16,640；max-prefill-tokens 同值 | 8,192（配置值，不代表承担主要 prefill） |
| max-num-seqs | 32 | 256；不等于实测同时 decode 数 |
| 显存比例 | 0.85 | 0.85 |
| Prefix cache | 开启；多轮固定会话到同一 P | 关闭 prefix caching，仍保留 decode 所需 KV |
| Engram | Host table，shared layout | Host table，shared layout |
| CUDA Graph | Prefill graph 关闭 | Decode graph 开启，本地 capture BS 1/2/4/8/16 |
| 投机解码 | 启用 DSPARK 配套路径 | DSPARK；性能测试模拟接受长度 4，draft 接受率约 60% |
| 公共开关 | BF16 dtype；关闭 PDL、KVStore；CUDA_LAUNCH_BLOCKING 未设置 | 同左；force-deterministic-rsag 开启（不代表每条路径必然执行 RSAG） |

## 各模块：库、实现与精度

配置证据来自上述已测版本；实现名称由本地 `deploy-review` 源码快照核对。条件分支仍受形状、布局与选择器影响，不能将后端配置等同所有算子的实际执行实现。

| 模块 | Prefill：库 / 实现 | Decode：库 / 实现 | 精度及优化说明 |
|---|---|---|---|
| Dense 投影 GEMM | DeepGEMM SM120 group-32 MXFP8 路径 | 同左；额外启用 TS Triton fused scale pack | E4M3 数据、UE8M0 scale；不满足 N/K 对齐的形状保留 Triton fallback，不是全部 GEMM 都走 DeepGEMM |
| 激活 scale 整理 | 原有 scale 准备 | TS 自有 Triton fused UE8M0 packing kernel | 融合转换、exp2 与转置/打包；不改变 GEMM 主计算 |
| Full Indexer 打分 | TS `deep_gemm_dsv41_index_topk`，启用 SM120 native FP4 | DeepGEMM FP4 Full-index logits（DG_INDEXER） | 属于稀疏位置筛选的打分，不能当作 attention 主体精度 |
| Reindex 打分 | 按 P 原有选择路径，不套用 D 实验配置 | TS Triton `_index_scan_kernel`，候选集合打分 | 已测版本未启用实验性 FP4 Reindex；Full 与 Reindex 必须分开描述 |
| TopK / 索引准备 | 与 native FP4 Indexer 配套 | TS 集成的 CuTe DSL DeepSelect；另有 Triton fused index pack、清理/映射 kernel | DeepSelect 是选择算法实现，不是通信库 DeepEP；适用形状之外保留原路径 |
| Prefill Attention | FlashInfer 0.7.2 `SparseMLASm120Wrapper`，TS 入口 `flashinfer_sm120.paged_prefill` | 不作为 D 主要 decode attention 路径 | `backend=sparse`、`compute_precision=bf16`；BF16 QK/PV、FP32 累加/softmax；SWA 临时 FP8 页，global 直接读原 FP4 页，不做全历史 BF16 gather |
| Decode Attention | 不套用 D 的 gather 开关 | TS 自有/迁入的 Triton gather-first selected attention，`SELECTED_ATTENTION=gather` | 所选 SWA FP8/global FP4 KV 反量化为 BF16 工作区；BF16 Tensor Core QK/PV，FP32 累加与 LSE merge；不是 FlashInfer 主 attention kernel，也不是 FP8/FP4 attention |
| wo_a 投影 | 默认 BF16 权重路径，未开启 MXFP8 实验开关 | TS `grouped_mxfp8_projection` 专用投影路径 | D 使用 MXFP8；不能仅由 dense-gemm-backend 推断 wo_a 的实际 kernel；与 attention QK/PV 精度独立 |
| Routed MoE 专家计算 | FlashInfer SM12x CuTe DSL fused MoE；TS prequant adapter | FlashInfer SM12x CuTe DSL kernel + TS `web_w4a8.online` 适配层（expert_impl=web） | MXFP4 权重、MXFP8 激活，W4A8；D 包含接收数据布局整理与 FC1/FC2 专家计算，并非直接复用 P 的全部 wrapper |
| MoE dispatch / combine | DeepEP Elastic：FP8 dispatch / BF16 combine，capacity 2048 | 定制 DeepEP 低延迟 wire1 + NVSHMEM：NVFP4 dispatch / FP8 combine，capacity 128 | D 开启 compact、cached-clean；这是通信格式，不代表专家 GEMM 是 W4A4；capacity 不等于请求 BS |
| Shared expert overlap | 未开启 D 专有开关 | TS runtime 的 shared-overlap 执行安排 | 属于调度/重叠优化，不是新的算子库；收益取决于实际并发与执行条件 |
| mHC | TS Triton `triton_mhc_pre` / `triton_mhc_mixes`；P 专用 MHC_MIXES 开关为 0 | 明确 override 为 TS Triton `triton_mhc_mixes` | 不将 P 的 MHC_PRE override 自动套给 D；未逐项确认的 D 子路径不声称已强制启用 |
| Hadamard | TS Triton `triton_hadamard_transform_128` | 同左 | 显式 kernel override |
| Engram | TS host-table，shared layout | 同左 | 表放在 host，节省 GPU 常驻权重显存；不是 KV cache offload |
| DSpark / CUDA Graph | TS DSpark 配套状态，prefill graph 关闭 | TS DSpark + CUDA Graph，本地 BS 1/2/4/8/16 | 模拟接受长度 4 仅用于本轮性能；精度/正常推理须取消模拟 |
| LM head | 默认路径 | 本轮未显式开启 LM-head TP4 | 不把历史独立 TP4 实验算进本次优化 |

### 关键源码入口

相对 TokenSpeed 仓库根目录：

- P attention：`tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/flashinfer_sm120.py`。
- FlashInfer 封装：`tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/flashinfer/sparse_mla_sm120.py`，明确设置 `compute_precision="bf16"`、`kv_cache_format="fp8"`、`extra_kv_fp4=has_global`。
- Indexer / selected attention 分发说明：`tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/README.md`。
- Dense GEMM / scale pack：`tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/README.md`、`thirdparty/deepgemm.py`（后者相对 kernel 包目录）。
- D MoE：`tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/deepep_sm120_experiment/low_latency_wire1.py` 与 `web_w4a8/online.py`。
- P MoE adapter：同目录 `prequant_fused_moe_opt.py`，调用 FlashInfer SM12x CuTe DSL FC1/激活量化与 FC2/finalize 实现。

**存储、通信、计算三种精度必须分开：KV 的 FP4/FP8、DeepEP 的 NVFP4/FP8，不意味着 attention 用 FP4/FP8 Tensor Core。** P 的分页 attention 只在支持的 KVP1/无 attention-head exchange 等条件下采用，其余布局保留原有路径。

## PD 协调与通信

| 项目 | 已验证配置 |
|---|---|
| 请求入口 | 修复 DP 均衡的 SMG；本轮未使用 Dynamo |
| P 分配 | 客户端按会话奇偶均分两个 P，各轮保持亲和；启动后通过日志核验 worker 索引映射 |
| D 分配 | SMG 的 DP rank 负载预占/释放；worker round_robin 与 rank 均衡不是同一层 |
| P→D KV | Mooncake，layerwise interval 0 |
| D EP 网络 | NVSHMEM / IBGDA，RDMA NIC 映射，配套 AH workaround；按目标机器配置 NIC/GID |
| NCCL 设置 | 本次 P2P、SHM 关闭，走配置的 IB 网卡；属于实测拓扑设置，不是所有机器的通用最优值 |
| 环境 | Python 3.12、Torch 2.14.0+cu130、CUDA 13、FlashInfer 0.7.2；P/D 隔离依赖，D 额外安装 wire1 |

## 验证边界

单 P C16/C32/C64、双 P C16/C32/C64/C128：100K prime 后完整 10 轮，每轮新增 10K、输出 1K，固定模拟接受长度 4。最终采用全部成功、前缀命中、正式区间无在线 JIT 的结果。模拟接受率仅用于性能，正常推理/精度测试须取消 `TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN`。

配置证据：[完整启动参数与环境变量](CURRENT_BEST_PD.md)。性能证据：[实验总结](SMG_SINGLE_DUAL_P_RESULTS.md)。Attention 精度另以本地保存的 kernel 源码及 PTX 核验，不将 KV 存储量化或 EP 通信格式当作 Tensor Core 计算类型。
