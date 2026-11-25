# H20 MegaMoE 与 DeepEP + Humming 压测记录

记录 2026-11-22～24 的 DeepSeek-V4.1-Flash-1017 测试方法、配置、结果和证据。本文归档既有实验，没有重新启动性能测试。主结果来自 11-24 的 staged decode：先让 P 完成全部请求的 KV 传输，再同时释放 D，减少 P 入场速度对 decode 吞吐的影响。TPOT 采用客户端逐请求算术平均，不使用满 batch TPOT。

## 后端和版本

|名称|实际实现|通信与分片|Draft|
|---|---|---|---|
|本篇主表 MegaMoE|FlashInfer 0.7.3 `fp8_mxfp4_bf16_pull_cutedsl`，TS `--moe-backend mega_moe`|单机 DP8/EP8；双机为两套独立副本，不是跨机 EP16 MegaMoE|继承 MegaMoE，与网页默认一致|
|本篇主表 DeepEP + Humming|Humming MXFP4 权重/FP8 激活，GPU 有效行自适应调度|两机 DP16/EP16，DeepEP low_latency|同为 Humming + DeepEP|
|11-22 原生 MegaMoE 历史组|TS `flashinfer_native`，与上述 CuTeDSL 入口不同|单机 DP8/EP8，无 DeepEP|CUTLASS W4A8 混合配置|

主表 TokenSpeed 源版本为 `345481b592a2f909953c63ca93251d4f8b73cee5`，Humming 优化版本为 `219fc5061738dd5a0f7bfc92d3d3c9d36cd773ae`；SMG 使用 `ae2139e` 对应的 1.13.0。主表没有启用后续新增 SWA KV RMSNorm 融合。当前分支已经包含后续改动，不能将历史成绩标作当前 HEAD 的新实测。

Humming 和 FlashInfer CUTLASS 是不同后端，不能因为都使用 W4A8 就合并命名。历史模拟接受长度 5.5/5.61 的数据也不能与本篇固定 4.0 的主表混算。

## 主实验配置

- P：tx-04，4 张 H20，TP4/QCP4/EP4，KVP1，eager，chunk 8192，prefix cache 开启。
- D：单副本 MegaMoE 使用 tx-01 的8卡；双副本使用 tx-01/tx-03 共16卡；Humming EP16 使用同两台共16卡。D 为 TP1，Engram host table/shared，GPU memory utilization 0.85，CUDA Graph 与 PDL 开启。
- 输入 131072 tokens；所有请求共享130816-token前缀，每请求256-token独立后缀。输出2048 tokens，ignore_eos。正式请求实际 cached_tokens 均为130816。
- 总并发 C32/C64/C128/C256；每档一次形状预热、三次正式轮次。双副本由网关分配总并发，不能向每套副本各发送完整 C。
- DSpark 模拟接受率60%，实际固定接受长度4.0；启动环境 `TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN=4.0`。这不是模型真实接受率或质量评估。
- PD 使用 Mooncake；DeepEP GPU NIC handler 使用 `NVSHMEM_IBGDA_NIC_HANDLER=gpu`。必须按实际 GPU/NIC 拓扑校验通信，不能从环境变量存在推断全部 GPU 已成功走该路径。

配置与源码快照见 [snapshots](snapshots/)，每组机器可读设置见 [data/staged-decode-20261124](data/staged-decode-20261124/)。快照保留历史 IP、端口、目录和依赖，不是可在任意机器直接执行的通用启动器。

|D 方案|关键参数|
|---|---|
|单机/每套 MegaMoE|`--tensor-parallel-size 1 --data-parallel-size 8 --world-size 8 --enable-expert-parallel --moe-backend mega_moe --all2all-backend none`|
|跨机 Humming|`--tensor-parallel-size 1 --data-parallel-size 16 --world-size 16 --nnodes 2 --node-rank <0或1> --enable-expert-parallel --moe-backend humming --all2all-backend deepep --deepep-mode low_latency --moe-mxfp4-fp8-activation --low-latency-max-num-tokens-per-gpu 256`|

单机 MegaMoE C32～C128 使用 `max-num-seqs=128`、Graph sizes `1 2 4 8 16`、MegaMoE capacity96；C256 使用 `max-num-seqs=256`、Graph sizes增加32、capacity192。Target verify 每请求6行，因此单机 C256 每 rank32请求需要192行容量。历史 Humming 调小容量32的方案仅适用于其固定小 batch，不能直接用于此处全部并发档位。

## staged decode 的执行顺序

1. 恢复历史源码/依赖与精确输入，启动 P、D 和启用 RL 控制的 SMG。单机与双副本的工作节点数量分别为1P+1D和1P+2D；EP16是一个跨机逻辑D实例。
2. 用共享输入做 prime，预热相应 Graph/形状。每档正式轮次的结果与 warmup 分开保存。
3. 暂停 P，客户端 barrier 同时发出 C 个请求；检查 `/loads` 的 D rank 入队数：单 DP8 为每rank `C/8`，双 DP8和DP16为 `C/16`。
4. 等待5秒 bootstrap grace，再暂停 D、恢复 P。该5秒只用于 bootstrap，不是 KV 就绪条件。
5. 持续检查 P 的 `RequestStats(status='finished')`，在当时版本中它在 KV transfer完成后产生。要求本阶段恰好 C 个完成请求，再释放 D；双副本要同时释放。
6. 等待全部请求成功、恰好输出2048 tokens。异常也必须恢复本轮暂停的 worker。保存 `control.json`、逐请求 SSE 事件/累计 token、元信息、DP batch trace和P/D日志。
7. 审计首个输出晚于 `all_transfers_completed`，cache/prompt/output计数正确，存在共同 decode 窗口。实际 batch 由独立 trace核对；最终 TPOT 不裁剪为满 batch 窗口。

双副本的早期 `results-attempt1` 有 bootstrap race/提前释放，已排除。主表只使用 `results/c*/formal1..3`，不挑选最快轮次。staging 人为增加等待，**其 TTFT 不能代表正常端到端 TTFT**。

### 统计公式

单请求 `TPOT_ms = (最后输出事件时间 - 首个输出事件时间) × 1000 / (2048 - 1)`；先平均全部 C 请求，再平均三轮。沿用当时客户端公式，首事件通常带5个tokens，不能偷换成按首末事件实际 token 增量计算的另一口径。

每轮输出吞吐 `C × 2048 / (最晚末输出事件 - 最早首输出事件)`，最终平均三轮。它包含请求入场/退出差异，不等于 `C × 1000 / 平均TPOT`。单卡吞吐再除以实际 D GPU 数（单机8，其余16），不计P卡；比较总吞吐时需同时说明卡数。

## 主结果：客户端平均 TPOT / 实例输出吞吐

下表每格为 `TPOT(ms) / output token/s`，不是每rank吞吐。

|总并发|单机 MegaMoE（8卡）|双副本 MegaMoE（16卡）|DeepEP + Humming EP16（16卡）|
|---|---:|---:|---:|
|C32|6.274 / 5077.9|5.071 / 6248.0|5.834 / 5458.3|
|C64|8.229 / 7721.8|6.133 / 10328.4|7.781 / 8156.2|
|C128|13.168 / 9615.7|8.196 / 15322.4|11.532 / 10952.6|
|C256|20.612 / 12233.0|13.096 / 19095.3|18.532 / 13569.0|

同16张D卡和相同总并发下，双副本 MegaMoE 相对 Humming EP16 的吞吐优势依次为14.47%、26.63%、39.90%、40.73%。这是后端加分片方式的服务方案对比，不能解释成单个 MoE kernel 的加速比。

保持每rank batch相同，双副本 C64/128/256 对单机 C32/64/128 的吞吐比分别为2.034/1.984/1.986。第一个比值略高于2，不代表确定的超线性收益；这些是顺序执行的有限轮次实测。具体见 [matched-rank-batch-scaling.json](data/staged-decode-20261124/matched-rank-batch-scaling.json)。

与正常PD双副本基线相比，其 C32/C64/C128/C256 输出吞吐为5421.7/8044.7/11069.6/12922.2 token/s；staged组为6248.0/10328.4/15322.4/19095.3。此差异说明 P/KV/请求入场节奏会影响整轮吞吐，不能视为计算后端本身升级。正常PD数据见 [client-mean-comparison.json](data/tx04-cutedsl-baseline/client-mean-comparison.json)。

## Profile 与历史优化

11-24配对profile使用单机 MegaMoE C32和跨机 Humming C64，均为每DP rank4请求。均值 Graph跨度约25.623/31.618ms，target MoE路径约14.621/19.558ms，draft跨度约1.804/2.897ms。Humming target combine累计约8.285ms、dispatch约2.086ms，但通信kernel包含同步等待，不能全算作纯网络传输。路径跨度与kernel累计可能重叠，不能相加替代整步时间。Profile不计入上面的三轮服务成绩。[profile-summary.csv](data/profile-paired-20261124/profile-summary.csv) 与 [profile-summary.json](data/profile-paired-20261124/profile-summary.json) 保留统计。

历史记录单独保存，指标不与主表交叉比较：

|实验|口径与用途|记录|
|---|---|---|
|11-22 单机四后端|C32、128K→1K、接受长度4、PDL off；满batch三轮中位数|[RESULTS](historical/RESULTS.md)|
|GPU NIC handler|开关及实际通信验证|[GPU_HANDLER_RESULTS](historical/GPU_HANDLER_RESULTS.md)|
|原生 MegaMoE拆分|原生入口的quant/staging/core，不是CuTeDSL主表|[MEGAMOE_PROFILE](historical/MEGAMOE_PROFILE.md)|
|Humming容量256→32|C32、128K→4K、接受长度5.5；共同输出中间80%窗口；吞吐5232→6496 token/s|[HUMMING_CAPACITY_C32](historical/HUMMING_CAPACITY_C32.md)|
|Humming有效行自适应|保留容量256；同历史C32窗口口径，吞吐6399.7 token/s，相比原版+22.32%|[HUMMING_ADAPTIVE_ROWS](historical/HUMMING_ADAPTIVE_ROWS.md)|
|其他原生/接受长度实验|量化/PDL/接受长度条件随实验变化|[historical](historical/)|

固定容量32不是自动适配方案；有效行自适应减少 padded 空行计算，但不自动缩小通信buffer，也不允许实际输入或Graph padding超过容量。

## 离线复算与归档恢复

完整原始材料已经归档到：

```text
viewfs://hadoop-meituan/user/hadoop-efficient-llm/wangchangran/archives/tx01-tx03-20261125/
```

23机器另有共享存储副本：

```text
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/wangchangran/archives/tx01-tx03-20261125/
```

`local-reports-and-plots.tar.gz` 内含本篇原始结果包；`tx01-codes-data.tar.gz` / `tx03-codes-data.tar.gz` 含完整启动树、环境脚本、精确 `sessions.json`、observer及服务日志。运行依赖采用共享的 `tx03-python-runtime.tar.gz`，恢复说明见归档 README。模型在共享CFS，没有重复打包。全部HDFS包已读回校验，归档有 `COMPLETE` 标记。

下载并解出原始结果后，可在CPU环境使用Python标准库复算，无需GPU或启动服务：

```bash
hadoop fs -get /user/hadoop-efficient-llm/wangchangran/archives/tx01-tx03-20261125/local-reports-and-plots.tar.gz .
hadoop fs -get /user/hadoop-efficient-llm/wangchangran/archives/tx01-tx03-20261125/local-reports-and-plots.sha256 .
sha256sum -c local-reports-and-plots.sha256
tar -xzf local-reports-and-plots.tar.gz h20-megamoe-bench-20261122/staged-decode-20261124
python3 docs/benchmarks/h20-moe/tools/recompute_staged.py \
  --raw-dir h20-megamoe-bench-20261122/staged-decode-20261124 \
  --output recomputed-staged.json
```

复算工具先校验三个内层原始包的SHA-256，再检查36轮/4320个正式请求、KV就绪门槛、token/cache计数和事件单调性，并从SSE时间戳重新计算TPOT/吞吐，与提交的comparison.csv比较。本次全部通过，记录见 [recomputed-staged.json](data/recomputed-staged.json)。它不替代batch trace或模型精度评估；batch审计另存于每组 `*-batch-audit.json`。

在线复测需先恢复完整目录树、历史运行版本及会话输入，再按新节点改好路径、IP、RPC端口与网卡。`snapshots/*/run.py` 会暂停/恢复配置中的P/D，因此只应指向专属测试服务。示意调用（在恢复后的对应实验目录执行）：

```bash
# 先启动所需P/D和启用RL的网关，完成共享前缀与形状预热。
BENCH_URL=http://<test-gateway>:28500 C=32 PHASE=formal1 python run.py
# 每个C执行formal1/2/3；单机C256先采用decode256.sh对应容量和Graph设置。
```

历史 `prime.py`、`suite.py` 及 `common.sh` 也可能固定网关地址，不能只改一个 `BENCH_URL` 就假定全套迁移完成。`snapshots` 是审核和复原依据；完整可复原树以归档为准。历史服务PID文件不可用于新机器的停止操作。
