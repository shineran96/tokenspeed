---
name: tokenspeed-pd-prefix-bench
description: 使用 LongBench 来源数据进行 TokenSpeed PD 十轮递增前缀压测，100K prime 后独立会话每轮新增 10K，统计缓存命中、TTFT、TPOT、实测吞吐、decode batch 和 DSpark 接受率。不用于网页单批次测试或模型精度评测。
---

# TokenSpeed PD 十轮递增前缀压测

用户指定配置优先。默认沿用已验证的十轮协议，先读 [数据](references/data.md) 和 [执行与统计](references/execution.md)。

- 1K=1024；100K prime 后正式输入依次为 110K–200K，共十轮，每请求输出 1K。各轮取同一序列更长切片，不拼接上一轮生成输出。
- prime 最多 8 条同时发出；全部会话 prime 完成后一次全局屏障。之后各会话上一轮完成立即发下一轮，不等待同轮其他会话。
- temperature=0、seed=37、ignore_eos=true。性能对照可沿用服务端模拟接受长度 4（5 个 draft 中接受 3 个，再加 target token，目标 draft 接受率 60%）；末步截断后约 59.92%。必须核验运行配置和计数，不能假定客户端已设置服务端开关。精度评测必须关闭模拟。
- 默认 C1/C2/C4/C8/C16/C32；资源允许时按用户要求增加 C64 及以上。不得默默循环复用会话补足并发；需要更多来源或独立会话标识时记录构造变化及跨会话共享前缀。
- P 开启 prefix cache，D 是否关闭 prefix 匹配按当前部署核验。每档先在服务空闲时清 KV，并检查 P 确实清理成功；保留 JIT 缓存。不要仅凭 HTTP 200 宣称清理成功。
- 100K prime 用于缓存，不等于充分编译预热。完整预热是否执行须记录；首次新形状可能在线 JIT。正式时间窗发现编译时标注并定位，不静默删首轮、伪称稳态或直接改 warmup 策略。
- 每轮预期缓存为上一轮输入长度。SWA/压缩器可能重放约 128 token；结合响应 cached_tokens 和调度日志判断。缓存驱逐造成重算必须单独报告。
- 单 P 保持同一 P；双 P 按会话均分并保持亲和。SMG worker 索引必须由当前注册表/日志验证，不能复用旧机器的索引。D DP 均衡与 P worker 选择是不同机制。
- 全部请求成功、输出长度及缓存校验后才发布完整结果；失败保留现场，不自动重启服务推进。此 skill 不额外授权重启、安装或停止其他服务。

## 执行资源与边界

`original/bench_v2.py` 和 `original/client_v2.py` 保留历史实测脚本供参考，**不是通用即用入口**：client 写死三个 worker 及 P 索引映射。迁移前按执行文档核对接口、映射和配置，在新实验目录调整副本，保存差异。

完整十轮输入已归档为 `data/sessions128.json.gz`，解压和哈希验证方法见 references/data.md。不要将四轮数据当成十轮输入。

## 交付

保存原始逐请求事件、配置/commit、输入哈希、逐轮及全十轮统计、JIT 与缓存证据。实际 decode 最大 BS 来自同步 DP metadata，不能用网关 running 数或各 rank 不同时刻峰值之和代替。缺失指标标记缺失。

部署背景见仓库 `docs/6000D_PD_CONFIG_SUMMARY.md`；单独安装此 skill 时该文档为可选背景，执行所需口径已在本目录 references 中说明。
