# 执行与统计

## 历史脚本迁移

original/ 中两个脚本保持原始字节。依赖 Python 和 requests；需要 SMG 的 /generate raw-token SSE 接口以及 /workers。

迁移到新实验目录时检查 client_v2.py：
- wait_ready 写死 3 个 worker，不适用于任意部署；按当前注册表校验 P/D 数量与健康状态。
- P 选择写死 `1-(index % PREFILL_COUNT)`。原实验 index 1 和 0 对应两台 P；先验证当前映射再调整，不能直接运行历史映射。
- BENCH_URL、BENCH_OUTPUT、SESSIONS_FILE 由环境提供，不复用原机器地址。

bench_v2.py 读取 PRIME_TOKENS=102400、CHUNK_TOKENS=10240、ROUNDS=10、OUTPUT_TOKENS=1024、PRIME_BARRIER=1；PREFILL_COUNT 为本次 P 数量。参数为 --concurrency 和 --tag。输出目录必须新建，脚本拒绝覆盖。

脚本不自动清 KV、不设置服务端模拟接受率、不完整预热，也不采集实际 GPU decode BS。须在运行前配置和验证这些条件。prime 也生成 1K，但统计正式十轮时排除 turn=0。

可只做 prime；此时不得声称已充分 JIT 预热。若选择完整编译预热，覆盖目标路径/形状并单独归档，结束后空闲清 KV，再 prime 和正式运行。两组采用相同预热政策。

## 时间与吞吐

每请求：TTFT=首个输出计数增加的事件时间−请求开始；TPOT=1000×(最后输出事件−首次输出事件)/(输出数−1)，单位 ms/token。一个 SSE 事件可含多个 token，不能等同逐 token ITL。保留 first_event_tokens。

正式集合为全部十轮成功请求，不含 prime。报告逐轮及全体请求的 mean/p50/p95，不静默排除首轮。

- 单请求 TPS = 1000 / 全部正式请求平均 TPOT(ms)。不是网页测试的“三轮 P50 均值”口径。
- 实测输出窗口吞吐 = Σ正式输出 tokens / (最晚输出事件−最早输出事件)。包含后续轮次 prefill、排队、爬升和尾部，不含 prime，不能称纯满 BS decode 吞吐。
- 单 D 卡吞吐 = 上述吞吐 / D GPU 数；不计 P GPU，但报告硬件资源。
- 全阶段输出吞吐 = Σ正式输出 tokens / (最后请求结束−最早正式请求开始)，包含首个请求 TTFT。必须与输出窗口口径分开命名。
- 输入吞吐 = Σ正式输入 tokens / 全阶段耗时；未缓存输入吞吐 = Σ(input_tokens−cached_tokens) / 同一耗时。两者分子都不加输出、不包含 prime；cached_tokens 缺失时不能用理论 10K 替代。

异步会话的同轮时间窗可能相互重叠，不能把各轮耗时简单相加当作总时长。全十轮应从原始时间戳统一计算。

## 验证

保存 request meta.id、输入/输出数、cached_tokens、事件时间以及 P/D JIT、驱逐、抢占日志。历史脚本允许某些 cache shortfall 并记录，success 不代表完整命中；汇总时单独审计。

最大 decode BS 从同一步 DP metadata 的各 rank decode 数求和并在正式窗口取最大值；网关 running/HTTP 活跃请求只是代理指标。记录观测成本，缺失则填不可用。

DSpark 当前配置每步 draft 5 tokens。若日志 L=(N−1)/verify_count 包括 target token，draft 接受率=(L−1)/5；组加权为 Σ(N−1−V)/(5ΣV)。有原始计数优先使用原始计数；仅有四舍五入 L 时不能伪造精确 V。模拟长度 4 只用于性能，输出不用于精度结论。

P→D 是否增量传输不能由 P prefix 命中推断。KV 记录 active、cached/retained、总量及最忙 rank，网关零值不能当真实 KV 为零。
