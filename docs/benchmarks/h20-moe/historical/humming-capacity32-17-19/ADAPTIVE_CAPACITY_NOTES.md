# DeepEP + Humming 随并发适配的设计记录

2026-11-23。此文为方案分析，尚未实现动态容量切换。

容量应该覆盖各 EP rank 实际送入 MoE 的最大 token 行数，包括 CUDA Graph padding；与注意力的 128K 上下文长度不是同一维度。当前 target verify 每请求 6 行，per-rank C1/C2/C4/C8/C16 分别为 6/12/24/48/96 行。可评估 8/16/32/64/128 的 bucket，但不是已验证支持的档位列表，必须检查 DeepEP 参数和内核的对齐限制。

当前 `DeepEPMode.low_latency` 固定通信布局读取 `DeepEPBuffer._num_max_dispatch_tokens_per_rank`。只改变每层 dispatcher 的 capacity 并不会缩小实际 receive 张量与 Humming 处理网格。独立进程分别建立 buffer 才能完成固定容量 A/B。

## 方案一：多个通信/Graph 档位

按服务支持的最大 batch 预建 bucket，运行时使用全 rank 一致的调度元数据选档；不得各 rank 自行独立选容量。target/draft 的 wire experts、counter offsets、buffer 生命周期和 replay 地址必须一致。每个档位需要稳定 buffer 地址及对应 CUDA Graph，避免捕获后替换底层存储。

优点是通信容量和 padded tensor 都能收缩；代价是显存、Graph 数量与跨档切换复杂度。当前 singleton DeepEPBuffer 不能直接满足多实例独立管理，需要修改封装及验证 cleanup、counter reset、在途操作和 graph replay 的隔离。

## 方案二：通信固定上限，量化只处理有效行（优先）

保留足够的固定通信容量以及地址，Humming 的 SwiGLU/FP8 process_input kernel 读取 GPU expert counts，只为有效行分配实际任务。可采用每专家固定 worker 循环或 device 任务队列，避免按照 `local_experts * capacity * EP_size` 给每个空行启动 CTA。

Graph 内 grid 可以固定为较少的 worker 数，实际循环次数随设备端 counts 变化，不需要把 counts 同步回 CPU。通信 buffer 的上限仍要满足所有请求形态，这个方案只让有效计算自适应，不是无限容量。

必须确保读取 counts 前完成 DeepEP/PDL 依赖、CTA barrier 参与一致、FP8 scale 输出索引与原 padded layout 相容。核对 FC2 仅消费有效行以及 padding 的清零约定，防止高并发回落后读取历史数据。输入路由的跨 rank 偏斜不能用平均 tokens/expert 代替真实 counts。

## 验收范围

- 全 rank 同步的并发档位变化，高→低→高，target/draft 交替。
- 单 rank 最大输入、全部路由集中同一组专家、部分 rank idle、全 idle 与恢复。
- eager、Graph 与 PDL 开关；动态有效行改变但指针不变。
- 数值对比应记录已有 Humming 非逐位确定性，不以无理由放宽容差代替验证。
- C1/C2/C4/C8/C16 per-rank 服务性能，实际满 batch、接受长度和 profile 分开核对。
