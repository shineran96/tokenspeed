# Multi-node query context parallelism (MQCP)

MQCP extends one prefill request across nodes while preserving a complete
node-local model split by TP and EP. `--mqcp-size 2 --tensor-parallel-size 8
--query-context-parallel-size 8 --enable-expert-parallel --world-size 16
--nnodes 2` gives two local TP8/QCP8/EP8 groups and one request replica.
MQCP is not attention DP and does not shard experts across nodes.

The implementation targets DeepSeek V4.1 eager prefill with KVP1 or KVP
equal to the global query width, GPU Engram, no mixed batch, and no pipeline
parallelism. Decode remains
on a separate ordinary worker. Omission of `--mqcp-size` leaves the existing
single-node QCP behavior unchanged.

The global query plan assigns each input row once using the existing
per-request zigzag layout. CED selection preserves ownership, including
empty ranks. Embedding, Dense and MoE use only their node's row counts.
Attention keeps global positions, causal visibility and complete KV history.
The existing cache writer and compression semantics remain authoritative.

Projected KV and compressor rows gather directly across the global query
group, then return to input-span order before cache writes or pooling.
The model uses this flat gather for both source and query spans. The gather
helper also supports explicit cross-node then local groups for collective
comparisons; that hierarchical path is not selected by the model.
Projected-row gathers carry the current forward's increments, not historical
KV. KVP additionally gathers packed history from its page owners. Compressor
windows crossing rank boundaries retain their original meaning, and projected
rows need not have the final quantized cache representation.

The host scheduler replica spans all MQCP ranks. PD advertises that full
replica width for cache-transfer coordination, while weight TP remains local.
With KVP1 the complete cache is replicated; with global-query KVP the
existing PD transfer planner selects the page owners across the full replica. Sampled rows and DSpark taps
are gathered globally before local LM-head and draft processing.

Chunk size is explicit and never multiplied implicitly. Global chunks 16384
and 32768 give approximately 1024 and 2048 query rows per GPU for MQCP2/QCP8.
A collective MoE workspace must cover the largest local query shard.

Validation must include odd and short spans, empty CED ranks, multi-request
boundaries, real two-node collectives, single-node output comparisons, PD
transfer and cold-cache TTFT. CPU topology tests alone do not establish
end-to-end correctness or performance. The measurements below record the checks completed for these configurations.

Implementation validation so far:

- Two-node, sixteen-GPU NCCL gather checks passed for BF16, FP32, int64
  and uint8, including odd spans, multiple requests and empty CED ranks.
- Logits and MQCP configuration tests passed (270 tests), including a query
  group spanning two independent vocabulary TP groups.
- Real sixteen-GPU embedding and Engram lookups matched full-table results;
  local MoE gather/reduce-scatter round trips passed, including empty ranks.
- Full-model startup and PD requests passed for 1, 17, 8193, 16385 and 32769
  input tokens, each returning 32 output tokens with no cache hits.
- First generated tokens matched single-node QCP8 for all five smoke inputs.
  Under the requested first-token acceptance criterion, these cases pass.
  The 32-token outputs differed for 8193 and 16385 input tokens (first
  differences at output positions 4 and 3); the other three matched fully.
  Repeated single-node outputs were stable. Full-sequence equivalence is
  not established. Chunk-size performance results follow below.

RDMA configuration must expose NICs reachable from every participating GPU.
A fixed NIC can register memory yet fail GPU-origin writes with a local
protection error. Validate writes and destination contents on every GPU;
a successful GPU0 test does not validate the other devices. In the initial
two-node setup, exposing all eight NICs passed eight GPU-to-D write checks.
This transport check does not substitute for full-cache PD validation.

## Measured prefill performance

Measured on 2026-11-24 with two eight-H20 P nodes and a four-H20 D node,
DeepSeek V4.1, eager execution, target native SM90 CuTeDSL MegaMoE, draft
CUTLASS W4A16, GPU Engram, PDL and autotuning enabled. C1 synthetic
pretokenized input, prefix cache disabled, one output token; each input
length has one warmup and three measured requests. Times are arithmetic
means in seconds. The single-node baseline uses QCP8 and a 16K chunk.

| Input tokens | Single P prefill | MQCP 16K prefill | MQCP 32K prefill | Single client TTFT | MQCP 16K TTFT | MQCP 32K TTFT |
| --- | --- | --- | --- | --- | --- | --- |
| 131072 | 2.953 | 1.702 | 1.612 | 3.031 | 1.778 | 1.682 |
| 262144 | 6.000 | 3.413 | 3.171 | 6.142 | 3.561 | 3.299 |
| 524288 | 12.914 | 7.332 | 7.019 | 13.186 | 7.600 | 7.259 |
| 1048576 | 33.700 | 18.748 | 17.584 | 34.212 | 19.304 | 18.081 |

Both MQCP settings completed all 16 requests without cache hits or preemption;
all first tokens matched the corresponding baseline. This is the accepted
first-token criterion, not a claim of task accuracy or full-sequence parity.
P prefill time comes from scheduler request statistics; client TTFT includes
request handling, transfer and D first-token delivery. Their difference is
not a measurement of RDMA alone. Startup/autotuning is excluded.

For 1M input, the 32K chunk reduces P prefill time by 47.8% versus the
single-node baseline and 6.2% versus MQCP 16K. The measured 18.08-second
client TTFT does not meet the six-second target. A direct CP16 gather
comparison follows below.

## Direct CP16 gather comparison

The direct-gather implementation retains node-local TP8/EP8 and the same global
query ownership. Source and query KV rows use the existing flat gather
over the sixteen-rank query group instead of the two-level gather.
This is not TP16. Both configurations use chunk 32768, C1, cold cache and
one output token; each length has one warmup and three measured requests.

| Input | Two-level prefill s | Direct CP16 prefill s | Direct change | Two-level TTFT s | Direct TTFT s |
| --- | --- | --- | --- | --- | --- |
| 131072 | 1.612 | 1.598 | -0.90% | 1.682 | 1.677 |
| 262144 | 3.171 | 3.200 | +0.93% | 3.299 | 3.327 |
| 524288 | 7.019 | 6.824 | -2.77% | 7.259 | 7.089 |
| 1048576 | 17.584 | 17.644 | +0.35% | 18.081 | 18.141 |

All sixteen requests passed first-token checks against the single-node
baseline with no cache hits or preemption. These sequential three-round
runs do not establish a consistent advantage for either communication
layout. No isolated communication timing or NCCL algorithm trace was
collected; network traffic and overlap cannot be inferred from TTFT alone.

## KVP over the MQCP replica

Weight TP and EP remain node-local. The cache owner group can span the full
request replica: TP8/QCP8 with MQCP2 accepts KVP1 or KVP16. Partial owner
groups remain rejected. `--kv-parallel-size 16` does not increase the world
size or turn the model's weight groups into TP16/EP16.

The page owner and replica coordinates are derived from the complete request
width, not local weight TP. Query configuration reports that same global
query group. Global KV and indexer K retain the existing page-cyclic cache
placement, while SWA, compressor tails and draft windows remain replicated.
The existing packed-history gather transports owned pages over the sixteen
rank KVP group; the model's projected KV gather uses the flat query group.
No intermediate dequantization or second cache ownership rule is introduced.
PD continues to advertise the whole request replica, so its sender planner
can locate all sixteen page owners.

KVP16 is opt-in (`--kv-parallel-size 16`); KVP1 remains the default.
The two-node validation utility `test/runtime/distributed/mqcp_kvp_collective_check.py`
checks projected rows in BF16, FP32, int64 and uint8, plus strided packed KV
pages with partial pages and empty owners. All twenty byte-exact checks
passed on each of sixteen ranks. Five short/odd-length PD requests and all
twelve long-input requests (including warmups) matched the KVP1 first token.
This does not establish full-sequence equivalence.

The following KVP16 run used the same local TP8/EP8, chunk 32768, C1 cold
cache and one output token. Each length had one warmup and three measured
requests; times are arithmetic means. The KVP1 figures are historical,
not an alternating paired control. KVP16 used an external gRPC bridge,
whereas the historical run used direct ZMQ routing.

| Input tokens | KVP1 prefill s | KVP16 prefill s | KVP1 client TTFT s | KVP16 client TTFT s |
| --- | --- | --- | --- | --- |
| 131072 | 1.598 | 1.634 | 1.677 | 1.711 |
| 524288 | 6.824 | 7.172 | 7.089 | 7.427 |
| 1048576 | 17.644 | 18.975 | 18.141 | 19.573 |

KVP16 did not improve latency in this run: at 1M, prefill increased by 7.5%
and client TTFT by 7.9%. Its benefit is persistent global-KV capacity, with
additional history communication. Each P rank reserved a 28.882 GiB KV arena
and 13.619 GiB fixed workspace. At 1M, per-GPU total memory peaks ranged from
126.06 to 126.33 GiB; observed active LCM pages corresponded to 1.303 GiB per
rank. Active-page capacity includes page-internal reservations and is not
the valid KV payload. It was inferred from rank-zero load metrics with an
approximately one-second refresh, not a simultaneous per-rank peak census.
