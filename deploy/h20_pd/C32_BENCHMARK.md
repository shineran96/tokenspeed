# H20 C32 PD benchmark: non-DeepEP vs Humming DeepEP

## Scope
Historical 2026-11-16 measurements. This is a shared-prefix single-batch HTTP/SMG experiment, not the ten-round agent benchmark and not an exact reproduction of the webpage native gRPC setup. The webpage used DP16, real acceptance and explicit rank pinning; this experiment used DP8, fixed accepted length 4 and SMG automatic rank selection.

## Configuration and method
- One 8-GPU H20 P and one 8-GPU H20 D. P QCP8/EP8; D TP1/DP8/EP8. Host shared Engram; maximum context 1M; memory fraction 0.85; max requests 256.
- P chunk 16384, eager and prefix cache enabled. D chunk 8192, DSpark and CUDA Graph enabled, D prefix cache disabled. Requested Graph sizes: 1/2/4/8/16/32. CUDA_LAUNCH_BLOCKING unset.
- Model: DeepSeek-V4.1-Flash-1017. Baseline D: auto backend resolved to CUTLASS MXFP4 W4A16, all2all=none. Candidate: Humming W4A8 + Legacy DeepEP low_latency, FP8 dispatch / BF16 combine, capacity 256 token rows/rank.
- Both groups used the same P/SMG and synthetic input_ids. The source dataset has 64 inputs; C32 takes the first 32. Each input is 128928 tokens: 128800 shared prefix plus 128 distinct suffix tokens.
- inputs.json SHA256: 2da77d5525c47b7c07d27320688bfaed06ed48d92939b8e7a758375016eb1158.
- Output 2048 tokens, temperature=0, ignore_eos=true. Fixed accepted length=4 (one anchor + three accepted drafts; draft acceptance 60% in this configuration). Performance only, not accuracy.
- First prime the shared prefix with one request; then run one complete C32 warmup batch followed by three formal C32 batches. Wait for all P/D ranks to be idle before each batch. Keep prefix cache between batches; warmup is excluded.
- A client barrier releases all 32 requests. Finish the whole batch before the next one. Client-provided data_parallel_rank was not honored by this SMG; actual automatic routing was audited from scheduler traces.

## Metric definitions
- TTFT = first output event time minus request start.
- Per-request TPOT = (last output event time - first output event time) / (2048 - 1). This follows the N-1 convention even if the first SSE chunk includes multiple tokens.
- Each round reports request TPOT mean and P50; aggregate each metric with an arithmetic mean across three rounds. Mean of round P50 is not pooled P50.
- Actual output throughput = sum(output tokens) / (latest last-output time - earliest first-output time). Includes ramp-up and tail, excludes time before the first output in the batch.
- Per-D-GPU throughput = output throughput / 8; exclude P GPUs. Do not substitute concurrency / TPOT.

## Three-round means
| D backend | Successful requests | TTFT mean (s) | TPOT mean (ms) | Mean round TPOT P50 (ms) | Output (token/s) | Per D GPU (token/s/GPU) |
|---|---:|---:|---:|---:|---:|---:|
| pd-ep8-default | 96/96 | 1.905 | 7.352 | 7.373 | 3981.99 | 497.75 |
| pd-ep8-bf16 | 96/96 | 1.933 | 8.402 | 8.417 | 3543.22 | 442.90 |

## Individual formal rounds
| Backend | Round | TTFT (s) | TPOT mean (ms) | TPOT P50 (ms) | Output (token/s) | Balanced full-BS steps |
|---|---|---:|---:|---:|---:|---:|
| pd-ep8-bf16 | c32-round-1 | 1.913 | 8.418 | 8.433 | 3513.64 | 446 |
| pd-ep8-bf16 | c32-round-2 | 2.007 | 8.420 | 8.435 | 3512.91 | 446 |
| pd-ep8-bf16 | c32-round-3 | 1.878 | 8.367 | 8.383 | 3603.12 | 453 |
| pd-ep8-default | c32-round-1 | 1.928 | 7.368 | 7.390 | 3955.86 | 419 |
| pd-ep8-default | c32-round-2 | 1.886 | 7.353 | 7.373 | 3973.65 | 422 |
| pd-ep8-default | c32-round-3 | 1.901 | 7.335 | 7.357 | 4016.45 | 426 |

## Interpretation and provenance
All six formal batches reached [4,4,4,4,4,4,4,4] actual decode rank counts with no observed waiting queue. Each group completed 96/96 formal requests; reported cached input was 128768 tokens. A common full-BS period does not mean the entire output window stayed full.
Humming + DeepEP increased mean TPOT from 7.352 to 8.402 ms (about +14.3%) and reduced actual output throughput from 3981.99 to 3543.22 token/s (about -11.0%). This cannot be attributed solely to communication: backend implementation and activation precision also differ.
The historical experiment used the 459b4cc08 implementation base plus experimental runtime/dependency adaptations. These numbers are not an acceptance test of the exact consolidated commit or of the newly committed launch scripts. No uncommitted implementation is included here.
Archived evidence identifiers: local_work/humming-fp8-wire/{pd-ep8-default,pd-ep8-bf16}/web-bench/results/c32-round-{1,2,3}; WEB_BENCH_COMPARISON.json; per-service decode-global-batches.jsonl. These archive paths are provenance references, not portable runtime dependencies.
Machine-readable extracted summaries are committed beside this document as c32-results.json. Raw token datasets, request text, traces and dependency binaries are not included. Reproduction requires the exact token input hash and compatible dependencies; same lengths alone are insufficient.
