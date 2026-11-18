# H20 DeepSeek-V4.1 DeepEP branch handoff

## Scope and baseline

This document covers committed code only on `perf/h20-deepep-consolidated`.
The implementation snapshot is `d26d33d2e`, based on community main
`a134936ae`. It is an experimental integration branch, not a claim that
all backends are ready for production.

The previous eight-commit series was consolidated into five commits without
changing its final Git tree. The original history is retained on
`archive/h20-before-commit-cleanup-20261118`. Other experiment branches are
not prerequisites for understanding this series.

## Commit map

| Commit | Responsibility |
|---|---|
| `989863a25` | Ordinary attention DP with EP; exchange CED row counts across active/idle ranks and preserve multimodal routing masks. |
| `bd436e02d` | Keep Hopper prefill indexing on the per-request path; restrict batching to supported native kernels and reduce reserved packed history. |
| `8e7993340` | Integrate DeepEP expert communication, shared low-latency layouts and CUTLASS MXFP4 receive packing. |
| `ccffff526` | Optional experimental Humming W4A8 backend. |
| `d26d33d2e` | Fixed target/draft wire layouts and fused CUTLASS receive packing. |

## Code reading guide

Paths below are relative to this repository.

- `python/tokenspeed/runtime/models/deepseek_v41.py`: DP-aware model execution,
  CED metadata and active/idle rank handling.
- `python/tokenspeed/runtime/models/deepseek_v41_moe.py`: expert collective
  integration and shared-expert execution; ranks must agree on communication mode.
- `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`,
  `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/deepseek_v41.py`,
  and `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/__init__.py`:
  Hopper indexer selection and associated allocation sizing.
- `tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/deep_ep.py`:
  shared buffer compatibility, logical-to-wire expert remapping and combine
  output shape restoration.
- `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/deep_ep.py`:
  MXFP4 dispatcher selection and transport activation settings.
- `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/flashinfer.py`:
  Legacy DeepEP to CUTLASS W4A16 adapter.
- `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/_deep_ep/layout.py`:
  fused device-side packing of valid rows, global expert IDs and unit weights.
- `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/humming.py`:
  backend registration, weight preparation and FP8 dispatch → masked experts →
  BF16 combine.
- `tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/humming.py`:
  optional Humming package adapter and FC1/activation/FC2 execution.
- `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/_deep_ep/humming_scale.py`:
  block-128 FP8 scale layout conversion for valid expert rows.

See also [DeepEP design notes](deepseek-v41-deepep.md) and the
[MoE backend README](../../tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/README.md).

## Data path and invariants

1. All ranks participate in expert communication, including ranks without
   local requests.
2. Dispatch routes tokens to expert ranks. The CUTLASS adapter uses BF16 wire
   activations; Humming uses FP8 dispatch.
3. CUTLASS compacts valid expert rows into graph-safe bounded storage.
   Humming consumes masked expert rows and device-side counts.
4. Expert computation produces unweighted outputs. DeepEP combine applies
   the original routing weights exactly once.
5. Low-latency-only mode retains one wire expert layout/capacity across target
   and draft. Smaller logical layouts remap expert IDs within each rank.
   Combine restores the wire shape. Normal/automatic modes retain layout
   preparation rather than inheriting this assumption.

Preserve these invariants when optimizing: no host synchronization to obtain
exact receive counts during Graph replay; mask unused capacity; preserve
target/draft counter addressing; do not apply routing weights twice.

## Supported paths and limitations

| Adapter | Architecture | Dispatch | Compute | Combine |
|---|---|---|---|---|
| Legacy DeepEP + CUTLASS MXFP4 | SM90 integration | BF16 | W4A16 | BF16 |
| Legacy DeepEP + Humming | SM90 registration | FP8, block 128 | W4A8 | BF16 |

Humming is optional and was developed against
`d0c7af98d7f556132f86e671de7f397991e2df2b`. It requires precomputed routes,
standard clamped SwiGLU and low-latency mode. The committed implementation
does not provide an FP8 combine selector.

Humming's default arithmetic can vary with layout. The original commit
explicitly records strict layout-reference tolerance failures; diagnostic
batch-invariant checks do not establish default-path model accuracy.
Do not treat integration or microbenchmark success as full accuracy acceptance.

This TS branch does not vendor a complete DeepEP/NVSHMEM dependency build.
A fresh environment must independently verify their compatibility, particularly
CPU-assisted notification and cross-node completion. No portable environment
reproduction is claimed by this handoff.

## Validation and next steps

Relevant committed tests:

- `test/runtime/test_deepseek_v41_model.py`
- `test/runtime/test_deepseek_v41_cache.py`
- `tokenspeed-kernel/test/nvidia/ops/test_deepep_prepare.py`
- `tokenspeed-kernel/test/nvidia/ops/moe/test_flashinfer_cutlass_mxfp4.py`
- `tokenspeed-kernel/test/nvidia/ops/moe/test_humming_scale.py`

Read each test's distributed/device prerequisites before execution. These
files cover targeted behavior; their existence is not evidence that the entire
branch passes on a new machine. History consolidation verified tree equality,
not a fresh runtime test run.

Recommended acceptance sequence:

1. Pin TS, Humming, DeepEP, NVSHMEM, CUDA and framework versions.
2. Validate target/draft transitions, empty/skewed ranks, changing inputs and
   repeated CUDA Graph replay against independent numerical references.
3. Validate single-node EP8 before cross-node EP16; record actual NIC/device
   mappings and dispatch/combine mode.
4. Profile send and receive phases separately. Communication kernels include
   peer waiting; do not interpret their full duration as pure network transfer.
5. Confirm improvements using repeated timing without profiler, then compare
   full-service TPOT/throughput and actual per-rank batch distributions.
6. Run model-level accuracy before promoting the Humming path.

For further optimization, prioritize measured dispatch overhead and notification
frequency, then capacity-driven packing/copying. Preserve existing backends and
validate each change separately. No performance numbers from external
experiments are asserted for this committed snapshot.
