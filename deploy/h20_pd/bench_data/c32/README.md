# C32 archived benchmark data

inputs.json.gz contains the exact original 64 token-ID inputs; C32 selects the first 32.
Decompress with gzip -dk inputs.json.gz and verify the uncompressed SHA256 against manifest.json.

Each backend tar.gz includes prefix prime, C32 warmup, all three formal rounds,
per-request streaming timestamps/results, worker-load observations, summary files,
and the original client/controller/provenance. Extract with tar -xzf into separate
directories. The original controllers also contain C64 and environment-specific
paths; they are archived evidence, not portable one-command runners.

Formal statistics exclude prime and warmup. Fixed accepted length 4 is performance-only.
The two archives do not include model weights, native dependencies or full GPU traces.
See ../../C32_BENCHMARK.md for protocol and interpretation.
