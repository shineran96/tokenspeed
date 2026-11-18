# DeepSeek V4.1 DeepEP integration

DeepEP owns expert dispatch and combine. The model passes local token rows to
the expert layer and must not apply the ordinary EP gather or output reduction.
Shared experts use the dense TP group, so attention DP with dense TP1 keeps a
complete shared expert on each rank and computes only that rank's rows.

Select the existing all-to-all and MoE flags explicitly. For Hopper MXFP4
experts the candidate is `--all2all-backend deepep --moe-backend marlin
--deepep-mode auto --low-latency-max-num-tokens-per-gpu 256`.
The capacity counts token rows, including speculative verification and graph
padding, rather than requests. Every EP rank, including idle ranks, participates.
Auto mode uses the common decode-or-idle decision; mixed/extend steps use normal
dispatch. DeepEP cannot be stacked with another backend that owns EP communication.

Validation status: model integration is under test. Package availability and
successful model construction do not establish distributed or graph correctness.
