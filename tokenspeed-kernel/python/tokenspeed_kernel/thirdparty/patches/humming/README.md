# Humming grouped-mask active-row scheduling

`active_rows.patch` carries Humming commit
`219fc5061738dd5a0f7bfc92d3d3c9d36cd773ae` on upstream base
`d0c7af98d7f556132f86e671de7f397991e2df2b` from
<https://github.com/vllm-project/humming>.
The patch includes the implementation, its layout regression tests, and the
upstream configuration documentation. It does not change TokenSpeed's MoE API.

With `grouped_mask` input and `zero_invalid=False`, fixed workers per expert
read valid row counts from the GPU and process only active rows. Padded storage
and CUDA Graph addresses remain fixed. Counts are read after the PDL dependency
wait; empty experts still complete dependencies. Input capacity must continue
to cover the largest batch and routing skew. This adapts to MoE row counts,
not attention context length. Zero-invalid processing and scale finalizers
retain their existing launch layouts.

## Install in the serving virtual environment

Activate the same Python virtual environment used to launch TokenSpeed. From
the TokenSpeed repository root, use a fresh destination for the Humming clone:

```bash
export TS_HUMMING_PATCH="$PWD/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/patches/humming/active_rows.patch"
git clone https://github.com/vllm-project/humming.git /path/to/humming-active-rows
git -C /path/to/humming-active-rows checkout --detach d0c7af98d7f556132f86e671de7f397991e2df2b
git -C /path/to/humming-active-rows apply --check --unidiff-zero "$TS_HUMMING_PATCH"
git -C /path/to/humming-active-rows apply --unidiff-zero "$TS_HUMMING_PATCH"
python -m pip install --no-deps --no-build-isolation -e /path/to/humming-active-rows
```

The patch has zero context lines so repository whitespace hooks do not rewrite
diff context. Apply it only to the exact base above, not an arbitrary checkout.
An installation already at commit `219fc50` must not apply it again. Restart
the serving process after installation so it imports the updated dependency.
Cloning TokenSpeed alone does not upgrade an already installed Humming package.

## Validation

Run on SM90 with the serving virtual environment active:

```bash
cd /path/to/humming-active-rows
python -m pytest -q tests/kernels/process_input/test_layout.py
```

The included changing-count tests cover empty and skewed experts, recovery,
int32/int64 counts, PDL on/off, untouched padding, and eager/CUDA Graph agreement.
Compare end-to-end performance using identical acceptance, token lengths,
concurrency, and client TPOT definitions. Simulated acceptance is a performance
test and does not establish model accuracy.
