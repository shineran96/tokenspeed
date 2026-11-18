# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Hopper MXFP4/FP8 experts over Legacy DeepEP low-latency dispatch."""
from __future__ import annotations
import torch
from tokenspeed_kernel._optional import has_optional_package
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_nvidia and has_optional_package("humming"):
    from tokenspeed_kernel.thirdparty.humming import HummingMaskedExperts
    from tokenspeed_kernel.ops.moe._deep_ep.humming_scale import copy_scales
    from tokenspeed_kernel.ops.moe.deep_ep import _mxfp4_get_dispatcher
    from tokenspeed_kernel.ops.communication.deep_ep import DeepEPMode

    def humming_mxfp4_moe_weights(plan: dict, w: torch.nn.Module):
        """Transform concatenated gate/up MXFP4 weights once and release raw storage."""
        if "humming_experts" in w._modules:
            return
        if w.w13_input_layout != "concatenated":
            raise ValueError("Humming requires concatenated gate/up rows")
        arg = w.swiglu_arg
        if arg is None or arg.limit is None or arg.alpha not in (None, 1.0) or w.swiglu_beta not in (None, 0.0):
            raise ValueError("Humming currently requires standard clamped SwiGLU")
        tensors=(w.w13_weight,w.w13_weight_scale,w.w2_weight,w.w2_weight_scale)
        if any(t.dtype != torch.uint8 or t.ndim != 3 for t in tensors):
            raise ValueError("Humming expects rank-3 packed MXFP4 weights and E8M0 scales")
        experts=HummingMaskedExperts(*tensors,limit=float(arg.limit),batch_invariant=False)
        w.add_module("humming_experts",experts)
        # Alias the transformed parameters instead of retaining raw checkpoint copies.
        w.w13_weight=experts.layers[0].weight
        w.w13_weight_scale=experts.layers[0].weight_scale
        w.w2_weight=experts.layers[1].weight
        w.w2_weight_scale=experts.layers[1].weight_scale

    @register_kernel(
        "moe","apply",name="humming_mxfp4_deepep_moe_apply",solution="humming",
        weight_preprocessor=humming_mxfp4_moe_weights,
        capability=CapabilityRequirement(vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(9,0),max_arch_version=ArchVersion(9,0)),
        signatures=format_signatures("x","dense",{torch.bfloat16}),
        traits={
            "weight_dtype":frozenset({"mxfp4"}),
            "activation":frozenset({"silu","swiglu"}),
            "swiglu_form":frozenset({"standard"}),
            "activation_clamped":frozenset({True}),
            "expert_id_repeats":frozenset({False}),
            "routing_mode":frozenset({"precomputed_topk"}),
            "supports_deferred_finalize":frozenset({False}),
            "supports_ep":frozenset({True}),
            "supports_all_to_all_ep":frozenset({True}),
            "deepep_modes":frozenset({"low_latency"}),
            "ispp_alignment":frozenset({128}),
            "hidden_alignment":frozenset({128}),
            "internal_activation_dtype":frozenset({"fp8"}),
            "supports_bias":frozenset({False}),
        },priority=Priority.PORTABLE,
    )
    def humming_mxfp4_deepep_moe_apply(
        plan: dict, x: torch.Tensor, w: torch.nn.Module, router_logits: torch.Tensor,
        topk_weights: torch.Tensor | None=None, topk_ids: torch.Tensor | None=None,
        num_tokens_global: int | None=None, max_num_tokens_per_gpu: int | None=None,
        do_finalize: bool=True, enable_pdl: bool=False,
        low_latency: bool | None=None, overlap_fn=None,
    ) -> torch.Tensor:
        """Dispatch FP8 rows, compute masked W4A8 experts, and combine BF16 once."""
        if not do_finalize or topk_ids is None or topk_weights is None:
            raise ValueError("Humming DeepEP requires precomputed routes and finalization")
        dispatcher=_mxfp4_get_dispatcher(plan,w,x,use_fp8=True)
        if dispatcher.deepep_mode.resolve(low_latency) != DeepEPMode.low_latency:
            raise ValueError("Humming currently supports DeepEP low_latency only")
        ids=topk_ids.to(torch.int64)
        weights=topk_weights.to(torch.float32)
        bound=num_tokens_global if num_tokens_global is not None else plan["deepep_low_latency_max_num_tokens_per_gpu"]*w.ep_size
        dispatcher.dispatch_a(x,ids,weights,low_latency=True)
        if overlap_fn is not None:
            overlap_fn()
        recv,_,_,_,_,_,counts=dispatcher.dispatch_b()
        values,scales=recv
        # Humming uses this only to select a tuning configuration. Device counts
        # still control all expert work, including skewed or empty routing.
        expected_local_routes=max(1,(bound*ids.shape[1]+w.ep_size-1)//w.ep_size)
        output=w.humming_experts(values,counts,input_scale=copy_scales(scales,counts),
            valid_rows=expected_local_routes,enable_pdl=enable_pdl)
        dispatcher.combine_a(output,ids,weights,low_latency=True)
        return dispatcher.combine_b()
