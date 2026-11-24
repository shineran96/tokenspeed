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

"""Optional Humming MXFP4 grouped-masked expert computation."""

from __future__ import annotations

import torch
from humming.forward import may_process_input
from humming.layer import HummingLayer


class HummingMaskedExperts(torch.nn.Module):
    """Own transformed weights; consume padded expert rows and GPU valid counts."""

    def __init__(self, w13, s13, w2, s2, *, limit: float, batch_invariant: bool):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.compute_config = {
            "gemm_type": "grouped_masked",
            "use_batch_invariant": batch_invariant,
        }
        self.activation = (
            "[](float gate, float up) { "
            f"gate = fminf(gate, {float(limit)}f); "
            f"up = fminf(fmaxf(up, -{float(limit)}f), {float(limit)}f); "
            "return (gate / (1.0f + expf(-gate))) * up; }(a, b)"
        )
        for weight, scale in ((w13, s13), (w2, s2)):
            layer = HummingLayer(
                shape_n=weight.shape[-2],
                shape_k=weight.shape[-1] * 2,
                num_experts=weight.shape[0],
                weight_config={"quant_method": "mxfp4"},
                input_config={"dtype": "float8_e4m3", "group_size": 128},
                torch_dtype=torch.bfloat16,
            ).to(weight.device)
            layer.load_from_tensors({"weight": weight, "weight_scale": scale})
            layer.transform()
            self.layers.append(layer)

    def forward(
        self,
        recv: torch.Tensor,
        counts: torch.Tensor,
        *,
        input_scale: torch.Tensor | None,
        valid_rows: int,
        enable_pdl: bool,
    ) -> torch.Tensor:
        """Apply unweighted experts; input_scale is flattened [E*capacity,H/128]."""
        a, b = self.layers
        x = recv.reshape(-1, recv.shape[-1])
        if input_scale is None:
            x, gs, ts = may_process_input(
                a.humming_config,
                x,
                layout="grouped_mask",
                expert_tokens=counts,
                use_pdl=enable_pdl,
            )
            input_scale = gs if gs is not None else ts
        up = a(
            x,
            input_scale=input_scale,
            expert_layout=counts,
            valid_shape_m=valid_rows,
            compute_config=self.compute_config,
            use_pdl=enable_pdl,
        )
        q, gs, ts = may_process_input(
            b.humming_config,
            up,
            layout="grouped_mask",
            expert_tokens=counts,
            activation_type="binary_split",
            activation_impl=self.activation,
            use_pdl=enable_pdl,
        )
        out = b(
            q,
            input_scale=gs if gs is not None else ts,
            expert_layout=counts,
            valid_shape_m=valid_rows,
            compute_config=self.compute_config,
            use_pdl=enable_pdl,
        )
        return out.reshape(*recv.shape[:-1], out.shape[-1])
