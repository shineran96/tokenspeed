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

"""Run on Hopper with torchrun --standalone --nproc-per-node=2 -m pytest."""

import math
import os
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.moe import moe_apply, moe_plan, moe_process_weights
from tokenspeed_kernel.platform import current_platform


@dataclass
class _Clamp:
    limit: float
    alpha: float


class _Experts(torch.nn.Module):
    def __init__(self, group):
        super().__init__()
        self.tp_size = 1
        self.ep_rank = group.rank()
        self.ep_size = group.size()
        self.num_local_experts = 4
        self.num_experts = self.num_local_experts * self.ep_size
        self.top_k = 1
        self.hidden_size = self.intermediate_size = 512
        self.swiglu_arg = _Clamp(limit=10.0, alpha=1.0)
        self._moe_backend_state = None
        # E2M1 code 2 is one. Make the down projection rank-dependent so
        # that a local-only computation cannot satisfy the reference.
        for name, rows, cols, value in (
            ("w13_weight", 1024, 256, 0x22),
            ("w2_weight", 512, 256, 0x22),
            ("w13_weight_scale", 1024, 16, 118),
            ("w2_weight_scale", 512, 16, 118 + self.ep_rank),
        ):
            self.register_buffer(
                name,
                torch.full(
                    (self.num_local_experts, rows, cols),
                    value,
                    device="cuda",
                    dtype=torch.uint8,
                ),
            )


@pytest.mark.skipif(
    not current_platform().is_hopper or int(os.environ.get("WORLD_SIZE", 1)) != 2,
    reason="requires two Hopper ranks under torchrun",
)
def test_hopper_mega_moe_remote_expert_graph():
    pytest.importorskip("nvshmem.core")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        group = dist.group.WORLD
        experts = _Experts(group)
        plan = moe_plan(
            "mxfp4",
            activation="swiglu",
            routing_mode="precomputed_topk",
            a2a_backend="none",
            ep_size=2,
            ispp=512,
            hidden=512,
            swiglu_form="standard",
            activation_clamped=True,
            expert_id_repeats=False,
            num_experts=8,
            top_k=1,
            process_group=group,
            persistent_max_num_tokens_per_gpu=8,
            fast_math=True,
            combine_order="rank",
            features=frozenset(),
            solution="mega_moe",
        )
        with torch.inference_mode():
            moe_process_weights(plan, experts)
        remote = 1 - group.rank()
        for rows in (2 + group.rank(), group.rank()):
            x = torch.ones(rows, 512, dtype=torch.bfloat16, device="cuda")
            ids = torch.full((rows, 1), 4 * remote, dtype=torch.int32, device="cuda")
            scores = torch.full((rows, 1), 0.5, device="cuda")

            def forward():
                return moe_apply(
                    plan,
                    x,
                    experts,
                    None,
                    scores,
                    ids,
                    max_num_tokens_per_gpu=3,
                    do_finalize=True,
                )

            with torch.inference_mode():
                forward()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                y = forward()
            scores.fill_(0.25)
            graph.replay()
            # FC1 produces gate=up=1; FC2 sums 512 copies of SiLU(1).
            expected = torch.full_like(x, 0.25 * 2**remote / (1 + math.exp(-1)))
            torch.testing.assert_close(y, expected, atol=0.025, rtol=0)
        experts._moe_backend_state.layer.destroy()
    finally:
        dist.destroy_process_group()
