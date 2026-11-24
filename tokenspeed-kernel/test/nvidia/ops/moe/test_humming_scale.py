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

import pytest
import torch
from tokenspeed_kernel.ops.moe._deep_ep.humming_scale import copy_scales

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_humming_scale_dynamic_graph():
    experts, capacity, groups = 8, 256, 40
    scales = torch.randn(experts, groups, capacity, device="cuda").transpose(1, 2)
    counts = torch.full((experts,), 24, device="cuda", dtype=torch.int32)
    out = torch.full((experts * capacity, groups), -123.0, device="cuda")
    copy_scales(scales, counts, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        copy_scales(scales, counts, out)
    for count in (0, 1, 24, 137, 256):
        counts.fill_(count)
        counts[::3] = 0
        out.fill_(-123)
        graph.replay()
        valid = torch.arange(capacity, device="cuda")[None, :] < counts[:, None]
        torch.testing.assert_close(
            out.view(experts, capacity, groups)[valid], scales[valid], atol=0, rtol=0
        )
        assert (out.view(experts, capacity, groups)[~valid] == -123).all()


def test_humming_scale_invalid_dtype():
    scales = torch.empty(2, 8, 4, device="cuda", dtype=torch.bfloat16)
    counts = torch.zeros(2, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="FP32"):
        copy_scales(scales, counts)
