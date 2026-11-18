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

import torch
from tokenspeed_kernel._triton import triton, tl

@triton.jit
def _copy_scale(src, dst, counts, capacity, groups, se, sr, sg,
                BLOCK: tl.constexpr, WORKERS: tl.constexpr):
    e=tl.program_id(0).to(tl.int64)
    start=tl.program_id(1)*BLOCK
    count=tl.load(counts+e)
    for offset in range(start,count*groups,WORKERS*BLOCK):
        idx=offset+tl.arange(0,BLOCK)
        row=(idx//groups).to(tl.int64)
        col=idx%groups
        val=tl.load(src+e*se+row*sr+col*sg,idx<count*groups,other=0)
        tl.store(dst+(e*capacity+row)*groups+col,val,idx<count*groups)

def copy_scales(scales: torch.Tensor, counts: torch.Tensor, out: torch.Tensor | None = None):
    """Copy only GPU-counted rows from DeepEP scales to Humming [E*C,G]."""
    experts,capacity,groups=scales.shape
    if scales.dtype != torch.float32 or counts.shape != (experts,):
        raise ValueError("expected FP32 expert scales and one count per expert")
    if out is None:
        out=torch.empty((experts*capacity,groups),device=scales.device,dtype=scales.dtype)
    _copy_scale[(experts,8)](scales,out,counts,capacity,groups,*scales.stride(),BLOCK=256,WORKERS=8)
    return out
