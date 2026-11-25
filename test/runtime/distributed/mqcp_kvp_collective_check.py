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

"""Two-node MQCP/KVP byte checks; torchrun with eight ranks per node."""

import json
import os

import torch
import torch.distributed as dist

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import process_group_manager as pg
from tokenspeed.runtime.distributed.comm_backend import set_global_backend
from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
from tokenspeed.runtime.execution.query_shard import QueryShardPlan
from tokenspeed.runtime.layers.paged_attention import QueryShardGather
from tokenspeed.runtime.layers.attention.deepseek_v41_kvp import gather_paged_history
from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement


def main():
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    mapping = Mapping(rank=rank, world_size=16, attn_tp_size=8, attn_dp_size=1,
                      attn_qcp_size=8, attn_kvp_size=16, mqcp_size=2,
                      dense_tp_size=8, moe_tp_size=1, moe_ep_size=8, nnodes=2)
    pg.init_distributed(mapping, timeout=180, device_id=device)
    pg.init_process_group(mapping.attn.query_group)
    set_global_backend(NcclBackend())
    checks = 0
    for lengths in [[1], [17, 3], [8193, 511], [32768]]:
        plan = QueryShardPlan.from_forward(total_tokens=sum(lengths), input_lengths=lengths,
                                         size=16, rank=rank, device=device)
        for dtype in [torch.bfloat16, torch.float32, torch.int64, torch.uint8]:
            full = (torch.arange(plan.total_rows * 32, device=device).reshape(-1, 32) % 113).to(dtype)
            gather = QueryShardGather(plan, mapping.attn.query_group,
                                      local_group=None, cross_node_group=None)
            actual = gather.gather(full[plan.local_row_ids].contiguous())
            torch.testing.assert_close(actual, full, rtol=0, atol=0)
            checks += 1
    for rows in [1, 65, 1057, 2111]:
        pages = (rows + 63) // 64
        table_cpu = (torch.arange(pages, dtype=torch.int32) * 37 % 64) + 1
        table = table_cpu.to(device)
        logical = torch.arange(pages * 64, device=device)
        virtual = table[logical // 64].long() * 64 + logical % 64
        expected = ((logical[:, None] * 13 + torch.arange(68, device=device)[None, :]) % 256).byte()
        mine = (virtual // 64 - 1) % 16 == rank
        local = ((virtual // 64 - 1) // 16 + 1) * 64 + virtual % 64
        storage = torch.zeros((5, 64, 84), dtype=torch.uint8, device=device)
        cache = storage[:, :, :68]
        cache[local[mine] // 64, local[mine] % 64] = expected[mine]
        placement = CachePlacement(64, 65, mapping.attn.kvp_group, mapping.attn.kvp_rank)
        actual = gather_paged_history(cache, table, table_cpu, rows=rows, placement=placement)
        assert not actual[0].any()
        torch.testing.assert_close(actual[1:].reshape(-1, 68), expected, rtol=0, atol=0)
        checks += 1
    dist.barrier()
    print(json.dumps(dict(rank=rank, status="passed", byte_exact_checks=checks)), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
