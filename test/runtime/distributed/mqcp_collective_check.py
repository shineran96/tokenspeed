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

"""Run with torchrun on two nodes, eight CUDA ranks each."""

import json
import os
import torch
import torch.distributed as dist

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import process_group_manager as pg
from tokenspeed.runtime.distributed.comm_backend import set_global_backend
from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.query_shard import QueryShardPlan
from tokenspeed.runtime.layers.paged_attention import QueryShardGather
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.models.deepseek_v41_engram import RowShardedEngramEmbedding


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    m = Mapping(rank=rank, world_size=16, attn_tp_size=8, attn_dp_size=1,
                attn_qcp_size=8, mqcp_size=2, dense_tp_size=8,
                moe_tp_size=1, moe_ep_size=8, nnodes=2)
    pg.init_distributed(m, timeout=180, device_id=device)
    for group in [m.world_group, m.attn.tp_group, m.attn.mqcp_group]:
        pg.init_process_group(group)
    set_global_backend(NcclBackend())
    with torch.device(device):
        embedding = VocabParallelEmbedding(512, 32, params_dtype=torch.bfloat16,
                    tp_rank=rank % 8, tp_size=8, tp_group=m.attn.tp_group)
    weights = (torch.arange(512*32, device=device).reshape(512,32) % 13).to(torch.bfloat16)
    embedding.weight_loader(embedding.weight, weights)
    engram = RowShardedEngramEmbedding(128,32,m,device,False,"gpu",1)
    codes = (torch.arange(128*32,device=device).reshape(128,32)%7).to(torch.float8_e4m3fn)
    scales = torch.full((128,1),127,device=device,dtype=torch.uint8)
    engram.weight_loader(engram.weight,codes[engram.row_start:engram.row_end])
    engram.weight_loader(engram.scale,scales[engram.row_start:engram.row_end])
    checks = 0
    for lengths in [[1], [17, 3], [8193, 511], [16384], [32768]]:
        plan = QueryShardPlan.from_forward(total_tokens=sum(lengths), input_lengths=lengths,
                        size=16, rank=rank, device=device)
        for narrow in [False, True]:
            current = plan
            if narrow:
                tail = torch.arange(max(0, sum(lengths)-4), sum(lengths))
                current = plan.select_rows(tail, input_lengths=[len(tail)],
                            sampled_rows=torch.tensor([len(tail)-1]))
            for dtype in [torch.bfloat16, torch.float32, torch.int64, torch.uint8]:
                full = (torch.arange(current.total_rows*32, device=device).reshape(-1,32) % 113).to(dtype)
                mine = full[current.local_row_ids].contiguous()
                gather = QueryShardGather(current, m.attn.query_group,
                          local_group=m.attn.qcp_group, cross_node_group=m.attn.mqcp_group)
                actual = gather.gather(mine)
                torch.testing.assert_close(actual, full, rtol=0, atol=0)
                checks += 1
            ctx = ForwardContext(attn_backend=None, token_to_kv_pool=None,
                bs=1, num_extends=1, input_num_tokens=current.total_rows,
                forward_mode=ForwardMode.EXTEND, output_layout=None,
                gather_ids=torch.tensor([current.total_rows-1]), query_shard=current)
            manager = CommManager(mapping=m, layer_id=1, is_moe=True, prev_is_moe=False,
                                  query_sharded=True, dense_batch_invariant=False)
            first = rank//8*8
            assert manager.scattered_num_tokens(ctx) == list(current.row_counts[first:first+8])
            assert manager.moe_tp_ep_group_scattered_num_tokens(ctx) == list(current.row_counts[first:first+8])
            ids = current.local_row_ids % 512
            actual = embedding(ids, query_shard=current)
            torch.testing.assert_close(actual, weights[ids], rtol=0, atol=0)
            hashes = (current.local_row_ids[:,None] + torch.arange(3,device=device)[None,:]) % 128
            actual = engram(hashes, query_shard=current)
            torch.testing.assert_close(actual, codes.to(torch.bfloat16)[hashes], rtol=0, atol=0)
            mine = torch.ones((current.local_rows,32),device=device,dtype=torch.bfloat16)
            gathered = manager.pre_moe_comm(mine,ctx)
            actual,_ = manager.post_moe_comm(gathered / 8,mine,ctx)
            torch.testing.assert_close(actual,mine,rtol=0,atol=0)
    dist.barrier()
    print(json.dumps({"rank":rank,"byte_exact_gathers":checks,"status":"passed"}),flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
