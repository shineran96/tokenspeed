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

"""MQCP keeps weight groups local while one request spans all query ranks."""

import pytest
import torch

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.query_shard import QueryShardPlan
from tokenspeed.runtime.pd.topology import PDParallelTopology


def mqcp_mapping(rank):
    return Mapping(rank=rank, world_size=16, attn_tp_size=8, attn_dp_size=1,
                   attn_qcp_size=8, mqcp_size=2, dense_tp_size=8,
                   moe_tp_size=1, moe_ep_size=8, nnodes=2)


def test_mqcp_local_weights_and_global_request_ownership():
    for rank in range(16):
        m = mqcp_mapping(rank)
        local = tuple(range(rank // 8 * 8, rank // 8 * 8 + 8))
        assert m.attn.tp_group == m.dense.tp_group == m.moe.ep_group == local
        assert m.lm_head.tp_group == local
        assert m.attn.dp_size == m.moe.dp_size == 1
        assert m.attn.query_group == m.attn.replica_group == tuple(range(16))
        assert m.attn.mqcp_group == (rank % 8, rank % 8 + 8)
        assert m.attn.head_tp_size == 1
        pd = PDParallelTopology.from_mapping(m)
        assert (pd.tp_size, pd.tp_rank, pd.dp_size) == (16, rank, 1)


@pytest.mark.parametrize("lengths", [[1], [17, 3], [16384], [32768], [8193, 511]])
def test_global_query_ownership_and_ced_tail(lengths):
    plans = [QueryShardPlan.from_forward(total_tokens=sum(lengths), input_lengths=lengths,
              size=16, rank=rank, device="cpu") for rank in range(16)]
    owned = torch.cat([p.local_row_ids for p in plans]).sort().values
    assert torch.equal(owned, torch.arange(sum(lengths)))
    if lengths == [32768]:
        assert all(p.local_rows == 2048 for p in plans)
    tail = torch.arange(max(0, sum(lengths)-4), sum(lengths))
    narrowed = [p.select_rows(tail, input_lengths=[len(tail)],
                sampled_rows=torch.tensor([len(tail)-1])) for p in plans]
    data = torch.arange(len(tail)*3).reshape(-1, 3)
    gathered = torch.cat([data[p.local_row_ids] for p in narrowed])
    for p in narrowed:
        assert torch.equal(p.to_span_order(gathered), data)
    assert sum(p.local_sampled_rows for p in narrowed) == 1


def test_mqcp_rejects_cross_node_weight_group():
    with pytest.raises((ValueError, AssertionError)):
        Mapping(world_size=16, attn_tp_size=8, attn_qcp_size=8, mqcp_size=2,
                dense_tp_size=8, moe_tp_size=1, moe_ep_size=16, nnodes=2)


def test_mqcp_launch_resolves_one_replica_and_local_ep():
    from tokenspeed.runtime.utils.server_args import prepare_server_args
    args = prepare_server_args([
        '--model', 'x', '--tensor-parallel-size', '8', '--nnodes', '2',
        '--mqcp-size', '2', '--query-context-parallel-size', '8',
        '--enable-expert-parallel', '--disaggregation-mode', 'prefill',
        '--enforce-eager', '--no-engram-host-table',
    ])
    assert args.mapping.world_size == 16
    assert args.mapping.attn.dp_size == 1
    assert args.mapping.moe.ep_size == 8
    assert args.mapping.dense.tp_size == 8


@pytest.mark.parametrize('extra', [
    ['--disaggregation-mode', 'decode'],
    ['--engram-host-table'],
    ['--dp-sampling'],
    ['--disable-sampling-tp-sync'],
    ['--kv-parallel-size', '8'],
    ['--query-context-parallel-size', '1'],
])
def test_mqcp_launch_rejects_unsupported_layout(extra):
    from tokenspeed.runtime.utils.server_args import prepare_server_args
    with pytest.raises((ValueError, AssertionError)):
        prepare_server_args([
            '--model', 'x', '--tensor-parallel-size', '8', '--nnodes', '2',
            '--mqcp-size', '2', '--query-context-parallel-size', '8',
            '--enable-expert-parallel', '--disaggregation-mode', 'prefill',
            '--enforce-eager', '--no-engram-host-table', *extra,
        ])


@pytest.mark.parametrize("kvp_size", [1, 16])
def test_mqcp_cache_owners_span_request_replica(kvp_size):
    from tokenspeed.runtime.layers.attention.dcp.placement import (
        CachePlacement, cyclic_slot_owner,
    )
    slots = torch.arange(1, 33, dtype=torch.int64) * 64
    owners = []
    for rank in range(16):
        m = Mapping(rank=rank, world_size=16, attn_tp_size=8, attn_dp_size=1,
                    attn_qcp_size=8, attn_kvp_size=kvp_size, mqcp_size=2,
                    dense_tp_size=8, moe_tp_size=1, moe_ep_size=8, nnodes=2)
        assert m.attn.tp_group == tuple(range(rank // 8 * 8, rank // 8 * 8 + 8))
        assert m.moe.ep_group == m.attn.tp_group
        assert m.attn.kvp_rank == rank % kvp_size
        assert m.attn.kvp_replica_rank == rank // kvp_size
        placement = CachePlacement(64, 65, m.attn.kvp_group, m.attn.kvp_rank)
        owners.append(cyclic_slot_owner(slots, placement) == m.attn.kvp_rank)
        if kvp_size == 16:
            assert m.attn.kvp_group == m.attn.query_group == tuple(range(16))
    assert torch.equal(torch.stack(owners).sum(0), torch.full((32,), 16 // kvp_size))


def test_mqcp_kvp16_launch_preserves_local_weights():
    from tokenspeed.runtime.utils.server_args import prepare_server_args
    args = prepare_server_args([
        '--model', 'x', '--tensor-parallel-size', '8', '--nnodes', '2',
        '--mqcp-size', '2', '--query-context-parallel-size', '8',
        '--kv-parallel-size', '16', '--enable-expert-parallel',
        '--disaggregation-mode', 'prefill', '--disable-prefill-graph',
        '--enforce-eager', '--no-engram-host-table',
    ])
    assert args.mapping.attn.kvp_size == args.mapping.attn.query_size == 16
    assert args.mapping.attn.tp_size == args.mapping.moe.ep_size == 8
    assert args.mapping.attn.dp_size == 1
