from tokenspeed.runtime.distributed.mapping import Mapping

import pytest

from tokenspeed.runtime.pd.topology import PDParallelTopology


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"tp_size": 0}, "tp_size must be greater than 0"),
        ({"dp_size": 0}, "dp_size must be greater than 0"),
        ({"tp_rank": -1}, "tp_rank must be in \\[0, 2\\)"),
        ({"tp_rank": 2}, "tp_rank must be in \\[0, 2\\)"),
        ({"dp_rank": -1}, "dp_rank must be in \\[0, 4\\)"),
        ({"dp_rank": 4}, "dp_rank must be in \\[0, 4\\)"),
        ({"world_size": 7}, "world_size must equal tp_size \\* dp_size"),
        ({"global_rank": -1}, "global_rank must be in \\[0, 8\\)"),
        ({"global_rank": 8}, "global_rank must be in \\[0, 8\\)"),
    ],
)
def test_direct_constructor_rejects_invalid_topology(
    override: dict[str, int], match: str
) -> None:
    values = {
        "tp_size": 2,
        "tp_rank": 1,
        "dp_size": 4,
        "dp_rank": 1,
        "world_size": 8,
        "global_rank": 7,
    }
    values.update(override)

    with pytest.raises(ValueError, match=match):
        PDParallelTopology(**values)


def _mapping(
    *,
    tp_size: int,
    dp_size: int,
    world_size: int,
    global_rank: int,
) -> Mapping:
    return Mapping(rank=global_rank, world_size=world_size,
                   attn_tp_size=tp_size, attn_dp_size=dp_size)


def test_topology_preserves_typed_attention_coordinates() -> None:
    mapping = _mapping(
        tp_size=2,
        dp_size=4,
        world_size=8,
        global_rank=3,
    )

    topology = PDParallelTopology.from_mapping(mapping)

    assert topology == PDParallelTopology(
        tp_size=2,
        tp_rank=1,
        dp_size=4,
        dp_rank=1,
        world_size=8,
        global_rank=3,
    )


def test_cache_pd_accepts_heterogeneous_tp() -> None:
    topology = PDParallelTopology.from_mapping(
        _mapping(
            tp_size=4,
            dp_size=2,
            world_size=8,
            global_rank=7,
        )
    )

    assert (topology.tp_size, topology.tp_rank) == (4, 3)
