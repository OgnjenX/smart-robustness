from __future__ import annotations

import numpy as np
import pytest

from smart_robustness.modeldb_projections import MODELDB_FIRST_ORDER
from smart_robustness.models.modeldb112923 import first_order_population_facts
from smart_robustness.partition import (
    PartitionedProjection,
    complementary_partitions,
    partition_edges,
)
from smart_robustness.synapses import modeldb_topology_pairs


def test_complementary_relay_partitions_preserve_global_order() -> None:
    clamped, free = complementary_partitions(
        population_size=81,
        selected_indices=(38, 39, 40, 41, 42),
        selected_name="clamped",
        remainder_name="free",
    )
    assert clamped.global_indices == (38, 39, 40, 41, 42)
    assert len(free.global_indices) == 76
    assert set(clamped.global_indices) | set(free.global_indices) == set(range(81))
    assert clamped.global_to_local[40] == 2
    assert free.global_to_local[40] == -1


def test_partitioned_edges_reconstruct_original_edge_list_losslessly() -> None:
    clamped, free = complementary_partitions(
        population_size=5, selected_indices=(1, 3), selected_name="clamped", remainder_name="free"
    )
    source = np.asarray((0, 1, 2, 3, 4, 1, 4))
    target = np.asarray((1, 2, 3, 4, 0, 3, 1))
    factor = np.linspace(0.1, 0.7, len(source))
    blocks = partition_edges(
        source,
        target,
        factor,
        source_partitions=(clamped, free),
        target_partitions=(clamped, free),
    )
    reconstructed = sorted(
        (int(i), int(j), float(w))
        for block in blocks
        for i, j, w in zip(
            block.source_global,
            block.target_global,
            block.spatial_factor,
            strict=True,
        )
    )
    expected = sorted(
        (int(i), int(j), float(w))
        for i, j, w in zip(source, target, factor, strict=True)
    )
    assert reconstructed == expected


def test_partition_edges_rejects_incomplete_cover() -> None:
    selected, _ = complementary_partitions(population_size=3, selected_indices=(0,))
    with pytest.raises(ValueError, match="missing=1"):
        partition_edges(
            np.asarray((0, 1)),
            np.asarray((0, 1)),
            np.ones(2),
            source_partitions=(selected,),
            target_partitions=(selected,),
        )


def test_relay_split_losslessly_partitions_every_first_order_modeldb_topology() -> None:
    facts = {fact.canonical_name: fact for fact in first_order_population_facts()}
    relay_parts = complementary_partitions(
        population_size=81,
        selected_indices=(38, 39, 40, 41, 42),
        selected_name="clamped",
        remainder_name="free",
    )
    for record in MODELDB_FIRST_ORDER.projections:
        if record.source_population not in facts or record.target_population not in facts:
            continue
        source_shape = facts[record.source_population].shape
        target_shape = facts[record.target_population].shape
        source, target, factor = modeldb_topology_pairs(
            record,
            source_shape=source_shape,
            target_shape=target_shape,
        )
        source_parts = (
            relay_parts
            if record.source_population == "thalamic_relay"
            else complementary_partitions(
                population_size=source_shape[0] * source_shape[1],
                selected_indices=tuple(range(source_shape[0] * source_shape[1])),
                selected_name="all",
                remainder_name="empty",
            )[:1]
        )
        target_parts = (
            relay_parts
            if record.target_population == "thalamic_relay"
            else complementary_partitions(
                population_size=target_shape[0] * target_shape[1],
                selected_indices=tuple(range(target_shape[0] * target_shape[1])),
                selected_name="all",
                remainder_name="empty",
            )[:1]
        )
        blocks = partition_edges(
            source,
            target,
            factor,
            source_partitions=source_parts,
            target_partitions=target_parts,
        )
        assert sum(len(block.source_local) for block in blocks) == len(source), record.id


class _Block:
    def __init__(self, weights: tuple[float, ...]) -> None:
        self.w = np.asarray(weights, dtype=float)
        self.modifiable = np.ones(len(weights))

    def __len__(self) -> int:
        return len(self.w)


def test_partitioned_projection_reconstructs_and_updates_global_weight_view() -> None:
    first = _Block((0.1, 0.2))
    second = _Block((0.3,))
    projection = PartitionedProjection(
        blocks=(first, second),
        source_global=(np.asarray((38, 39)), np.asarray((40,))),
        target_global=(np.asarray((1, 2)), np.asarray((3,))),
    )
    assert projection.i.tolist() == [38, 39, 40]
    assert projection.j.tolist() == [1, 2, 3]
    assert projection.read("w") == pytest.approx((0.1, 0.2, 0.3))
    projection.write("w", np.asarray((0.4, 0.5, 0.6)))
    assert first.w == pytest.approx((0.4, 0.5))
    assert second.w == pytest.approx((0.6,))
    projection.write("modifiable", 0.0)
    assert first.modifiable == pytest.approx(0)
    assert second.modifiable == pytest.approx(0)
