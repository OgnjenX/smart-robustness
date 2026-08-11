from __future__ import annotations

import numpy as np
import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.classic_sector import (
    build_first_order_chemical_sector,
    build_first_order_connected_sector,
)
from smart_robustness.projections import SUPPLEMENTARY_TABLE3
from smart_robustness.synapses import topology_pairs


def test_topology_pairs_cover_one_to_one_all_to_one_and_gaussian() -> None:
    one = SUPPLEMENTARY_TABLE3.by_id("l4_e.proximal_dendrite.from_l4_e.ampa")
    pre, post, factor = topology_pairs(one, source_shape=(9, 9), target_shape=(9, 9))
    assert np.array_equal(pre, post)
    assert np.all(factor == 1)

    all_to_one = SUPPLEMENTARY_TABLE3.by_id("l5_e.distal_dendrite.from_matrix.ampa")
    pre, post, factor = topology_pairs(all_to_one, source_shape=(1, 1), target_shape=(9, 9))
    assert len(pre) == 81
    assert set(post) == set(range(81))
    assert np.all(factor == 1)

    gaussian = SUPPLEMENTARY_TABLE3.by_id("l4_i.proximal_dendrite.from_l4_e.ampa")
    pre, post, factor = topology_pairs(gaussian, source_shape=(9, 9), target_shape=(9, 9))
    center = (pre == 40) & (post == 40)
    corner = (pre == 40) & (post == 0)
    assert factor[center][0] > factor[corner][0]


def test_first_order_chemical_sector_builds_all_in_scope_records() -> None:
    brian.start_scope()
    sector = build_first_order_chemical_sector(brian=brian)
    assert len(sector.projections) == 48
    assert all(len(projection) > 0 for projection in sector.projections.values())
    sector.network.run(0 * brian.ms)


def test_first_order_connected_sector_adds_all_gap_junction_records() -> None:
    brian.start_scope()
    sector = build_first_order_connected_sector(brian=brian)
    assert len(sector.projections) == 52
    assert (
        sum(
            len(population.compiled.gap_junction_ports)
            for population in sector.populations.values()
        )
        == 4
    )
    sector.network.run(0 * brian.ms)
