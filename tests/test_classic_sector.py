from __future__ import annotations

import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.classic_sector import build_first_order_intrinsic_sector


def test_first_order_intrinsic_sector_builds_all_source_cells() -> None:
    brian.start_scope()
    brian.defaultclock.dt = 0.01 * brian.ms
    sector = build_first_order_intrinsic_sector(brian=brian)
    assert len(sector.populations) == 12
    assert sector.cell_count == 812
    assert sector.compartment_count == 1950
    assert int(sector.populations["thalamic_relay"].group.N) == 81
    assert int(sector.populations["thalamic_nonspecific"].group.N) == 1
    sector.network.run(0 * brian.ms)


def test_network_layer5_and_layer6ii_use_serialized_fast_ahp() -> None:
    brian.start_scope()
    sector = build_first_order_intrinsic_sector(brian=brian)
    for name in ("layer5_excitatory_v1", "layer6ii_excitatory_v1"):
        population = sector.populations[name]
        assert population.compiled.ahp_ach_enabled
        assert population.compiled.ahp_convention.value == "smart_network_112923"
        assert "dahp_fall/dt=-ahp_fall/(20.0*ms)" in population.compiled.equations
