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
    assert (
        sum(len(population.compiled.synaptic_ports) for population in sector.populations.values())
        == 49
    )
    assert (
        sum(
            len(population.compiled.external_input_ports)
            for population in sector.populations.values()
        )
        == 2
    )
    assert (
        sum(
            len(population.compiled.gap_junction_ports)
            for population in sector.populations.values()
        )
        == 4
    )
    sector.network.run(0 * brian.ms)


def test_network_layer5_and_layer6ii_use_serialized_fast_ahp() -> None:
    brian.start_scope()
    sector = build_first_order_intrinsic_sector(brian=brian)
    for name in ("layer5_excitatory_v1", "layer6ii_excitatory_v1"):
        population = sector.populations[name]
        assert population.compiled.ahp_ach_enabled
        assert population.compiled.ahp_convention.value == "smart_network_112923"
        assert "dahp_fall/dt=-ahp_fall/(20.0*ms)" in population.compiled.equations


def test_bottom_up_gate_is_explicitly_addressable_by_source_record() -> None:
    brian.start_scope()
    sector = build_first_order_intrinsic_sector(brian=brian)
    relay = sector.populations["thalamic_relay"]
    record_id = "relay.proximal_dendrite.from_input.input"
    relay.set_external_input(record_id, 0.25, indices=[40])
    port = next(port for port in relay.compiled.external_input_ports if port.record_id == record_id)
    assert getattr(relay.group, port.name)[40] == pytest.approx(0.25)
    assert getattr(relay.group, port.name)[0] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="between zero and one"):
        relay.set_external_input(record_id, 1.1)
