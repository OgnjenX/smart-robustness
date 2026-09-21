from __future__ import annotations

from dataclasses import replace

import pytest

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.classic_sector import first_order_population_parameters
from smart_robustness.models.inhibitory_routing import (
    PROJECTION036_ID,
    REGISTERED_INHIBITORY_PROJECTION_IDS,
    InhibitoryRoutingMode,
    cortical_inhibitory_projection_ids,
    make_inhibitory_routing_sector_builder,
    make_layer4_projection036_somatic_population_factory,
    retarget_projection036_to_soma,
)
from smart_robustness.models.modeldb112923 import first_order_population_facts


def test_source_inventory_is_exact() -> None:
    assert cortical_inhibitory_projection_ids() == REGISTERED_INHIBITORY_PROJECTION_IDS


def test_legacy_wrapper_returns_sector_without_mutation() -> None:
    marker = object()
    sector = type(
        "Sector",
        (),
        {"projections": {projection_id: marker for projection_id in REGISTERED_INHIBITORY_PROJECTION_IDS}},
    )()
    calls = []

    def base_builder(**kwargs):
        calls.append(kwargs)
        return sector

    builder = make_inhibitory_routing_sector_builder(
        mode=InhibitoryRoutingMode.LEGACY_AGGREGATE,
        base_builder=base_builder,
    )
    assert builder(example=3) is sector
    assert calls == [{"example": 3}]
    assert all(value is marker for value in sector.projections.values())


def test_legacy_wrapper_rejects_positional_arguments() -> None:
    builder = make_inhibitory_routing_sector_builder(
        mode="legacy_aggregate",
        base_builder=lambda **_: None,
    )
    with pytest.raises(TypeError, match="keyword arguments"):
        builder(1)


def test_legacy_wrapper_fails_closed_on_missing_projection() -> None:
    sector = type("Sector", (), {"projections": {}})()
    builder = make_inhibitory_routing_sector_builder(
        mode="legacy_aggregate",
        base_builder=lambda **_: sector,
    )
    with pytest.raises(RuntimeError, match="lacks registered inhibitory projections"):
        builder()


def test_unregistered_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        make_inhibitory_routing_sector_builder(
            mode="pv_like",
            base_builder=lambda **_: None,
        )


def layer4_excitatory_parameters():
    baseline = load_frozen_classic_baseline(
        "configs/baselines/classic_smart_calibrated_v1.yaml"
    )
    facts = next(
        item
        for item in first_order_population_facts()
        if item.canonical_name == "layer4_excitatory_v1"
    )
    return first_order_population_parameters(
        facts, conventions=baseline.runtime_conventions()
    )


def test_projection036_somatic_transform_changes_only_target_compartment() -> None:
    original = layer4_excitatory_parameters()
    transformed = retarget_projection036_to_soma(original)
    assert transformed is not original
    assert original["synaptic_ports"] is not transformed["synaptic_ports"]
    assert {
        key: value for key, value in transformed.items() if key != "synaptic_ports"
    } == {key: value for key, value in original.items() if key != "synaptic_ports"}
    original_ports = {port.record_id: port for port in original["synaptic_ports"]}
    transformed_ports = {port.record_id: port for port in transformed["synaptic_ports"]}
    assert original_ports.keys() == transformed_ports.keys()
    for record_id, original_port in original_ports.items():
        if record_id == PROJECTION036_ID:
            assert original_port.compartment == "proximal_dendrite"
            assert transformed_ports[record_id].compartment == "soma"
            assert transformed_ports[record_id] == replace(
                original_port, compartment="soma"
            )
        else:
            assert transformed_ports[record_id] == original_port
    assert original_ports[PROJECTION036_ID].compartment == "proximal_dendrite"


def test_projection036_somatic_transform_exactly_preserves_total_conductance() -> None:
    original = layer4_excitatory_parameters()
    transformed = retarget_projection036_to_soma(original)
    cell = original["cell_spec"]
    old = next(port for port in original["synaptic_ports"] if port.record_id == PROJECTION036_ID)
    new = next(port for port in transformed["synaptic_ports"] if port.record_id == PROJECTION036_ID)
    old_total = old.conductance_density_mS_cm2 * cell.compartment(old.compartment).lateral_area_cm2
    new_total = new.conductance_density_mS_cm2 * cell.compartment(new.compartment).lateral_area_cm2
    assert old_total == new_total


def test_pv_like_population_factory_passes_other_populations_through() -> None:
    calls = []

    def base_factory(**kwargs):
        calls.append(kwargs)
        return kwargs

    factory = make_layer4_projection036_somatic_population_factory(
        base_factory=base_factory
    )
    unrelated = {"synaptic_ports": ()}
    result = factory(name="smart_v1_trn", params=unrelated, size=1, brian=object())
    assert result["params"] is unrelated
    params = layer4_excitatory_parameters()
    result = factory(
        name="smart_v1_layer4_excitatory_v1",
        params=params,
        size=81,
        brian=object(),
    )
    changed = next(
        port for port in result["params"]["synaptic_ports"] if port.record_id == PROJECTION036_ID
    )
    assert changed.compartment == "soma"
    assert params["synaptic_ports"] is not result["params"]["synaptic_ports"]


def test_projection036_transform_fails_closed_if_port_is_missing() -> None:
    params = layer4_excitatory_parameters()
    params["synaptic_ports"] = tuple(
        port for port in params["synaptic_ports"] if port.record_id != PROJECTION036_ID
    )
    with pytest.raises(RuntimeError, match="exactly once"):
        retarget_projection036_to_soma(params)
