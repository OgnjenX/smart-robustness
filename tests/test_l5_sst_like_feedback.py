from __future__ import annotations

from types import SimpleNamespace

import brian2 as brian
import pytest

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.classic_sector import first_order_population_parameters
from smart_robustness.models.compartmental_hh import create_compartmental_hh_population
from smart_robustness.models.modeldb112923 import first_order_population_facts
from smart_robustness.models.sst_like_feedback import (
    L5_SST_LIKE_PORT_NAME,
    L5_SST_LIKE_ROUTE_ID,
    connect_l5_sst_like_feedback,
    create_l5_sst_like_synapse,
    make_l5_sst_like_population_factory,
    make_l5_sst_like_sector_builder,
)


def _l5_parameters():
    baseline = load_frozen_classic_baseline(
        "configs/baselines/classic_smart_calibrated_v1.yaml"
    )
    fact = next(
        item
        for item in first_order_population_facts()
        if item.canonical_name == "layer5_excitatory_v1"
    )
    return first_order_population_parameters(
        fact, conventions=baseline.runtime_conventions()
    )


def test_zero_resource_builder_is_an_exact_passthrough() -> None:
    sentinel = object()
    calls = []

    def base_builder(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    for delay_ms in (1.0, 3.0, 7.0):
        builder = make_l5_sst_like_sector_builder(
            base_builder=base_builder,
            total_conductance_nS=0.0,
            delay_ms=delay_ms,
        )
        marker = object()
        assert builder(marker, registered=True) is sentinel
    assert calls == [
        ((calls[index][0][0],), {"registered": True}) for index in range(3)
    ]
    assert all("population_factory" not in kwargs for _, kwargs in calls)


def test_nonzero_factory_adds_only_one_exactly_normalized_l5_port() -> None:
    calls = []

    def base_factory(**kwargs):
        calls.append(kwargs)
        return kwargs

    total_nS = 48.32162200302801
    factory = make_l5_sst_like_population_factory(
        total_conductance_nS=total_nS,
        base_factory=base_factory,
    )
    unrelated = {"synaptic_ports": ()}
    result = factory(name="smart_v1_trn", params=unrelated, size=1, brian=object())
    assert result["params"] is unrelated
    original = _l5_parameters()
    result = factory(
        name="smart_v1_layer5_excitatory_v1",
        params=original,
        size=3,
        brian=object(),
    )
    transformed = result["params"]
    assert transformed is not original
    assert transformed["synaptic_ports"][:-1] == original["synaptic_ports"]
    port = transformed["synaptic_ports"][-1]
    assert port.record_id == L5_SST_LIKE_ROUTE_ID
    assert port.compartment == "distal_dendrite"
    area = transformed["cell_spec"].compartment(port.compartment).lateral_area_cm2
    assert port.conductance_density_mS_cm2 * area * 1e6 == pytest.approx(total_nS)


def test_nonzero_route_is_same_index_and_registered_on_network() -> None:
    brian.start_scope()
    brian.prefs.codegen.target = "numpy"
    total_nS = 24.160811001514005
    factory = make_l5_sst_like_population_factory(
        total_conductance_nS=total_nS,
        base_factory=create_compartmental_hh_population,
    )
    population = factory(
        name="test_layer5_excitatory_v1",
        size=3,
        params=_l5_parameters(),
        brian=brian,
    )
    network = brian.Network(population.group)
    sector = SimpleNamespace(
        populations={"layer5_excitatory_v1": population},
        projections={},
        network=network,
    )
    result = connect_l5_sst_like_feedback(
        sector,
        total_conductance_nS=total_nS,
        delay_ms=3.0,
        brian=brian,
    )
    assert result is sector
    synapse = sector.projections[L5_SST_LIKE_ROUTE_ID]
    assert list(synapse.i[:]) == [0, 1, 2]
    assert list(synapse.j[:]) == [0, 1, 2]
    assert all(synapse.delay[:] == 3.0 * brian.ms)
    assert "delivered" in synapse.variables
    assert "pre_signal" in synapse.variables
    assert f"{L5_SST_LIKE_PORT_NAME}_rise_post" not in synapse.variables
    assert f"{L5_SST_LIKE_PORT_NAME}_fall_post" not in synapse.variables
    assert synapse in network.objects
    network.run(0 * brian.ms)


def test_connector_rejects_an_unregistered_port() -> None:
    brian.start_scope()
    brian.prefs.codegen.target = "numpy"
    factory = make_l5_sst_like_population_factory(
        total_conductance_nS=24.160811001514005,
        base_factory=create_compartmental_hh_population,
    )
    population = factory(
        name="test_layer5_excitatory_v1",
        size=1,
        params=_l5_parameters(),
        brian=brian,
    )
    unrelated = _l5_parameters()["synaptic_ports"][0]
    with pytest.raises(ValueError, match="registered port"):
        create_l5_sst_like_synapse(
            pre_group=population.group,
            post_population=population,
            port=unrelated,
            delay_ms=3.0,
            brian=brian,
            name="invalid_sst_like_connector",
        )


@pytest.mark.parametrize(
    ("total_nS", "delay_ms"),
    [(-1.0, 3.0), (float("nan"), 3.0), (1.0, 0.0), (1.0, float("inf"))],
)
def test_route_parameters_fail_closed(total_nS: float, delay_ms: float) -> None:
    with pytest.raises(ValueError):
        make_l5_sst_like_sector_builder(
            base_builder=lambda **_: None,
            total_conductance_nS=total_nS,
            delay_ms=delay_ms,
        )
