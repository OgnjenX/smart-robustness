from __future__ import annotations

import numpy as np
import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.classic_sector import build_full_smart_network
from smart_robustness.models.adex import (
    create_somatic_adex_population,
    load_adex_parameter_map,
)


def _params(cell_class: str) -> dict[str, object]:
    params: dict[str, object] = {
        "cell_class": cell_class,
        "axial_convention": "symmetric_cable",
        "leak_convention": "table3_reversal",
        "voltage_coordinate": "relative_to_table3_leak",
        "nak_rate_convention": "printed_smart",
        "calcium_gate_convention": "reciprocal",
        "calcium_voltage_coordinate": "integrated_voltage",
        "gate_initialization_convention": "steady_state_at_initial_voltage",
        "membrane_initialization_convention": "physical_leak_voltage",
        "spike_event_coordinate": "absolute_physical",
        "spike_event_threshold_mV": 30.0,
        "spike_event_rule": "latched_peak_then_zero",
        "calcium_density_convention": "table3",
        "ahp_convention": "paper_text",
        "specific_capacitance_uF_cm2": 1.0,
        "enable_ahp_ach": cell_class == "layer5_excitatory",
    }
    if cell_class == "layer5_excitatory":
        params.update(
            {
                "ahp_max_conductance_nS": 1.0,
                "ahp_event_weight": 1.0,
                "e_ahp_mV": -90.0,
            }
        )
    return params


def test_somatic_adex_spikes_and_retains_smart_compartments() -> None:
    brian.start_scope()
    brian.prefs.codegen.target = "numpy"
    brian.defaultclock.dt = 0.01 * brian.ms
    population = create_somatic_adex_population(
        name="adex_layer4", size=2, params=_params("layer4_excitatory"), brian=brian
    )
    spikes = brian.SpikeMonitor(population.group)
    voltage = brian.StateMonitor(population.group, "v_soma", record=True)
    population.group.i_drive_soma = 1000 * brian.pA
    brian.Network(population.group, spikes, voltage).run(20 * brian.ms)

    assert population.compiled.somatic_spike_model == "adex"
    assert population.compartments == ("soma", "proximal_dendrite")
    assert "w_adex" in population.group.variables
    assert "i_na_soma" not in population.group.variables
    assert len(spikes.t) > 0
    assert np.all(np.isfinite(np.asarray(voltage.v_soma / brian.mV)))


def test_adex_substitution_retains_t_current_and_active_layer5_dendrite() -> None:
    brian.start_scope()
    relay = create_somatic_adex_population(
        name="adex_relay", size=1, params=_params("thalamic_relay"), brian=brian
    )
    assert "i_ca_proximal_dendrite" in relay.group.variables
    assert "i_ca_distal_dendrite" in relay.group.variables

    brian.start_scope()
    layer5 = create_somatic_adex_population(
        name="adex_layer5", size=1, params=_params("layer5_excitatory"), brian=brian
    )
    assert "i_na_soma" not in layer5.group.variables
    assert "i_na_distal_dendrite" in layer5.group.variables
    assert "i_ahp" in layer5.group.variables


def test_adex_factory_builds_all_24_full_smart_populations() -> None:
    brian.start_scope()
    brian.prefs.codegen.target = "numpy"
    sector = build_full_smart_network(
        projection_ids=frozenset(),
        population_factory=create_somatic_adex_population,
        brian=brian,
    )
    assert len(sector.populations) == 24
    assert sector.cell_count == 1624
    assert sector.compartment_count == 3900
    assert {
        population.compiled.somatic_spike_model
        for population in sector.populations.values()
    } == {"adex"}
    sector.network.run(0 * brian.ms)


def test_layered_parameter_manifest_overrides_only_relay() -> None:
    base = load_adex_parameter_map("configs/models/adex_current_step_matched_v1.yaml")
    extended = load_adex_parameter_map("configs/models/adex_relay_transfer_matched_v1.yaml")
    assert extended.keys() == base.keys()
    assert extended["thalamic_relay"].effective_leak_offset_mV == pytest.approx(-10.9375)
    assert extended["trn"] == base["trn"]
