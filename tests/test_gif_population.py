from __future__ import annotations

import numpy as np
import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.classic_sector import build_full_smart_network
from smart_robustness.models.gif import (
    create_somatic_gif_population,
    load_gif_parameter_map,
)
from smart_robustness.models.gif_parameters import (
    LITERATURE_EXCITATORY,
    LITERATURE_INHIBITORY,
    GIFParameters,
    literature_gif_parameters,
)
from smart_robustness.validation.gif_isolated_cell_matching import (
    generate_gif_sobol_candidates,
    run_gif_candidate_batch,
)
from smart_robustness.validation.isolated_cell_matching import CurrentStepProtocol


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


def test_gif_parameter_validation_and_literature_mapping() -> None:
    assert literature_gif_parameters("layer4_excitatory_v1") == LITERATURE_EXCITATORY
    assert literature_gif_parameters("thalamic_relay") == LITERATURE_EXCITATORY
    assert literature_gif_parameters("layer4_inhibitory_v1") == LITERATURE_INHIBITORY
    assert literature_gif_parameters("trn") == LITERATURE_INHIBITORY
    with pytest.raises(ValueError, match="stochasticity_mV must be positive"):
        GIFParameters.from_mapping(LITERATURE_EXCITATORY.as_dict() | {"stochasticity_mV": 0})


def test_somatic_gif_spikes_and_retains_smart_compartments() -> None:
    brian.start_scope()
    brian.seed(1234)
    brian.prefs.codegen.target = "numpy"
    brian.defaultclock.dt = 0.01 * brian.ms
    population = create_somatic_gif_population(
        name="gif_layer4", size=2, params=_params("layer4_excitatory"), brian=brian
    )
    spikes = brian.SpikeMonitor(population.group)
    voltage = brian.StateMonitor(population.group, "v_soma", record=True)
    population.group.i_drive_soma = 1000 * brian.pA
    brian.Network(population.group, spikes, voltage).run(20 * brian.ms)

    assert population.compiled.somatic_spike_model == "gif"
    assert population.compartments == ("soma", "proximal_dendrite")
    assert "eta_fast_gif" in population.group.variables
    assert "gamma_fast_gif" in population.group.variables
    assert "lambda_gif" in population.group.variables
    assert "i_na_soma" not in population.group.variables
    assert len(spikes.t) > 0
    assert np.all(np.isfinite(np.asarray(voltage.v_soma / brian.mV)))


def test_gif_retains_t_current_active_dendrite_and_ahp() -> None:
    brian.start_scope()
    relay = create_somatic_gif_population(
        name="gif_relay", size=1, params=_params("thalamic_relay"), brian=brian
    )
    assert "i_ca_proximal_dendrite" in relay.group.variables
    assert "i_ca_distal_dendrite" in relay.group.variables

    brian.start_scope()
    layer5 = create_somatic_gif_population(
        name="gif_layer5", size=1, params=_params("layer5_excitatory"), brian=brian
    )
    assert "i_na_soma" not in layer5.group.variables
    assert "i_na_distal_dendrite" in layer5.group.variables
    assert "i_ahp" in layer5.group.variables


def test_gif_factory_builds_all_24_full_smart_populations() -> None:
    brian.start_scope()
    brian.prefs.codegen.target = "numpy"
    sector = build_full_smart_network(
        projection_ids=frozenset(),
        population_factory=create_somatic_gif_population,
        brian=brian,
    )
    assert len(sector.populations) == 24
    assert sector.cell_count == 1624
    assert sector.compartment_count == 3900
    assert {
        population.compiled.somatic_spike_model
        for population in sector.populations.values()
    } == {"gif"}
    sector.network.run(0 * brian.ms)


def test_gif_candidate_batch_is_seed_reproducible() -> None:
    bounds = {
        "threshold_offset_mV": (10.0, 40.0),
        "reset_offset_mV": (-15.0, 35.0),
        "eta_fast_pA": (-50.0, 150.0),
        "eta_slow_pA": (-50.0, 100.0),
        "gamma_fast_mV": (-10.0, 30.0),
        "gamma_slow_mV": (-10.0, 20.0),
    }
    candidates = generate_gif_sobol_candidates(
        bounds, base=LITERATURE_EXCITATORY, count=2
    )
    protocol = CurrentStepProtocol(
        training_currents_pA=(100.0,),
        holdout_currents_pA=(200.0,),
        pre_ms=2.0,
        step_ms=5.0,
        post_ms=1.0,
        dt_ms=0.02,
    )
    first = run_gif_candidate_batch(
        candidates=candidates,
        population_params=_params("layer4_excitatory"),
        currents_pA=(0.0, 1000.0),
        protocol=protocol,
        seed=991,
        brian=brian,
    )
    second = run_gif_candidate_batch(
        candidates=candidates,
        population_params=_params("layer4_excitatory"),
        currents_pA=(0.0, 1000.0),
        protocol=protocol,
        seed=991,
        brian=brian,
    )
    assert first == second


def test_frozen_gif_parameter_map_covers_all_source_classes() -> None:
    parameters = load_gif_parameter_map("configs/models/gif_current_step_matched_v1.yaml")
    assert len(parameters) == 12
    assert parameters["thalamic_relay"].threshold_offset_mV == pytest.approx(10.9375)
    assert parameters["trn"] == LITERATURE_INHIBITORY
