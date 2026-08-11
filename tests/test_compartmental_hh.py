from __future__ import annotations

import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.models.compartmental_hh import create_compartmental_hh_population


def _params(cell_class: str = "thalamic_relay") -> dict[str, str]:
    params = {
        "cell_class": cell_class,
        "axial_convention": "symmetric_cable",
        "leak_convention": "table3_reversal",
        "voltage_coordinate": "relative_to_table3_leak",
        "calcium_gate_convention": "reciprocal",
        "calcium_density_convention": "table3",
    }
    if cell_class == "layer5_excitatory":
        params["ahp_max_conductance_nS"] = 1.0
    return params


@pytest.mark.parametrize("cell_class", ["thalamic_relay", "layer4_excitatory", "layer5_excitatory"])
def test_vectorized_population_runs_and_exposes_published_compartments(cell_class: str) -> None:
    brian.start_scope()
    brian.defaultclock.dt = 0.01 * brian.ms
    population = create_compartmental_hh_population(
        name=f"test_{cell_class}", size=3, params=_params(cell_class), brian=brian
    )
    monitor = brian.StateMonitor(population.group, "v_soma", record=True)
    brian.Network(population.group, monitor).run(0.1 * brian.ms)
    assert population.compartments == tuple(
        compartment.name for compartment in population.cell_spec.compartments
    )
    assert brian.numpy_.isfinite(monitor.v_soma[:]).all()


def test_relay_uses_table3_calcium_density_without_silent_global_override() -> None:
    brian.start_scope()
    population = create_compartmental_hh_population(
        name="relay_table3_ca", size=1, params=_params(), brian=brian
    )
    expected = population.cell_spec.compartment("proximal_dendrite").conductance_nS("ca")
    assert population.group.g_ca_proximal_dendrite[0] / brian.nsiemens == pytest.approx(expected)


def test_spike_event_arms_above_30_and_emits_once_below_zero() -> None:
    brian.start_scope()
    brian.defaultclock.dt = 0.1 * brian.ms
    group = brian.NeuronGroup(
        1,
        "dv_soma/dt=0*volt/second : volt\narmed : 1",
        threshold="armed > 0.5 and v_soma < 0*mV",
        reset="armed = 0",
        events={"arm_spike": "armed < 0.5 and v_soma > 30*mV"},
        method="euler",
    )
    group.run_on_event("arm_spike", "armed=1", when="after_thresholds", order=1)
    spike_monitor = brian.SpikeMonitor(group)
    network = brian.Network(group, spike_monitor)
    group.v_soma = 31 * brian.mV
    network.run(0.1 * brian.ms)
    assert group.armed[0] == 1
    group.v_soma = -1 * brian.mV
    network.run(0.1 * brian.ms)
    assert spike_monitor.count[0] == 1
    network.run(0.2 * brian.ms)
    assert spike_monitor.count[0] == 1


def test_layer5_requires_source_unidentified_ahp_conductance_explicitly() -> None:
    brian.start_scope()
    params = _params("layer5_excitatory")
    del params["ahp_max_conductance_nS"]
    with pytest.raises(ValueError, match="explicit ahp_max_conductance_nS"):
        create_compartmental_hh_population(
            name="layer5_missing_ahp", size=1, params=params, brian=brian
        )


def test_layer5_spike_generates_ahp_and_ach_suppresses_it() -> None:
    brian.start_scope()
    brian.defaultclock.dt = 0.1 * brian.ms
    population = create_compartmental_hh_population(
        name="layer5_modulation", size=1, params=_params("layer5_excitatory"), brian=brian
    )
    group = population.group
    group.v_soma = -1 * brian.mV
    group.armed = 1
    network = brian.Network(group)
    network.run(0.1 * brian.ms)
    assert group.ahp_rise[0] > 0
    assert group.ahp_fall[0] > 0
    network.run(10 * brian.ms)
    unsuppressed = float(group.ahp_gate[0] * (1 - group.ach_gate[0]))
    population.trigger_ach()
    network.run(5.5 * brian.ms)
    suppressed = float(group.ahp_gate[0] * (1 - group.ach_gate[0]))
    assert group.ach_gate[0] > 0.9
    assert suppressed < unsuppressed
