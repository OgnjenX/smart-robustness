from dataclasses import replace

import numpy as np
import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.classic_sector import (
    figure6_runtime_conventions,
    first_order_population_parameters,
)
from smart_robustness.models.compartmental_hh import create_compartmental_hh_population
from smart_robustness.models.modeldb112923 import first_order_population_facts
from smart_robustness.validation.layer6i_replay import (
    LAYER6I_REPLAY_DYNAMIC_VARIABLES,
    LAYER6I_REPLAY_FORCED_VARIABLES,
    LAYER6I_REPLAY_MONITOR_VARIABLES,
    capture_layer6i_initial_state,
    run_layer6i_replay,
    write_layer6i_replay_trace,
)


def _conventions():
    return replace(
        figure6_runtime_conventions(),
        spike_event_coordinate="absolute_physical",
        spike_event_rule="falling_threshold_crossing",
        spike_event_threshold_mV=-20.0,
    )


def _population(conventions):
    fact = next(
        item
        for item in first_order_population_facts()
        if item.canonical_name == "layer6i_excitatory_v1"
    )
    return create_compartmental_hh_population(
        name="layer6i_replay_source",
        size=1,
        params=first_order_population_parameters(fact, conventions=conventions),
        brian=brian,
    )


def test_layer6i_gate_trace_replays_isolated_cell_losslessly(tmp_path):
    brian.start_scope()
    brian.prefs.codegen.target = "numpy"
    dt_ms = 0.01
    duration_ms = 0.2
    brian.defaultclock.dt = dt_ms * brian.ms
    conventions = _conventions()
    population = _population(conventions)
    group = population.group
    initial_state = capture_layer6i_initial_state(group, cell_index=0, brian=brian)

    samples = int(duration_ms / dt_ms)
    assignments = []
    for name in LAYER6I_REPLAY_FORCED_VARIABLES:
        values = np.linspace(0.0, 0.5, samples) if name == "port_002_gate" else np.zeros(samples)
        replay_name = f"source_{name}"
        unit = brian.pA if name.startswith("i_") else 1
        group.namespace[replay_name] = brian.TimedArray(values * unit, dt=dt_ms * brian.ms)
        assignments.append(f"{name} = {replay_name}(t)")
    driver = group.run_regularly("\n".join(assignments), when="groups", order=-1)
    spikes = brian.SpikeMonitor(group)
    state = brian.StateMonitor(
        group,
        LAYER6I_REPLAY_MONITOR_VARIABLES,
        record=True,
        when="thresholds",
        order=0,
    )
    brian.Network(group, driver, spikes, state).run(duration_ms * brian.ms)

    path = tmp_path / "layer6i-replay.npz"
    write_layer6i_replay_trace(
        state,
        spikes,
        path,
        initial_state=initial_state,
        cell_index=0,
        mismatch_start_ms=0.0,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        condition="intact",
        fingerprint=conventions.fingerprint,
        brian=brian,
    )
    result = run_layer6i_replay(path, conventions=conventions, brian=brian)

    assert result.exact_spike_train
    assert result.finite
    assert result.max_voltage_error_mV < 1e-12
    assert max(error for _, error in result.max_abs_error_by_variable) < 1e-12
    with np.load(path, allow_pickle=False) as archive:
        assert tuple(archive["dynamic_variable_names"]) == (LAYER6I_REPLAY_DYNAMIC_VARIABLES)
        assert tuple(archive["forced_variable_names"]) == (LAYER6I_REPLAY_FORCED_VARIABLES)
        np.testing.assert_allclose(archive["time_ms"], np.arange(samples) * dt_ms)


def test_layer6i_replay_capture_rejects_invalid_cell_index():
    brian.start_scope()
    group = _population(_conventions()).group
    with pytest.raises(TypeError, match="must be an integer"):
        capture_layer6i_initial_state(group, cell_index=True, brian=brian)
    with pytest.raises(ValueError, match="out of range"):
        capture_layer6i_initial_state(group, cell_index=1, brian=brian)
