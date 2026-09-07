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
from smart_robustness.validation.nonspecific_replay import (
    NONSPECIFIC_REPLAY_DYNAMIC_VARIABLES,
    NONSPECIFIC_REPLAY_FORCED_VARIABLES,
    NONSPECIFIC_REPLAY_MONITOR_VARIABLES,
    capture_nonspecific_initial_state,
    run_nonspecific_replay,
    write_nonspecific_replay_trace,
)


def _conventions():
    return replace(
        figure6_runtime_conventions(),
        spike_event_rule="falling_threshold_crossing",
        spike_event_threshold_mV=-20.0,
    )


def _population(conventions):
    fact = next(
        item
        for item in first_order_population_facts()
        if item.canonical_name == "thalamic_nonspecific"
    )
    return create_compartmental_hh_population(
        name="nonspecific_replay_source",
        size=1,
        params=first_order_population_parameters(fact, conventions=conventions),
        brian=brian,
    )


def test_nonspecific_gate_trace_replays_isolated_cell_losslessly(tmp_path):
    brian.start_scope()
    brian.prefs.codegen.target = "numpy"
    dt_ms = 0.01
    duration_ms = 0.2
    brian.defaultclock.dt = dt_ms * brian.ms
    conventions = _conventions()
    population = _population(conventions)
    group = population.group
    group.external_001_input_green = 600
    group.external_001_input_source_count = 5
    initial_state = capture_nonspecific_initial_state(group, brian=brian)

    samples = int(duration_ms / dt_ms)
    assignments = []
    for index, name in enumerate(NONSPECIFIC_REPLAY_FORCED_VARIABLES):
        if name == "port_001_gate":
            values = np.linspace(0.0, 0.8, samples)
        elif name == "port_004_gate":
            values = np.linspace(0.2, 0.0, samples)
        elif name == "external_001_input_green":
            values = np.full(samples, 600.0)
        elif name == "external_001_input_source_count":
            values = np.full(samples, 5.0)
        elif name.endswith("input_source_count"):
            values = np.ones(samples)
        else:
            values = np.zeros(samples)
        replay_name = f"source_{name}"
        unit = brian.pA if name.startswith("i_") else 1
        group.namespace[replay_name] = brian.TimedArray(
            values * unit, dt=dt_ms * brian.ms
        )
        assignments.append(f"{name} = {replay_name}(t)")
    driver = group.run_regularly(
        "\n".join(assignments), when="groups", order=-1
    )
    spikes = brian.SpikeMonitor(group)
    state = brian.StateMonitor(
        group,
        NONSPECIFIC_REPLAY_MONITOR_VARIABLES,
        record=True,
        when="thresholds",
        order=0,
    )
    brian.Network(group, driver, spikes, state).run(duration_ms * brian.ms)

    path = tmp_path / "nonspecific-replay.npz"
    write_nonspecific_replay_trace(
        state,
        spikes,
        path,
        initial_state=initial_state,
        stimulus_start_ms=0.0,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        condition="match",
        fingerprint=conventions.fingerprint,
        brian=brian,
    )
    result = run_nonspecific_replay(path, conventions=conventions, brian=brian)

    assert result.exact_spike_train
    assert result.calcium_conductance_scale == 1.0
    assert result.finite
    assert result.max_voltage_error_mV < 1e-12
    assert max(error for _, error in result.max_abs_error_by_variable) < 1e-12
    with np.load(path, allow_pickle=False) as archive:
        assert tuple(archive["dynamic_variable_names"]) == (
            NONSPECIFIC_REPLAY_DYNAMIC_VARIABLES
        )
        assert tuple(archive["forced_variable_names"]) == (
            NONSPECIFIC_REPLAY_FORCED_VARIABLES
        )
        np.testing.assert_allclose(archive["time_ms"], np.arange(samples) * dt_ms)


def test_nonspecific_replay_rejects_calcium_flag_type(tmp_path):
    with pytest.raises(TypeError, match="calcium ablation"):
        run_nonspecific_replay(
            tmp_path / "unused.npz",
            conventions=_conventions(),
            ablate_calcium=1,  # type: ignore[arg-type]
            brian=brian,
        )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        run_nonspecific_replay(
            tmp_path / "unused.npz",
            conventions=_conventions(),
            calcium_conductance_scale=-0.1,
            brian=brian,
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        run_nonspecific_replay(
            tmp_path / "unused.npz",
            conventions=_conventions(),
            ablate_calcium=True,
            calcium_conductance_scale=0.5,
            brian=brian,
        )
