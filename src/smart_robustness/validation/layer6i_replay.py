"""Lossless isolated replay for one SMART layer-6I cell.

The Figure 10 network can deliver different combinations of relay, layer-2/3,
and layer-5 excitation to layer 6I.  This module captures the exact receptor
gates and mismatch-onset intrinsic state seen by one connected-network cell,
then replays those gates into an isolated copy with the same runtime profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from numbers import Real
from pathlib import Path

import numpy as np

LAYER6I_REPLAY_PROJECTION_IDS = (
    "modeldb112923.projection.023",
    "modeldb112923.projection.024",
    "modeldb112923.projection.025",
)

LAYER6I_REPLAY_DYNAMIC_VARIABLES = (
    "v_soma",
    "v_proximal_dendrite",
    "m_soma",
    "h_soma",
    "n_soma",
    "previous_spike_voltage",
    "armed",
    "last_spike_onset",
    "drive_spikes_until_clear",
    "transmitter",
)

LAYER6I_REPLAY_FORCED_VARIABLES = (
    "port_000_gate",
    "port_001_gate",
    "port_002_gate",
    "i_syn_soma",
    "i_syn_proximal_dendrite",
    "i_drive_soma",
    "i_drive_proximal_dendrite",
    "clear_drive_on_spike",
)

LAYER6I_REPLAY_MONITOR_VARIABLES = (
    *LAYER6I_REPLAY_DYNAMIC_VARIABLES,
    *LAYER6I_REPLAY_FORCED_VARIABLES,
)


def _unit_for(name: str, brian):
    if name.startswith("v_") or name == "previous_spike_voltage":
        return brian.mV, "mV"
    if name.startswith("i_"):
        return brian.pA, "pA"
    if name == "last_spike_onset":
        return brian.ms, "ms"
    return 1, "dimensionless"


def capture_layer6i_initial_state(
    group,
    *,
    cell_index: int,
    brian,
) -> dict[str, float]:
    """Copy every intrinsic state needed to restart one layer-6I cell."""

    if isinstance(cell_index, bool) or not isinstance(cell_index, int):
        raise TypeError("layer-6I replay cell index must be an integer")
    if cell_index < 0 or cell_index >= len(group):
        raise ValueError("layer-6I replay cell index is out of range")
    missing = set(LAYER6I_REPLAY_DYNAMIC_VARIABLES) - set(group.variables)
    if missing:
        raise ValueError(f"layer-6I replay state is incomplete: {sorted(missing)}")
    snapshot: dict[str, float] = {}
    for name in LAYER6I_REPLAY_DYNAMIC_VARIABLES:
        unit, _ = _unit_for(name, brian)
        values = np.asarray(getattr(group, name) / unit)
        snapshot[name] = float(values[cell_index])
    return snapshot


def write_layer6i_replay_trace(
    monitor,
    spikes,
    path,
    *,
    initial_state: dict[str, float],
    cell_index: int,
    mismatch_start_ms: float,
    duration_ms: float,
    dt_ms: float,
    condition: str,
    fingerprint: str,
    brian,
) -> str:
    """Write a unit-labelled receptor-gate replay archive without overwrite."""

    if monitor.when != "thresholds" or monitor.order != 0:
        raise ValueError("layer-6I replay gates must be captured at thresholds order zero")
    if tuple(monitor.record_variables) != LAYER6I_REPLAY_MONITOR_VARIABLES:
        raise ValueError("layer-6I replay monitor variables do not match the schema")
    if tuple(int(value) for value in monitor.record) != (cell_index,):
        raise ValueError("layer-6I replay monitor must contain exactly the selected cell")
    if set(initial_state) != set(LAYER6I_REPLAY_DYNAMIC_VARIABLES):
        raise ValueError("layer-6I replay initial state does not match the schema")

    absolute_time_ms = np.asarray(monitor.t / brian.ms, dtype=float)
    relative_time_ms = absolute_time_ms - float(mismatch_start_ms)
    selected = (relative_time_ms >= -dt_ms / 2) & (relative_time_ms < duration_ms - dt_ms / 2)
    expected_samples = round(duration_ms / dt_ms)
    if int(np.count_nonzero(selected)) != expected_samples:
        raise ValueError("layer-6I replay trace must contain one sample per mismatch step")

    values: dict[str, np.ndarray] = {}
    units: list[str] = []
    for name in LAYER6I_REPLAY_MONITOR_VARIABLES:
        unit, label = _unit_for(name, brian)
        trace = np.asarray(getattr(monitor, name) / unit, dtype=float)
        values[name] = trace[:, selected].copy()
        units.append(label)
    for name in LAYER6I_REPLAY_DYNAMIC_VARIABLES:
        values[f"initial__{name}"] = np.asarray(initial_state[name])

    spike_times_ms = np.asarray(spikes.t / brian.ms, dtype=float)
    spike_indices = np.asarray(spikes.i, dtype=int)
    spike_selected = (
        (spike_indices == cell_index)
        & (spike_times_ms >= mismatch_start_ms - dt_ms / 2)
        & (spike_times_ms < mismatch_start_ms + duration_ms - dt_ms / 2)
    )
    values.update(
        schema_version=np.asarray(1),
        population=np.asarray("layer6i_excitatory_v1"),
        projection_ids=np.asarray(LAYER6I_REPLAY_PROJECTION_IDS),
        time_ms=relative_time_ms[selected].copy(),
        dt_ms=np.asarray(dt_ms),
        duration_ms=np.asarray(duration_ms),
        cell_index=np.asarray(cell_index),
        variable_names=np.asarray(LAYER6I_REPLAY_MONITOR_VARIABLES),
        variable_units=np.asarray(units),
        dynamic_variable_names=np.asarray(LAYER6I_REPLAY_DYNAMIC_VARIABLES),
        forced_variable_names=np.asarray(LAYER6I_REPLAY_FORCED_VARIABLES),
        source_spike_times_ms=(spike_times_ms[spike_selected] - mismatch_start_ms).copy(),
        mismatch_start_ms=np.asarray(mismatch_start_ms),
        condition=np.asarray(condition),
        runtime_fingerprint=np.asarray(fingerprint),
        monitor_when=np.asarray(monitor.when),
        monitor_order=np.asarray(monitor.order),
    )

    output = Path(path)
    with output.open("xb") as stream:
        np.savez_compressed(stream, **values)
    return sha256(output.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class Layer6iReplayResult:
    trace_path: str
    trace_sha256: str
    runtime_fingerprint: str
    source_spike_times_ms: tuple[float, ...]
    replay_spike_times_ms: tuple[float, ...]
    exact_spike_train: bool
    max_abs_error_by_variable: tuple[tuple[str, float], ...]
    finite: bool
    projection025_conductance_scale: float
    soma_peak_mV: float
    proximal_peak_mV: float

    @property
    def max_voltage_error_mV(self) -> float:
        return max(
            error
            for name, error in self.max_abs_error_by_variable
            if name.startswith("v_") or name == "previous_spike_voltage"
        )


def run_layer6i_replay(
    trace_path: str | Path,
    *,
    conventions,
    projection025_conductance_scale: float = 1.0,
    brian=None,
) -> Layer6iReplayResult:
    """Replay a captured input history into one isolated layer-6I cell.

    ``projection025_conductance_scale`` changes only the maximal conductance
    of the layer-5 AMPA port.  The captured receptor gate is left unchanged,
    keeping this sensitivity distinct from event-multiplicity alternatives.
    """

    if (
        isinstance(projection025_conductance_scale, bool)
        or not isinstance(projection025_conductance_scale, Real)
        or not isfinite(float(projection025_conductance_scale))
        or projection025_conductance_scale <= 0
    ):
        raise ValueError("projection-025 conductance scale must be finite and positive")
    projection025_conductance_scale = float(projection025_conductance_scale)

    if brian is None:
        import brian2 as brian

    from ..classic_sector import first_order_population_parameters
    from ..models.compartmental_hh import create_compartmental_hh_population
    from ..models.modeldb112923 import first_order_population_facts

    path = Path(trace_path)
    digest = sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["schema_version"]) != 1:
            raise ValueError("unsupported layer-6I replay schema")
        if str(archive["population"]) != "layer6i_excitatory_v1":
            raise ValueError("replay archive is not a layer-6I trace")
        fingerprint = str(archive["runtime_fingerprint"])
        if fingerprint != conventions.fingerprint:
            raise ValueError("layer-6I trace and runtime fingerprints differ")
        if tuple(archive["projection_ids"].tolist()) != LAYER6I_REPLAY_PROJECTION_IDS:
            raise ValueError("layer-6I replay projection identities differ")
        if tuple(archive["dynamic_variable_names"].tolist()) != (LAYER6I_REPLAY_DYNAMIC_VARIABLES):
            raise ValueError("layer-6I replay dynamic-state schema differs")
        if tuple(archive["forced_variable_names"].tolist()) != (LAYER6I_REPLAY_FORCED_VARIABLES):
            raise ValueError("layer-6I replay forced-input schema differs")
        dt_ms = float(archive["dt_ms"])
        duration_ms = float(archive["duration_ms"])
        time_ms = np.asarray(archive["time_ms"], dtype=float).copy()
        source_spike_times = np.asarray(archive["source_spike_times_ms"], dtype=float).copy()
        source_values = {
            name: np.asarray(archive[name], dtype=float).copy()
            for name in LAYER6I_REPLAY_MONITOR_VARIABLES
        }
        initial_state = {
            name: float(archive[f"initial__{name}"]) for name in LAYER6I_REPLAY_DYNAMIC_VARIABLES
        }

    if dt_ms <= 0 or duration_ms <= 0 or not np.all(np.isfinite(time_ms)):
        raise ValueError("layer-6I replay timing must be finite and positive")
    expected_samples = round(duration_ms / dt_ms)
    if time_ms.shape != (expected_samples,) or not np.allclose(
        time_ms, np.arange(expected_samples) * dt_ms, atol=dt_ms * 1e-6, rtol=0
    ):
        raise ValueError("layer-6I replay time grid is not a complete regular trial")
    if any(values.shape != (1, expected_samples) for values in source_values.values()):
        raise ValueError("layer-6I replay variables must have shape (one cell, time)")

    fact = next(
        item
        for item in first_order_population_facts()
        if item.canonical_name == "layer6i_excitatory_v1"
    )
    params = first_order_population_parameters(fact, conventions=conventions)
    brian.start_scope()
    brian.defaultclock.dt = dt_ms * brian.ms
    population = create_compartmental_hh_population(
        name="isolated_layer6i_replay", size=1, params=params, brian=brian
    )
    group = population.group
    group.g_port_002 = group.g_port_002 * projection025_conductance_scale
    for name, value in initial_state.items():
        unit, _ = _unit_for(name, brian)
        setattr(group, name, value * unit)

    assignments: list[str] = []
    for name in LAYER6I_REPLAY_FORCED_VARIABLES:
        unit, _ = _unit_for(name, brian)
        replay_name = f"replay_{name}"
        group.namespace[replay_name] = brian.TimedArray(
            source_values[name][0] * unit, dt=dt_ms * brian.ms
        )
        assignments.append(f"{name} = {replay_name}(t)")
    driver = group.run_regularly(
        "\n".join(assignments),
        dt=dt_ms * brian.ms,
        when="groups",
        order=-1,
        name="isolated_layer6i_replay_inputs",
    )
    spikes = brian.SpikeMonitor(group, name="isolated_layer6i_replay_spikes")
    state = brian.StateMonitor(
        group,
        LAYER6I_REPLAY_DYNAMIC_VARIABLES,
        record=True,
        when="thresholds",
        order=0,
        name="isolated_layer6i_replay_state",
    )
    brian.Network(group, driver, spikes, state).run(duration_ms * brian.ms)

    errors: list[tuple[str, float]] = []
    finite = True
    for name in LAYER6I_REPLAY_DYNAMIC_VARIABLES:
        unit, _ = _unit_for(name, brian)
        replay_values = np.asarray(getattr(state, name) / unit, dtype=float)
        finite = finite and bool(np.all(np.isfinite(replay_values)))
        errors.append((name, float(np.max(np.abs(replay_values - source_values[name])))))
    replay_spike_times = np.asarray(spikes.t / brian.ms, dtype=float)
    soma_values_mV = np.asarray(state.v_soma / brian.mV, dtype=float)
    proximal_values_mV = np.asarray(
        state.v_proximal_dendrite / brian.mV, dtype=float
    )
    return Layer6iReplayResult(
        trace_path=str(path),
        trace_sha256=digest,
        runtime_fingerprint=fingerprint,
        source_spike_times_ms=tuple(float(value) for value in source_spike_times),
        replay_spike_times_ms=tuple(float(value) for value in replay_spike_times),
        exact_spike_train=bool(np.array_equal(source_spike_times, replay_spike_times)),
        max_abs_error_by_variable=tuple(errors),
        finite=finite,
        projection025_conductance_scale=projection025_conductance_scale,
        soma_peak_mV=float(np.max(soma_values_mV)),
        proximal_peak_mV=float(np.max(proximal_values_mV)),
    )
