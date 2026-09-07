"""Lossless input replay for the one-cell SMART nonspecific thalamus population.

The connected Figure 7 network can feed back through nonspecific thalamus.  A
calcium ablation in that network therefore cannot, by itself, establish a
cell-autonomous rebound mechanism.  This module preserves the target cell's
stimulus-onset state and the receptor gates actually seen by its membrane
integrator, then applies those same gates to an isolated copy of the cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np


NONSPECIFIC_REPLAY_PROJECTION_IDS = (
    "modeldb112923.projection.047",
    "modeldb112923.projection.048",
    "modeldb112923.projection.049",
    "modeldb112923.projection.050",
    "modeldb112923.projection.051",
)

NONSPECIFIC_REPLAY_DYNAMIC_VARIABLES = (
    "v_soma",
    "v_proximal_dendrite",
    "v_distal_dendrite",
    "m_soma",
    "h_soma",
    "n_soma",
    "m_ca_proximal_dendrite",
    "h_ca_proximal_dendrite",
    "m_ca_distal_dendrite",
    "h_ca_distal_dendrite",
    "previous_spike_voltage",
    "armed",
    "last_spike_onset",
    "drive_spikes_until_clear",
)

NONSPECIFIC_REPLAY_FORCED_VARIABLES = (
    "port_000_gate",
    "port_001_gate",
    "port_002_gate",
    "port_003_gate",
    "port_004_gate",
    "external_000_input_red",
    "external_000_input_green",
    "external_000_input_blue",
    "external_000_input_alpha",
    "external_000_input_source_count",
    "external_001_input_red",
    "external_001_input_green",
    "external_001_input_blue",
    "external_001_input_alpha",
    "external_001_input_source_count",
    "i_syn_soma",
    "i_syn_proximal_dendrite",
    "i_syn_distal_dendrite",
    "i_drive_soma",
    "i_drive_proximal_dendrite",
    "i_drive_distal_dendrite",
    "clear_drive_on_spike",
)

NONSPECIFIC_REPLAY_MONITOR_VARIABLES = (
    *NONSPECIFIC_REPLAY_DYNAMIC_VARIABLES,
    *NONSPECIFIC_REPLAY_FORCED_VARIABLES,
)


def _unit_for(name: str, brian):
    if name.startswith("v_") or name == "previous_spike_voltage":
        return brian.mV, "mV"
    if name.startswith("i_"):
        return brian.pA, "pA"
    if name == "last_spike_onset":
        return brian.ms, "ms"
    return 1, "dimensionless"


def capture_nonspecific_initial_state(group, *, brian) -> dict[str, float]:
    """Copy every intrinsic state needed to restart the target cell exactly."""

    missing = set(NONSPECIFIC_REPLAY_DYNAMIC_VARIABLES) - set(group.variables)
    if missing:
        raise ValueError(f"nonspecific replay state is incomplete: {sorted(missing)}")
    snapshot: dict[str, float] = {}
    for name in NONSPECIFIC_REPLAY_DYNAMIC_VARIABLES:
        unit, _ = _unit_for(name, brian)
        values = np.asarray(getattr(group, name) / unit)
        if values.shape != (1,):
            raise ValueError("nonspecific replay requires a one-cell population")
        snapshot[name] = float(values[0])
    return snapshot


def write_nonspecific_replay_trace(
    monitor,
    spikes,
    path,
    *,
    initial_state: dict[str, float],
    stimulus_start_ms: float,
    duration_ms: float,
    dt_ms: float,
    condition,
    fingerprint: str,
    brian,
) -> str:
    """Write a stimulus-only, unit-labelled replay archive without overwriting."""

    if monitor.when != "thresholds":
        raise ValueError("nonspecific replay gates must be captured at thresholds")
    if tuple(monitor.record_variables) != NONSPECIFIC_REPLAY_MONITOR_VARIABLES:
        raise ValueError("nonspecific replay monitor variables do not match the schema")
    if set(initial_state) != set(NONSPECIFIC_REPLAY_DYNAMIC_VARIABLES):
        raise ValueError("nonspecific replay initial state does not match the schema")

    absolute_time_ms = np.asarray(monitor.t / brian.ms, dtype=float)
    relative_time_ms = absolute_time_ms - float(stimulus_start_ms)
    selected = (relative_time_ms >= -dt_ms / 2) & (
        relative_time_ms < duration_ms - dt_ms / 2
    )
    expected_samples = int(round(duration_ms / dt_ms))
    if int(np.count_nonzero(selected)) != expected_samples:
        raise ValueError(
            "nonspecific replay trace does not contain exactly one sample per trial step"
        )

    values: dict[str, np.ndarray] = {}
    units: list[str] = []
    for name in NONSPECIFIC_REPLAY_MONITOR_VARIABLES:
        unit, label = _unit_for(name, brian)
        trace = np.asarray(getattr(monitor, name) / unit, dtype=float)
        values[name] = trace[:, selected].copy()
        units.append(label)
    for name in NONSPECIFIC_REPLAY_DYNAMIC_VARIABLES:
        values[f"initial__{name}"] = np.asarray(initial_state[name])

    spike_times_ms = np.asarray(spikes.t / brian.ms, dtype=float)
    spike_indices = np.asarray(spikes.i, dtype=int)
    spike_selected = (spike_times_ms >= stimulus_start_ms - dt_ms / 2) & (
        spike_times_ms < stimulus_start_ms + duration_ms - dt_ms / 2
    )
    values.update(
        schema_version=np.asarray(1),
        population=np.asarray("thalamic_nonspecific"),
        projection_ids=np.asarray(NONSPECIFIC_REPLAY_PROJECTION_IDS),
        time_ms=relative_time_ms[selected].copy(),
        dt_ms=np.asarray(dt_ms),
        duration_ms=np.asarray(duration_ms),
        cell_indices=np.asarray(monitor.record, dtype=int),
        variable_names=np.asarray(NONSPECIFIC_REPLAY_MONITOR_VARIABLES),
        variable_units=np.asarray(units),
        dynamic_variable_names=np.asarray(NONSPECIFIC_REPLAY_DYNAMIC_VARIABLES),
        forced_variable_names=np.asarray(NONSPECIFIC_REPLAY_FORCED_VARIABLES),
        source_spike_indices=spike_indices[spike_selected].copy(),
        source_spike_times_ms=(
            spike_times_ms[spike_selected] - stimulus_start_ms
        ).copy(),
        stimulus_start_ms=np.asarray(stimulus_start_ms),
        condition=np.asarray(str(condition)),
        runtime_fingerprint=np.asarray(fingerprint),
        monitor_when=np.asarray(monitor.when),
        monitor_order=np.asarray(monitor.order),
    )

    output = Path(path)
    with output.open("xb") as stream:
        np.savez_compressed(stream, **values)
    return sha256(output.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class NonspecificReplayResult:
    trace_path: str
    trace_sha256: str
    runtime_fingerprint: str
    calcium_ablated: bool
    source_spike_times_ms: tuple[float, ...]
    replay_spike_times_ms: tuple[float, ...]
    exact_spike_train: bool
    max_abs_error_by_variable: tuple[tuple[str, float], ...]
    finite: bool

    @property
    def max_voltage_error_mV(self) -> float:
        return max(
            error
            for name, error in self.max_abs_error_by_variable
            if name.startswith("v_") or name == "previous_spike_voltage"
        )


def run_nonspecific_replay(
    trace_path: str | Path,
    *,
    conventions,
    ablate_calcium: bool = False,
    brian=None,
) -> NonspecificReplayResult:
    """Replay one captured gate history into an isolated nonspecific cell."""

    if not isinstance(ablate_calcium, bool):
        raise TypeError("nonspecific replay calcium ablation must be boolean")
    if brian is None:
        import brian2 as brian

    from ..classic_sector import first_order_population_parameters
    from ..models.compartmental_hh import create_compartmental_hh_population
    from ..models.modeldb112923 import first_order_population_facts

    path = Path(trace_path)
    digest = sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["schema_version"]) != 1:
            raise ValueError("unsupported nonspecific replay schema")
        if str(archive["population"]) != "thalamic_nonspecific":
            raise ValueError("replay archive is not a nonspecific-thalamus trace")
        fingerprint = str(archive["runtime_fingerprint"])
        if fingerprint != conventions.fingerprint:
            raise ValueError("replay trace and runtime convention fingerprints differ")
        if tuple(archive["projection_ids"].tolist()) != NONSPECIFIC_REPLAY_PROJECTION_IDS:
            raise ValueError("replay trace projection identities differ from SMART 047--051")
        if tuple(archive["dynamic_variable_names"].tolist()) != NONSPECIFIC_REPLAY_DYNAMIC_VARIABLES:
            raise ValueError("replay trace dynamic-state schema differs")
        if tuple(archive["forced_variable_names"].tolist()) != NONSPECIFIC_REPLAY_FORCED_VARIABLES:
            raise ValueError("replay trace forced-input schema differs")
        dt_ms = float(archive["dt_ms"])
        duration_ms = float(archive["duration_ms"])
        time_ms = np.asarray(archive["time_ms"], dtype=float).copy()
        source_spike_times = np.asarray(
            archive["source_spike_times_ms"], dtype=float
        ).copy()
        source_values = {
            name: np.asarray(archive[name], dtype=float).copy()
            for name in NONSPECIFIC_REPLAY_MONITOR_VARIABLES
        }
        initial_state = {
            name: float(archive[f"initial__{name}"])
            for name in NONSPECIFIC_REPLAY_DYNAMIC_VARIABLES
        }

    if dt_ms <= 0 or duration_ms <= 0 or not np.all(np.isfinite(time_ms)):
        raise ValueError("replay timing must be finite and positive")
    expected_samples = int(round(duration_ms / dt_ms))
    if time_ms.shape != (expected_samples,) or not np.allclose(
        time_ms, np.arange(expected_samples) * dt_ms, atol=dt_ms * 1e-6, rtol=0
    ):
        raise ValueError("replay trace time grid is not a complete regular trial")
    if any(values.shape != (1, expected_samples) for values in source_values.values()):
        raise ValueError("replay trace variables must have shape (one cell, time)")

    fact = next(
        item
        for item in first_order_population_facts()
        if item.canonical_name == "thalamic_nonspecific"
    )
    params = first_order_population_parameters(fact, conventions=conventions)
    brian.start_scope()
    brian.defaultclock.dt = dt_ms * brian.ms
    population = create_compartmental_hh_population(
        name="isolated_nonspecific_replay", size=1, params=params, brian=brian
    )
    group = population.group
    for name, value in initial_state.items():
        unit, _ = _unit_for(name, brian)
        setattr(group, name, value * unit)
    if ablate_calcium:
        group.g_ca_proximal_dendrite = 0 * brian.nsiemens
        group.g_ca_distal_dendrite = 0 * brian.nsiemens

    assignments: list[str] = []
    for name in NONSPECIFIC_REPLAY_FORCED_VARIABLES:
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
        name="isolated_nonspecific_replay_inputs",
    )
    spikes = brian.SpikeMonitor(group, name="isolated_nonspecific_replay_spikes")
    state = brian.StateMonitor(
        group,
        NONSPECIFIC_REPLAY_DYNAMIC_VARIABLES,
        record=True,
        when="thresholds",
        order=0,
        name="isolated_nonspecific_replay_state",
    )
    brian.Network(group, driver, spikes, state).run(duration_ms * brian.ms)

    errors: list[tuple[str, float]] = []
    finite = True
    for name in NONSPECIFIC_REPLAY_DYNAMIC_VARIABLES:
        unit, _ = _unit_for(name, brian)
        replay_values = np.asarray(getattr(state, name) / unit, dtype=float)
        finite = finite and bool(np.all(np.isfinite(replay_values)))
        errors.append(
            (name, float(np.max(np.abs(replay_values - source_values[name]))))
        )
    replay_spike_times = np.asarray(spikes.t / brian.ms, dtype=float)
    return NonspecificReplayResult(
        trace_path=str(path),
        trace_sha256=digest,
        runtime_fingerprint=fingerprint,
        calcium_ablated=ablate_calcium,
        source_spike_times_ms=tuple(float(value) for value in source_spike_times),
        replay_spike_times_ms=tuple(float(value) for value in replay_spike_times),
        exact_spike_train=bool(np.array_equal(source_spike_times, replay_spike_times)),
        max_abs_error_by_variable=tuple(errors),
        finite=finite,
    )
