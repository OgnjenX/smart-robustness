"""Network-blind fixed-event assay for the registered L5 SST-like route."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..classic_sector import FirstOrderRuntimeConventions, first_order_population_parameters
from ..models.compartmental_hh import create_compartmental_hh_population
from ..models.modeldb112923 import first_order_population_facts
from ..models.sst_like_feedback import (
    L5_SST_LIKE_PORT_NAME,
    create_l5_sst_like_synapse,
    l5_sst_like_port,
)

RESOURCE_ANCHOR_NS = 193.28648801211204
RESOURCE_FRACTIONS = (0.125, 0.25, 0.5, 1.0)
TOTAL_CONDUCTANCES_NS = tuple(RESOURCE_ANCHOR_NS * value for value in RESOURCE_FRACTIONS)
DELAYS_MS = (1.0, 3.0, 7.0)
DT_MS = (0.01, 0.005)
EMISSIONS_MS = np.array([50.0, 70.0, 90.0, 110.0, 130.0])
DURATION_MS = 220.0
CLAMP_MV = -55.0
EXPECTED_EVENT_COUNT = 5
RESOURCE_ABSOLUTE_TOLERANCE_NS = 1e-12
CONVERGENCE_RELATIVE_TOLERANCES = {
    "peak_distal_gate": 0.001,
    "integral_distal_gate_ms": 0.001,
    "peak_absolute_distal_current_pA": 0.01,
    "integral_absolute_distal_current_pA_ms": 0.01,
}
CLAMP_VOLTAGE_ABSOLUTE_TOLERANCE_MV = 1e-12


def l5_fact() -> Any:
    return next(
        fact
        for fact in first_order_population_facts()
        if fact.canonical_name == "layer5_excitatory_v1"
    )


def isolated_parameters(
    *,
    total_conductance_nS: float,
    conventions: FirstOrderRuntimeConventions | None = None,
) -> dict[str, Any]:
    """Return one clamped L5 cell with only the registered distal GABA port."""

    fact = l5_fact()
    params = first_order_population_parameters(
        fact, conventions=conventions or FirstOrderRuntimeConventions()
    )
    params["synaptic_ports"] = (
        l5_sst_like_port(
            cell_spec=fact.cell,
            total_conductance_nS=total_conductance_nS,
        ),
    )
    params["gap_junction_ports"] = ()
    params["external_input_ports"] = ()
    params["injection_ports"] = ()
    params["depletion_epsilon"] = None
    params["depletion_recovery_ms"] = None
    params["voltage_clamps_mV"] = {
        compartment: CLAMP_MV
        for compartment in ("soma", "proximal_dendrite", "distal_dendrite")
    }
    return params


def build_assay(
    *,
    baseline: Any,
    total_conductance_nS: float,
    delay_ms: float,
    dt_ms: float,
    brian: Any,
) -> dict[str, Any]:
    """Construct one isolated source-to-target assay without running it."""

    if dt_ms not in DT_MS:
        raise ValueError(f"unregistered time step: {dt_ms}")
    if delay_ms not in DELAYS_MS:
        raise ValueError(f"unregistered delay: {delay_ms}")
    if total_conductance_nS not in TOTAL_CONDUCTANCES_NS:
        raise ValueError(f"unregistered conductance: {total_conductance_nS}")
    brian.defaultclock.dt = dt_ms * brian.ms
    source = brian.SpikeGeneratorGroup(
        1,
        np.zeros(EXPECTED_EVENT_COUNT, dtype=int),
        EMISSIONS_MS * brian.ms,
        name="isolated_l5_sst_like_source",
    )
    target = create_compartmental_hh_population(
        name="isolated_l5_sst_like_target",
        size=1,
        params=isolated_parameters(
            total_conductance_nS=total_conductance_nS,
            conventions=baseline.runtime_conventions(),
        ),
        brian=brian,
    )
    port = target.compiled.synaptic_ports[0]
    synapse = create_l5_sst_like_synapse(
        pre_group=source,
        post_population=target,
        port=port,
        delay_ms=delay_ms,
        brian=brian,
        name="isolated_l5_sst_like_route",
    )
    state = brian.StateMonitor(
        target.group,
        (
            "v_soma",
            "v_proximal_dendrite",
            "v_distal_dendrite",
            f"{L5_SST_LIKE_PORT_NAME}_gate",
            f"i_{L5_SST_LIKE_PORT_NAME}",
        ),
        record=True,
        when="end",
        name="isolated_l5_sst_like_state",
    )
    source_spikes = brian.SpikeMonitor(source, name="isolated_l5_sst_like_source_spikes")
    target_spikes = brian.SpikeMonitor(
        target.group, name="isolated_l5_sst_like_target_spikes"
    )
    network = brian.Network(
        source,
        target.group,
        synapse,
        state,
        source_spikes,
        target_spikes,
    )
    return {
        "source": source,
        "target": target,
        "port": port,
        "synapse": synapse,
        "state": state,
        "source_spikes": source_spikes,
        "target_spikes": target_spikes,
        "network": network,
    }


def simulate(
    *, baseline: Any, total_conductance_nS: float, delay_ms: float, dt_ms: float
) -> dict[str, np.ndarray]:
    """Execute one deterministic isolated condition in a fresh Brian scope."""

    import brian2 as brian

    brian.start_scope()
    brian.prefs.codegen.target = "numpy"
    assay = build_assay(
        baseline=baseline,
        total_conductance_nS=total_conductance_nS,
        delay_ms=delay_ms,
        dt_ms=dt_ms,
        brian=brian,
    )
    assay["network"].run(DURATION_MS * brian.ms)
    state = assay["state"]
    target = assay["target"]
    synapse = assay["synapse"]
    realized_nS = float(
        getattr(target.group, f"g_{L5_SST_LIKE_PORT_NAME}")[0] / brian.nsiemens
    )
    return {
        "time_ms": np.asarray(state.t / brian.ms, dtype=float),
        "source_spike_times_ms": np.asarray(
            assay["source_spikes"].t / brian.ms, dtype=float
        ),
        "source_spike_indices": np.asarray(
            assay["source_spikes"].i, dtype=np.int64
        ),
        "target_spike_times_ms": np.asarray(
            assay["target_spikes"].t / brian.ms, dtype=float
        ),
        "target_spike_indices": np.asarray(
            assay["target_spikes"].i, dtype=np.int64
        ),
        "distal_gate": np.asarray(
            getattr(state, f"{L5_SST_LIKE_PORT_NAME}_gate")[0], dtype=float
        ),
        "distal_synaptic_current_pA": np.asarray(
            getattr(state, f"i_{L5_SST_LIKE_PORT_NAME}")[0] / brian.pA,
            dtype=float,
        ),
        "v_soma_mV": np.asarray(state.v_soma[0] / brian.mV, dtype=float),
        "v_proximal_dendrite_mV": np.asarray(
            state.v_proximal_dendrite[0] / brian.mV, dtype=float
        ),
        "v_distal_dendrite_mV": np.asarray(
            state.v_distal_dendrite[0] / brian.mV, dtype=float
        ),
        "delivery_count": np.asarray(synapse.delivered[:], dtype=np.int64),
        "requested_total_conductance_nS": np.array(total_conductance_nS),
        "realized_total_conductance_nS": np.array(realized_nS),
        "resource_fraction": np.array(total_conductance_nS / RESOURCE_ANCHOR_NS),
        "delay_ms": np.array(delay_ms),
        "dt_ms": np.array(dt_ms),
    }


def trace_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.asarray(arrays[key])
        digest.update(key.encode())
        digest.update(value.dtype.str.encode())
        digest.update(repr(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def exact_trace_repeat(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> bool:
    return first.keys() == second.keys() and all(
        np.asarray(first[key]).dtype == np.asarray(second[key]).dtype
        and np.array_equal(first[key], second[key])
        for key in first
    )


def source_timing_gate(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Test exact discrete-clock identity without comparing converted float bits."""

    dt_ms = float(arrays["dt_ms"])
    source_times = np.asarray(arrays["source_spike_times_ms"], dtype=float)
    source_indices = np.asarray(arrays["source_spike_indices"], dtype=np.int64)
    recorded_ticks = np.rint(source_times / dt_ms).astype(np.int64)
    expected_ticks = np.rint(EMISSIONS_MS / dt_ms).astype(np.int64)
    expected_indices = np.zeros(EXPECTED_EVENT_COUNT, dtype=np.int64)
    ticks_exact = bool(np.array_equal(recorded_ticks, expected_ticks))
    indices_exact = bool(np.array_equal(source_indices, expected_indices))
    maximum_float_error_ms = (
        float(np.max(np.abs(source_times - EMISSIONS_MS)))
        if source_times.shape == EMISSIONS_MS.shape
        else None
    )
    return {
        "recorded_source_ticks": recorded_ticks.tolist(),
        "expected_source_ticks": expected_ticks.tolist(),
        "source_clock_ticks_exact": ticks_exact,
        "source_indices_exact": indices_exact,
        "maximum_float_serialization_error_ms": maximum_float_error_ms,
        "pass": ticks_exact and indices_exact,
    }


def summarize(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Compute only the preregistered engineering metrics and gates."""

    numeric = [
        np.asarray(value)
        for value in arrays.values()
        if np.issubdtype(np.asarray(value).dtype, np.number)
    ]
    finite = all(bool(np.all(np.isfinite(value))) for value in numeric)
    time = np.asarray(arrays["time_ms"], dtype=float)
    gate = np.asarray(arrays["distal_gate"], dtype=float)
    current = np.asarray(arrays["distal_synaptic_current_pA"], dtype=float)
    requested = float(arrays["requested_total_conductance_nS"])
    realized = float(arrays["realized_total_conductance_nS"])
    source_times = np.asarray(arrays["source_spike_times_ms"], dtype=float)
    delivery_count = int(np.sum(arrays["delivery_count"]))
    voltages = {
        compartment: np.asarray(arrays[f"v_{compartment}_mV"], dtype=float)
        for compartment in ("soma", "proximal_dendrite", "distal_dendrite")
    }
    clamp_errors = {
        compartment: float(np.max(np.abs(values - CLAMP_MV)))
        for compartment, values in voltages.items()
    }
    timing = source_timing_gate(arrays)
    resource_error = abs(realized - requested)
    per_run_pass = bool(
        finite
        and timing["pass"]
        and source_times.size == EXPECTED_EVENT_COUNT
        and delivery_count == EXPECTED_EVENT_COUNT
        and resource_error <= RESOURCE_ABSOLUTE_TOLERANCE_NS
        and float(np.max(gate)) > 0
        and float(np.max(np.abs(current))) > 0
        and np.asarray(arrays["target_spike_times_ms"]).size == 0
        and max(clamp_errors.values()) <= CLAMP_VOLTAGE_ABSOLUTE_TOLERANCE_MV
    )
    return {
        "trace_sha256": trace_sha256(arrays),
        "all_numeric_arrays_finite": finite,
        "source_event_count": int(source_times.size),
        "source_times_exact": timing["pass"],
        "source_timing": timing,
        "delivery_count": delivery_count,
        "target_spike_count": int(
            np.asarray(arrays["target_spike_times_ms"]).size
        ),
        "requested_total_conductance_nS": requested,
        "realized_total_conductance_nS": realized,
        "resource_absolute_error_nS": resource_error,
        "peak_distal_gate": float(np.max(gate)),
        "integral_distal_gate_ms": float(np.trapz(gate, time)),
        "peak_absolute_distal_current_pA": float(np.max(np.abs(current))),
        "integral_absolute_distal_current_pA_ms": float(
            np.trapz(np.abs(current), time)
        ),
        "clamp_maximum_absolute_errors_mV": clamp_errors,
        "per_run_gates_pass": per_run_pass,
    }


def _relative_difference(first: float, second: float) -> float:
    scale = max(abs(first), abs(second), np.finfo(float).tiny)
    return abs(first - second) / scale


def numerical_gate(coarse: dict[str, Any], fine: dict[str, Any]) -> dict[str, Any]:
    relative_errors = {
        name: _relative_difference(float(coarse[name]), float(fine[name]))
        for name in CONVERGENCE_RELATIVE_TOLERANCES
    }
    clamp_error = max(
        abs(
            float(coarse["clamp_maximum_absolute_errors_mV"][compartment])
            - float(fine["clamp_maximum_absolute_errors_mV"][compartment])
        )
        for compartment in ("soma", "proximal_dendrite", "distal_dendrite")
    )
    return {
        "relative_errors": relative_errors,
        "relative_tolerances": dict(CONVERGENCE_RELATIVE_TOLERANCES),
        "clamp_error_difference_mV": clamp_error,
        "clamp_error_tolerance_mV": CLAMP_VOLTAGE_ABSOLUTE_TOLERANCE_MV,
        "pass": bool(
            all(
                relative_errors[name] <= tolerance
                for name, tolerance in CONVERGENCE_RELATIVE_TOLERANCES.items()
            )
            and clamp_error <= CLAMP_VOLTAGE_ABSOLUTE_TOLERANCE_MV
        ),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_trace(path: Path, arrays: dict[str, np.ndarray]) -> str:
    """Atomically publish one immutable, object-free compressed trace."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("object arrays are forbidden in raw traces")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".l5-sst-like-trace-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return file_sha256(path)


def load_trace(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    path = Path(path)
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"raw trace fingerprint mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}
