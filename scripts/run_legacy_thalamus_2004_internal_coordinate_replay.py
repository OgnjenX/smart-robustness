"""Replay the 2004 relay benchmark in its source-implied internal coordinate."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from run_legacy_thalamus_2004_replay import (
    _legacy_input_ports,
    _legacy_relay_cell,
    _set_port,
    _sha256,
)

from smart_robustness.models.compartmental_hh import create_compartmental_hh_population


def _model_params(method: str) -> dict[str, object]:
    """Return the archived cell in KInNeSS's internal-zero voltage axis."""

    return {
        "cell_spec": _legacy_relay_cell(),
        "cell_class": "thalamic_relay",
        "axial_convention": "kinness_serialized_edge",
        "leak_convention": "printed_zero",
        "voltage_coordinate": "absolute",
        "nak_rate_convention": "standard_traub_miles",
        "calcium_gate_convention": "modeldb_112923",
        "calcium_voltage_coordinate": "internal_zero_plus_serialized_leak",
        "gate_initialization_convention": "steady_state_at_initial_voltage",
        "membrane_initialization_convention": "kinness_internal_zero",
        "spike_event_coordinate": "absolute_physical",
        "spike_event_threshold_mV": -20.0,
        "spike_event_rule": "falling_threshold_crossing",
        "calcium_density_convention": "table3",
        "ahp_convention": "modeldb_112923",
        "specific_capacitance_uF_cm2": 1.0,
        "enable_ahp_ach": False,
        "external_input_ports": _legacy_input_ports(),
        "method": method,
        "e_na_mV": 50.0,
        "e_k_mV": -90.0,
        "e_ca_mV": 180.0,
    }


def _run_tonic(*, input_value: float, method: str, duration_ms: float, dt_ms: float, brian):
    brian.start_scope()
    brian.defaultclock.dt = dt_ms * brian.ms
    population = create_compartmental_hh_population(
        name="legacy_2004_internal_tonic",
        size=1,
        params=_model_params(method),
        brian=brian,
    )
    _set_port(population.group, "legacy_depolarization", "green", input_value)
    voltage = brian.StateMonitor(population.group, "v_soma", record=True)
    spikes = brian.SpikeMonitor(population.group)
    brian.Network(population.group, voltage, spikes).run(duration_ms * brian.ms)
    return (
        np.asarray(voltage.t / brian.ms, dtype=float),
        np.asarray(voltage.v_soma[0] / brian.mV, dtype=float),
        np.asarray(spikes.t / brian.ms, dtype=float),
    )


def _run_burst(
    *,
    input_value: float,
    method: str,
    duration_ms: float,
    hyperpolarization_on_ms: float,
    hyperpolarization_off_ms: float,
    dt_ms: float,
    brian,
):
    brian.start_scope()
    brian.defaultclock.dt = dt_ms * brian.ms
    population = create_compartmental_hh_population(
        name="legacy_2004_internal_burst",
        size=1,
        params=_model_params(method),
        brian=brian,
    )
    voltage = brian.StateMonitor(population.group, "v_soma", record=True)
    spikes = brian.SpikeMonitor(population.group)
    network = brian.Network(population.group, voltage, spikes)
    network.run(hyperpolarization_on_ms * brian.ms)
    _set_port(population.group, "legacy_hyperpolarization", "red", input_value)
    network.run((hyperpolarization_off_ms - hyperpolarization_on_ms) * brian.ms)
    _set_port(population.group, "legacy_hyperpolarization", "red", 0.0)
    network.run((duration_ms - hyperpolarization_off_ms) * brian.ms)
    return (
        np.asarray(voltage.t / brian.ms, dtype=float),
        np.asarray(voltage.v_soma[0] / brian.mV, dtype=float),
        np.asarray(spikes.t / brian.ms, dtype=float),
    )


def _load_archived_voltage(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path, dtype=float)
    return values[:, 0], values[:, 1]


def _window_metrics(
    candidate_time_ms: np.ndarray,
    candidate_voltage_mV: np.ndarray,
    archived_time_ms: np.ndarray,
    archived_voltage_mV: np.ndarray,
    start_ms: float,
    stop_ms: float,
) -> dict[str, float | None]:
    selected = (candidate_time_ms >= start_ms) & (candidate_time_ms < stop_ms)
    candidate = candidate_voltage_mV[selected]
    archived = np.interp(candidate_time_ms[selected], archived_time_ms, archived_voltage_mV)
    if candidate.size == 0 or not np.all(np.isfinite(candidate)):
        return {"candidate_mean_mV": None, "archived_mean_mV": None, "mean_error_mV": None, "rmse_mV": None, "correlation": None}
    correlation = None
    if np.std(candidate) > 0 and np.std(archived) > 0:
        correlation = float(np.corrcoef(candidate, archived)[0, 1])
    return {
        "candidate_mean_mV": float(np.mean(candidate)),
        "archived_mean_mV": float(np.mean(archived)),
        "mean_error_mV": float(np.mean(candidate) - np.mean(archived)),
        "rmse_mV": float(np.sqrt(np.mean((candidate - archived) ** 2))),
        "correlation": correlation,
    }


def _event_summary(events_ms: np.ndarray, delay_ms: float) -> dict[str, object]:
    delayed = np.asarray(events_ms, dtype=float) + delay_ms
    intervals = np.diff(delayed)
    summary: dict[str, object] = {
        "event_count": int(delayed.size),
        "median_interevent_interval_ms": float(np.median(intervals)) if intervals.size else None,
        "first_event_ms": float(delayed[0]) if delayed.size else None,
        "last_event_ms": float(delayed[-1]) if delayed.size else None,
    }
    if delayed.size <= 250:
        summary["event_times_ms"] = [float(value) for value in delayed]
    else:
        summary["first_twenty_event_times_ms"] = [float(value) for value in delayed[:20]]
        summary["last_twenty_event_times_ms"] = [float(value) for value in delayed[-20:]]
    return summary


def _evaluate(
    *,
    tonic_events_ms: np.ndarray,
    burst_events_ms: np.ndarray,
    tonic_voltage_mV: np.ndarray,
    burst_voltage_mV: np.ndarray,
    burst_preinput: dict[str, float | None],
    burst_hyperpolarized: dict[str, float | None],
    registration: dict[str, object],
) -> dict[str, object]:
    gates = registration["acceptance_gates"]
    targets = registration["archived_targets"]
    delay_ms = float(registration["protocol"]["axonal_delay_ms"])
    tonic_delayed = tonic_events_ms + delay_ms
    burst_delayed = burst_events_ms + delay_ms
    tonic_intervals = np.diff(tonic_delayed)
    tonic_median = float(np.median(tonic_intervals)) if tonic_intervals.size else None
    target_burst = np.asarray(targets["burst_event_times_ms"], dtype=float)
    burst_timing_error = (
        float(np.max(np.abs(burst_delayed - target_burst)))
        if burst_delayed.shape == target_burst.shape
        else None
    )
    preinput_error = burst_preinput["mean_error_mV"]
    hyper_error = burst_hyperpolarized["mean_error_mV"]
    checks = {
        "finite_voltage": bool(
            np.all(np.isfinite(tonic_voltage_mV)) and np.all(np.isfinite(burst_voltage_mV))
        ),
        "tonic_event_count": abs(int(tonic_delayed.size) - int(targets["tonic_event_count"]))
        <= int(gates["tonic_event_count_tolerance"]),
        "tonic_median_interevent_interval": tonic_median is not None
        and abs(tonic_median - float(targets["tonic_median_interevent_interval_ms"]))
        <= float(gates["tonic_median_interevent_interval_tolerance_ms"]),
        "burst_event_count": int(burst_delayed.size) == int(targets["burst_event_count"]),
        "burst_event_timing": burst_timing_error is not None
        and burst_timing_error <= float(gates["burst_max_event_time_error_ms"]),
        "burst_preinput_mean_voltage": preinput_error is not None
        and abs(float(preinput_error)) <= float(gates["burst_preinput_mean_error_mV"]),
        "burst_hyperpolarized_mean_voltage": hyper_error is not None
        and abs(float(hyper_error)) <= float(gates["burst_hyperpolarized_mean_error_mV"]),
    }
    return {
        "checks": checks,
        "joint_pass": all(checks.values()),
        "burst_max_event_time_error_ms": burst_timing_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    registration_path = Path(args.registration)
    registration = yaml.safe_load(registration_path.read_text())

    for source in registration["sources"].values():
        path = Path(source["path"])
        if _sha256(path) != source["sha256"]:
            raise ValueError(f"source differs from registration: {path}")
    script_path = Path(registration["script"])
    if _sha256(script_path) != registration["script_sha256"]:
        raise ValueError("runner differs from registration")

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    protocol = registration["protocol"]
    tonic_archive_time, tonic_archive_voltage = _load_archived_voltage(
        Path(registration["sources"]["tonic_voltage"]["path"])
    )
    burst_archive_time, burst_archive_voltage = _load_archived_voltage(
        Path(registration["sources"]["burst_voltage"]["path"])
    )
    outcomes = []
    for method in protocol["integration_methods"]:
        for input_mode, input_value in protocol["input_encodings"].items():
            tonic_time, tonic_voltage, tonic_events = _run_tonic(
                input_value=float(input_value),
                method=method,
                duration_ms=float(protocol["duration_ms"]),
                dt_ms=float(protocol["dt_ms"]),
                brian=brian,
            )
            burst_time, burst_voltage, burst_events = _run_burst(
                input_value=float(input_value),
                method=method,
                duration_ms=float(protocol["duration_ms"]),
                hyperpolarization_on_ms=float(protocol["hyperpolarization_on_ms"]),
                hyperpolarization_off_ms=float(protocol["hyperpolarization_off_ms"]),
                dt_ms=float(protocol["dt_ms"]),
                brian=brian,
            )
            tonic_windows = {
                "steady": _window_metrics(
                    tonic_time,
                    tonic_voltage,
                    tonic_archive_time,
                    tonic_archive_voltage,
                    100.0,
                    float(protocol["duration_ms"]),
                )
            }
            burst_windows = {
                "preinput": _window_metrics(
                    burst_time,
                    burst_voltage,
                    burst_archive_time,
                    burst_archive_voltage,
                    100.0,
                    1000.0,
                ),
                "hyperpolarized": _window_metrics(
                    burst_time,
                    burst_voltage,
                    burst_archive_time,
                    burst_archive_voltage,
                    1500.0,
                    2150.0,
                ),
                "rebound": _window_metrics(
                    burst_time,
                    burst_voltage,
                    burst_archive_time,
                    burst_archive_voltage,
                    float(protocol["hyperpolarization_off_ms"]),
                    2250.0,
                ),
            }
            evaluation = _evaluate(
                tonic_events_ms=tonic_events,
                burst_events_ms=burst_events,
                tonic_voltage_mV=tonic_voltage,
                burst_voltage_mV=burst_voltage,
                burst_preinput=burst_windows["preinput"],
                burst_hyperpolarized=burst_windows["hyperpolarized"],
                registration=registration,
            )
            outcomes.append(
                {
                    "integration_method": method,
                    "input_encoding": input_mode,
                    "input_value": float(input_value),
                    "tonic": {
                        **_event_summary(tonic_events, float(protocol["axonal_delay_ms"])),
                        "minimum_voltage_mV": float(np.min(tonic_voltage)),
                        "maximum_voltage_mV": float(np.max(tonic_voltage)),
                        "windows": tonic_windows,
                    },
                    "burst": {
                        **_event_summary(burst_events, float(protocol["axonal_delay_ms"])),
                        "minimum_voltage_mV": float(np.min(burst_voltage)),
                        "maximum_voltage_mV": float(np.max(burst_voltage)),
                        "windows": burst_windows,
                    },
                    "evaluation": evaluation,
                }
            )

    result = {
        "schema_version": 1,
        "id": registration["result_id"],
        "date": registration["date"],
        "status": "completed-internal-coordinate-replay",
        "classification": registration["classification"],
        "registration": str(registration_path),
        "archived_targets": registration["archived_targets"],
        "outcomes": outcomes,
        "passing_arms": [
            {
                "integration_method": outcome["integration_method"],
                "input_encoding": outcome["input_encoding"],
            }
            for outcome in outcomes
            if outcome["evaluation"]["joint_pass"]
        ],
        "selection_performed": False,
        "original_figure8_reproduced": False,
        "baseline_frozen": False,
    }
    Path(registration["result"]).write_text(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()
