"""Reconstruct missing 2004 relay inputs from voltage, then validate spikes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import run_legacy_thalamus_2004_internal_coordinate_replay as replay
import yaml

_BASE_MODEL_PARAMS = replay._model_params


def _model_params(method: str) -> dict[str, object]:
    """Return the fixed, coordinate-complete relay implementation."""

    params = _BASE_MODEL_PARAMS(method)
    params.update(
        {
            "calcium_voltage_coordinate": "integrated_voltage",
            "e_na_mV": 120.0,
            "e_k_mV": -20.0,
            "e_ca_mV": 250.0,
            "spike_event_voltage_offset_mV": -70.0,
        }
    )
    return params


def _tonic_outcome(
    *,
    method: str,
    input_value: float,
    protocol: dict[str, object],
    archive_time: np.ndarray,
    archive_voltage: np.ndarray,
    brian,
) -> dict[str, object]:
    time_ms, voltage_mV, events_ms = replay._run_tonic(
        input_value=input_value,
        method=method,
        duration_ms=float(protocol["duration_ms"]),
        dt_ms=float(protocol["dt_ms"]),
        brian=brian,
    )
    steady = replay._window_metrics(
        time_ms,
        voltage_mV,
        archive_time,
        archive_voltage,
        float(protocol["tonic_calibration_window_ms"][0]),
        float(protocol["tonic_calibration_window_ms"][1]),
    )
    return {
        "input_value": input_value,
        **replay._event_summary(events_ms, float(protocol["axonal_delay_ms"])),
        "minimum_voltage_mV": float(np.min(voltage_mV)),
        "maximum_voltage_mV": float(np.max(voltage_mV)),
        "finite": bool(np.all(np.isfinite(voltage_mV))),
        "calibration_window": steady,
    }


def _burst_outcome(
    *,
    method: str,
    input_value: float,
    protocol: dict[str, object],
    archive_time: np.ndarray,
    archive_voltage: np.ndarray,
    brian,
) -> dict[str, object]:
    time_ms, voltage_mV, events_ms = replay._run_burst(
        input_value=input_value,
        method=method,
        duration_ms=float(protocol["duration_ms"]),
        hyperpolarization_on_ms=float(protocol["hyperpolarization_on_ms"]),
        hyperpolarization_off_ms=float(protocol["hyperpolarization_off_ms"]),
        dt_ms=float(protocol["dt_ms"]),
        brian=brian,
    )
    preinput = replay._window_metrics(
        time_ms,
        voltage_mV,
        archive_time,
        archive_voltage,
        float(protocol["burst_preinput_window_ms"][0]),
        float(protocol["burst_preinput_window_ms"][1]),
    )
    hyperpolarized = replay._window_metrics(
        time_ms,
        voltage_mV,
        archive_time,
        archive_voltage,
        float(protocol["burst_calibration_window_ms"][0]),
        float(protocol["burst_calibration_window_ms"][1]),
    )
    return {
        "input_value": input_value,
        **replay._event_summary(events_ms, float(protocol["axonal_delay_ms"])),
        "minimum_voltage_mV": float(np.min(voltage_mV)),
        "maximum_voltage_mV": float(np.max(voltage_mV)),
        "finite": bool(np.all(np.isfinite(voltage_mV))),
        "preinput_window": preinput,
        "calibration_window": hyperpolarized,
    }


def _select_by_voltage(outcomes: list[dict[str, object]]) -> dict[str, object]:
    finite = [
        outcome
        for outcome in outcomes
        if outcome["finite"] and outcome["calibration_window"]["mean_error_mV"] is not None
    ]
    return min(
        finite,
        key=lambda outcome: (
            abs(float(outcome["calibration_window"]["mean_error_mV"])),
            float(outcome["input_value"]),
        ),
    )


def _held_out_gates(
    *,
    tonic: dict[str, object],
    burst: dict[str, object],
    registration: dict[str, object],
) -> dict[str, object]:
    targets = registration["archived_targets"]
    gates = registration["held_out_acceptance_gates"]
    tonic_interval = tonic["median_interevent_interval_ms"]
    burst_times = np.asarray(burst.get("event_times_ms", []), dtype=float)
    target_burst_times = np.asarray(targets["burst_event_times_ms"], dtype=float)
    max_burst_error = (
        float(np.max(np.abs(burst_times - target_burst_times)))
        if burst_times.shape == target_burst_times.shape
        else None
    )
    checks = {
        "tonic_event_count": abs(int(tonic["event_count"]) - int(targets["tonic_event_count"]))
        <= int(gates["tonic_event_count_tolerance"]),
        "tonic_median_interevent_interval": tonic_interval is not None
        and abs(float(tonic_interval) - float(targets["tonic_median_interevent_interval_ms"]))
        <= float(gates["tonic_median_interevent_interval_tolerance_ms"]),
        "burst_event_count": int(burst["event_count"]) == int(targets["burst_event_count"]),
        "burst_event_timing": max_burst_error is not None
        and max_burst_error <= float(gates["burst_max_event_time_error_ms"]),
        "burst_preinput_mean_voltage": abs(
            float(burst["preinput_window"]["mean_error_mV"])
        )
        <= float(gates["burst_preinput_mean_error_mV"]),
    }
    return {
        "checks": checks,
        "joint_pass": all(checks.values()),
        "burst_max_event_time_error_ms": max_burst_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    registration_path = Path(args.registration)
    registration = yaml.safe_load(registration_path.read_text())

    for source in registration["sources"].values():
        path = Path(source["path"])
        if replay._sha256(path) != source["sha256"]:
            raise ValueError(f"source differs from registration: {path}")
    script_path = Path(registration["script"])
    if replay._sha256(script_path) != registration["script_sha256"]:
        raise ValueError("runner differs from registration")

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    replay._model_params = _model_params
    protocol = registration["protocol"]
    tonic_archive_time, tonic_archive_voltage = replay._load_archived_voltage(
        Path(registration["sources"]["tonic_voltage"]["path"])
    )
    burst_archive_time, burst_archive_voltage = replay._load_archived_voltage(
        Path(registration["sources"]["burst_voltage"]["path"])
    )

    method_results = []
    for method in protocol["integration_methods"]:
        tonic_outcomes = [
            _tonic_outcome(
                method=method,
                input_value=float(input_value),
                protocol=protocol,
                archive_time=tonic_archive_time,
                archive_voltage=tonic_archive_voltage,
                brian=brian,
            )
            for input_value in protocol["effective_input_grid"]
        ]
        burst_outcomes = [
            _burst_outcome(
                method=method,
                input_value=float(input_value),
                protocol=protocol,
                archive_time=burst_archive_time,
                archive_voltage=burst_archive_voltage,
                brian=brian,
            )
            for input_value in protocol["effective_input_grid"]
        ]
        tonic_selected = _select_by_voltage(tonic_outcomes)
        burst_selected = _select_by_voltage(burst_outcomes)
        method_results.append(
            {
                "integration_method": method,
                "tonic_outcomes": tonic_outcomes,
                "burst_outcomes": burst_outcomes,
                "voltage_selected": {
                    "tonic": tonic_selected,
                    "burst": burst_selected,
                },
                "held_out_evaluation": _held_out_gates(
                    tonic=tonic_selected,
                    burst=burst_selected,
                    registration=registration,
                ),
            }
        )

    result = {
        "schema_version": 1,
        "id": registration["result_id"],
        "date": registration["date"],
        "status": "completed-calibrated-input-protocol-reconstruction",
        "classification": registration["classification"],
        "registration": str(registration_path),
        "archived_targets": registration["archived_targets"],
        "method_results": method_results,
        "passing_methods": [
            outcome["integration_method"]
            for outcome in method_results
            if outcome["held_out_evaluation"]["joint_pass"]
        ],
        "selection_basis": "voltage means only",
        "spike_outputs_used_for_selection": False,
        "exact_legacy_protocol_recovered": False,
        "original_figure8_reproduced": False,
        "baseline_frozen": False,
    }
    Path(registration["result"]).write_text(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()
