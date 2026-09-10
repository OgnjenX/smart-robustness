"""Refine the missing 2004 relay inputs on the source's exact 8-bit grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import run_legacy_thalamus_2004_internal_coordinate_replay as replay
import run_legacy_thalamus_input_reconstruction as coarse
import yaml

from smart_robustness.models.compartmental_hh import (
    create_compartmental_hh_population,
)


def _set_port_vector(group, name: str, channel: str, values: np.ndarray) -> None:
    """Assign one source value per cell while clearing the other color gates."""

    for other in ("red", "green", "blue", "alpha"):
        setattr(group, f"{name}_input_{other}", values if other == channel else 0.0)
    setattr(group, f"{name}_input_source_count", 1.0)


def _event_summary_by_cell(spikes, size: int, delay_ms: float, brian) -> list[dict[str, object]]:
    indices = np.asarray(spikes.i, dtype=int)
    event_times_ms = np.asarray(spikes.t / brian.ms, dtype=float)
    return [
        replay._event_summary(event_times_ms[indices == index], delay_ms)
        for index in range(size)
    ]


def _run_tonic_grid(
    *, values: np.ndarray, method: str, protocol: dict[str, object], brian
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    brian.start_scope()
    brian.defaultclock.dt = float(protocol["dt_ms"]) * brian.ms
    population = create_compartmental_hh_population(
        name=f"legacy_2004_source_grid_tonic_{method}",
        size=len(values),
        params=coarse._model_params(method),
        brian=brian,
    )
    _set_port_vector(population.group, "legacy_depolarization", "green", values)
    voltage = brian.StateMonitor(population.group, "v_soma", record=True)
    spikes = brian.SpikeMonitor(population.group)
    brian.Network(population.group, voltage, spikes).run(
        float(protocol["duration_ms"]) * brian.ms
    )
    return (
        np.asarray(voltage.t / brian.ms, dtype=float),
        np.asarray(voltage.v_soma / brian.mV, dtype=float),
        _event_summary_by_cell(
            spikes, len(values), float(protocol["axonal_delay_ms"]), brian
        ),
    )


def _run_burst_grid(
    *, values: np.ndarray, method: str, protocol: dict[str, object], brian
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    brian.start_scope()
    brian.defaultclock.dt = float(protocol["dt_ms"]) * brian.ms
    population = create_compartmental_hh_population(
        name=f"legacy_2004_source_grid_burst_{method}",
        size=len(values),
        params=coarse._model_params(method),
        brian=brian,
    )
    voltage = brian.StateMonitor(population.group, "v_soma", record=True)
    spikes = brian.SpikeMonitor(population.group)
    network = brian.Network(population.group, voltage, spikes)
    on_ms = float(protocol["hyperpolarization_on_ms"])
    off_ms = float(protocol["hyperpolarization_off_ms"])
    duration_ms = float(protocol["duration_ms"])
    network.run(on_ms * brian.ms)
    _set_port_vector(population.group, "legacy_hyperpolarization", "red", values)
    network.run((off_ms - on_ms) * brian.ms)
    _set_port_vector(
        population.group,
        "legacy_hyperpolarization",
        "red",
        np.zeros_like(values),
    )
    network.run((duration_ms - off_ms) * brian.ms)
    return (
        np.asarray(voltage.t / brian.ms, dtype=float),
        np.asarray(voltage.v_soma / brian.mV, dtype=float),
        _event_summary_by_cell(
            spikes, len(values), float(protocol["axonal_delay_ms"]), brian
        ),
    )


def _summarize_tonic(
    *,
    values: np.ndarray,
    time_ms: np.ndarray,
    voltage_mV: np.ndarray,
    events: list[dict[str, object]],
    protocol: dict[str, object],
    archive_time: np.ndarray,
    archive_voltage: np.ndarray,
) -> list[dict[str, object]]:
    return [
        {
            "input_byte": round(input_value * 255.0),
            "input_value": float(input_value),
            **event_summary,
            "minimum_voltage_mV": float(np.min(voltage_mV[index])),
            "maximum_voltage_mV": float(np.max(voltage_mV[index])),
            "finite": bool(np.all(np.isfinite(voltage_mV[index]))),
            "calibration_window": replay._window_metrics(
                time_ms,
                voltage_mV[index],
                archive_time,
                archive_voltage,
                float(protocol["tonic_calibration_window_ms"][0]),
                float(protocol["tonic_calibration_window_ms"][1]),
            ),
        }
        for index, (input_value, event_summary) in enumerate(zip(values, events))
    ]


def _summarize_burst(
    *,
    values: np.ndarray,
    time_ms: np.ndarray,
    voltage_mV: np.ndarray,
    events: list[dict[str, object]],
    protocol: dict[str, object],
    archive_time: np.ndarray,
    archive_voltage: np.ndarray,
) -> list[dict[str, object]]:
    return [
        {
            "input_byte": round(input_value * 255.0),
            "input_value": float(input_value),
            **event_summary,
            "minimum_voltage_mV": float(np.min(voltage_mV[index])),
            "maximum_voltage_mV": float(np.max(voltage_mV[index])),
            "finite": bool(np.all(np.isfinite(voltage_mV[index]))),
            "preinput_window": replay._window_metrics(
                time_ms,
                voltage_mV[index],
                archive_time,
                archive_voltage,
                float(protocol["burst_preinput_window_ms"][0]),
                float(protocol["burst_preinput_window_ms"][1]),
            ),
            "calibration_window": replay._window_metrics(
                time_ms,
                voltage_mV[index],
                archive_time,
                archive_voltage,
                float(protocol["burst_calibration_window_ms"][0]),
                float(protocol["burst_calibration_window_ms"][1]),
            ),
        }
        for index, (input_value, event_summary) in enumerate(zip(values, events))
    ]


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
    protocol = registration["protocol"]
    tonic_values = np.asarray(protocol["tonic_input_bytes"], dtype=float) / 255.0
    burst_values = np.asarray(protocol["burst_input_bytes"], dtype=float) / 255.0
    tonic_archive_time, tonic_archive_voltage = replay._load_archived_voltage(
        Path(registration["sources"]["tonic_voltage"]["path"])
    )
    burst_archive_time, burst_archive_voltage = replay._load_archived_voltage(
        Path(registration["sources"]["burst_voltage"]["path"])
    )

    method_results = []
    for method in protocol["integration_methods"]:
        tonic_time, tonic_voltage, tonic_events = _run_tonic_grid(
            values=tonic_values, method=method, protocol=protocol, brian=brian
        )
        burst_time, burst_voltage, burst_events = _run_burst_grid(
            values=burst_values, method=method, protocol=protocol, brian=brian
        )
        tonic_outcomes = _summarize_tonic(
            values=tonic_values,
            time_ms=tonic_time,
            voltage_mV=tonic_voltage,
            events=tonic_events,
            protocol=protocol,
            archive_time=tonic_archive_time,
            archive_voltage=tonic_archive_voltage,
        )
        burst_outcomes = _summarize_burst(
            values=burst_values,
            time_ms=burst_time,
            voltage_mV=burst_voltage,
            events=burst_events,
            protocol=protocol,
            archive_time=burst_archive_time,
            archive_voltage=burst_archive_voltage,
        )
        tonic_selected = coarse._select_by_voltage(tonic_outcomes)
        burst_selected = coarse._select_by_voltage(burst_outcomes)
        method_results.append(
            {
                "integration_method": method,
                "tonic_outcomes": tonic_outcomes,
                "burst_outcomes": burst_outcomes,
                "voltage_selected": {
                    "tonic": tonic_selected,
                    "burst": burst_selected,
                },
                "held_out_evaluation": coarse._held_out_gates(
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
        "status": "completed-source-grid-input-refinement",
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
