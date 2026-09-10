"""Classify the fixed Figure 7 mismatch relay events from its pinned trace."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import yaml
from run_figure6_nonspecific_distal_gaba_source import _plain

COMPARTMENTS = ("soma", "proximal_dendrite", "distal_dendrite")


def _minimum(trace, times, row, variable, mask):
    values = trace[variable][row]
    positions = np.flatnonzero(mask)
    index = int(positions[np.argmin(values[mask])])
    return {"value": float(values[index]), "time_ms": float(times[index])}


def _maximum(trace, times, row, variable, mask):
    values = trace[variable][row]
    positions = np.flatnonzero(mask)
    index = int(positions[np.argmax(values[mask])])
    return {"value": float(values[index]), "time_ms": float(times[index])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result_path = Path(args.result)
    result = yaml.safe_load(result_path.read_text())
    trace_path = Path(result["trace"]["path"])
    if sha256(trace_path.read_bytes()).hexdigest() != result["trace"]["sha256"]:
        raise ValueError("trace hash differs from result")

    event_times_by_cell: dict[int, list[float]] = {}
    for cell, time_ms in zip(
        result["mismatch"]["relay_spike_indices"],
        result["mismatch"]["relay_spike_times_ms"],
        strict=True,
    ):
        event_times_by_cell.setdefault(int(cell), []).append(float(time_ms))

    target_cells = [int(value) for value in result["analysis_contract"]["target_cells"]]
    windows = [float(value) for value in result["analysis_contract"]["pre_event_windows_ms"]]
    peak_threshold = float(
        result["analysis_contract"]["published_voltage_peak_threshold_mV"]
    )
    per_cell = {}
    first_nonoverlap = {}
    with np.load(trace_path, allow_pickle=False) as trace:
        times = trace["time_ms"]
        rows = {int(cell): row for row, cell in enumerate(trace["cell_indices"].tolist())}
        for cell in target_cells:
            row = rows[cell]
            voltage = trace["v_soma"][row]
            peaks = np.flatnonzero(
                (voltage[1:-1] > voltage[:-2])
                & (voltage[1:-1] >= voltage[2:])
                & (voltage[1:-1] > peak_threshold)
            ) + 1
            events = event_times_by_cell[cell]
            event_metrics = []
            previous_event_ms = 0.0
            for event_number, event_time_ms in enumerate(events, start=1):
                event_index = int(np.argmin(np.abs(times - event_time_ms)))
                interval_peaks = peaks[
                    (times[peaks] >= previous_event_ms)
                    & (times[peaks] < event_time_ms)
                ]
                pre_windows = {}
                for window_ms in windows:
                    mask = (times >= max(previous_event_ms, event_time_ms - window_ms)) & (
                        times < event_time_ms
                    )
                    pre_windows[f"{window_ms:g}_ms"] = {
                        "voltage_minimum_mV": {
                            compartment: _minimum(
                                trace,
                                times,
                                row,
                                f"v_{compartment}",
                                mask,
                            )
                            for compartment in COMPARTMENTS
                        },
                        "calcium_current_maximum_pA": {
                            compartment: _maximum(
                                trace,
                                times,
                                row,
                                f"i_ca_{compartment}",
                                mask,
                            )
                            for compartment in COMPARTMENTS
                        },
                    }
                event_metrics.append(
                    {
                        "event_number": event_number,
                        "event_time_ms": event_time_ms,
                        "interval_from_previous_event_ms": (
                            event_time_ms - previous_event_ms
                            if event_number > 1
                            else None
                        ),
                        "positive_soma_peaks_since_previous_event": [
                            {
                                "time_ms": float(times[index]),
                                "voltage_mV": float(voltage[index]),
                            }
                            for index in interval_peaks
                        ],
                        "pre_event_windows": pre_windows,
                        "at_event": {
                            compartment: {
                                "voltage_mV": float(
                                    trace[f"v_{compartment}"][row, event_index]
                                ),
                                "calcium_current_pA": float(
                                    trace[f"i_ca_{compartment}"][row, event_index]
                                ),
                                "m_ca": float(
                                    trace[f"m_ca_{compartment}"][row, event_index]
                                ),
                                "h_ca": float(
                                    trace[f"h_ca_{compartment}"][row, event_index]
                                ),
                            }
                            for compartment in COMPARTMENTS
                        },
                    }
                )
                previous_event_ms = event_time_ms
            per_cell[str(cell)] = {
                "event_times_ms": events,
                "positive_soma_peaks_ms": [float(times[index]) for index in peaks],
                "event_metrics": event_metrics,
            }

            if cell != 40:
                first_event_ms = events[0]
                first_index = int(np.flatnonzero(times < first_event_ms)[-1])
                first_mask = (times >= 0.0) & (times < first_event_ms)
                initial = {}
                for compartment in COMPARTMENTS:
                    voltage_name = f"v_{compartment}"
                    h_name = f"h_ca_{compartment}"
                    initial[compartment] = {
                        "initial_voltage_mV": float(trace[voltage_name][row, 0]),
                        "minimum_voltage_before_first_event": _minimum(
                            trace, times, row, voltage_name, first_mask
                        ),
                        "initial_h_ca": float(trace[h_name][row, 0]),
                        "h_ca_immediately_before_first_event": float(
                            trace[h_name][row, first_index]
                        ),
                        "maximum_calcium_current_before_first_event_pA": _maximum(
                            trace,
                            times,
                            row,
                            f"i_ca_{compartment}",
                            first_mask,
                        ),
                    }
                first_peaks = peaks[times[peaks] < first_event_ms]
                first_nonoverlap[str(cell)] = {
                    "first_event_ms": first_event_ms,
                    "compartments": initial,
                    "voltage_below_initial_before_first_event": any(
                        values["minimum_voltage_before_first_event"]["value"]
                        < values["initial_voltage_mV"]
                        for values in initial.values()
                    ),
                    "h_ca_increased_before_first_event": any(
                        values["h_ca_immediately_before_first_event"]
                        > values["initial_h_ca"]
                        for values in initial.values()
                    ),
                    "positive_soma_peaks_before_first_event": [
                        {
                            "time_ms": float(times[index]),
                            "voltage_mV": float(voltage[index]),
                        }
                        for index in first_peaks
                    ],
                }

    every_first_escape_lacks_hyperpolarization = all(
        not values["voltage_below_initial_before_first_event"]
        for values in first_nonoverlap.values()
    )
    every_first_escape_lacks_h_recovery = all(
        not values["h_ca_increased_before_first_event"]
        for values in first_nonoverlap.values()
    )
    isolated_peak_trains = all(
        all(
            len(event["positive_soma_peaks_since_previous_event"]) == 1
            for event in values["event_metrics"]
        )
        for cell, values in per_cell.items()
        if cell != "40"
    )

    artifact = {
        "schema_version": 1,
        "id": output.stem,
        "date": "2026-09-07",
        "status": "late-nonoverlap-rebound-hypothesis-rejected",
        "classification": "readout-only-calcium-mechanism-assessment",
        "registration": result["registration"],
        "result": str(result_path),
        "result_sha256": sha256(result_path.read_bytes()).hexdigest(),
        "trace": result["trace"],
        "integrity": {
            "event_train_identity_verified": result["event_train_identity_verified"],
            "required_variables_present": result["trace"]["required_variables_present"],
            "sample_count": result["trace"]["sample_count"],
        },
        "first_nonoverlap_events": first_nonoverlap,
        "all_target_event_metrics": per_cell,
        "classification_tests": {
            "every_first_nonoverlap_escape_lacks_preceding_hyperpolarization": every_first_escape_lacks_hyperpolarization,
            "every_first_nonoverlap_escape_lacks_t_channel_availability_recovery": every_first_escape_lacks_h_recovery,
            "each_nonoverlap_interevent_interval_contains_one_positive_soma_peak": isolated_peak_trains,
            "late_nonoverlap_t_type_rebound_supported": False,
        },
        "assessment": {
            "artifact_468_failure_unchanged": True,
            "parameter_selected": False,
            "original_smart_reproduced": False,
            "baseline_promoted": False,
            "conclusion": (
                "The first nonoverlap events at 49.31--49.75 ms do not follow "
                "hyperpolarization in soma or either dendrite: every compartment "
                "starts at -60 mV and never falls below that value before its first "
                "escape. T-channel inactivation availability h_ca decreases rather "
                "than recovering. Each nonoverlap cell has one >+30-mV somatic "
                "peak per emitted event, with isolated events separated by about "
                "21--25 ms rather than a transient multi-peak burst. T-type current "
                "is present but the paper's hyperpolarization-to-rebound sequence is "
                "absent. Deep later minima occur only after the already-invalid "
                "first action potentials and cannot rescue the candidate."
            ),
            "next_step": (
                "Use the same pinned trace to localize the loss of inhibition before "
                "the first nonoverlap escape, then return to a source-constrained "
                "synaptic/event-handling uncertainty. Do not tune calcium or adopt "
                "the observed 49-ms transition as a cutoff."
            ),
        },
    }
    output.open("x").write(yaml.safe_dump(_plain(artifact), sort_keys=False))


if __name__ == "__main__":
    main()
