"""Summarize preserved relay diagnostics without assigning a burst mechanism."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import yaml


def summarize(result: dict) -> dict:
    indices = result["relay_spike_indices"]
    times = result["relay_spike_times_ms"]
    if len(indices) != len(times):
        raise ValueError("relay event indices and times must have equal lengths")
    if not np.all(np.isfinite(times)):
        raise ValueError("relay event times must be finite")
    event_pairs = set(zip(indices, times, strict=True))
    currents = result.get("relay_pre_event_current_samples_pA", [])
    voltages = result.get("relay_pre_event_voltage_samples_mV", [])
    for sample in [*currents, *voltages]:
        index, time, offset, _, value = sample
        if (index, time) not in event_pairs:
            raise ValueError("diagnostic sample does not address a recorded event")
        if not np.isfinite(offset) or offset <= 0 or not np.isfinite(value):
            raise ValueError("diagnostic samples require finite values and positive offsets")
    cells = {}
    for index in sorted(set(indices)):
        event_times = sorted(t for i, t in zip(indices, times, strict=True) if i == index)
        cell_currents = [s for s in currents if s[0] == index]
        cell_voltages = [s for s in voltages if s[0] == index]
        labels = sorted({s[3] for s in cell_currents})
        cells[int(index)] = {
            "event_times_ms": event_times,
            "interevent_intervals_ms": np.diff(event_times).tolist(),
            "sample_offsets_ms": sorted({s[2] for s in cell_currents}),
            "sampled_current_ranges_pA_by_label": {
                label: [min(s[4] for s in cell_currents if s[3] == label),
                        max(s[4] for s in cell_currents if s[3] == label)]
                for label in labels
            },
            "positive_soma_samples_before_emitted_event": sum(
                s[3] == "soma" and s[4] > 0 for s in cell_voltages
            ),
        }
    return {
        "cells": cells,
        "interpretation": {
            "burst_mechanism_identified": False,
            "causal_calcium_contribution_identified": False,
            "limitations": [
                "Interevent intervals alone do not establish a T-type rebound mechanism.",
                "Sparse samples do not recover the preceding calcium inactivation history.",
                "An emitted falling-phase event is not the time of spike initiation.",
                "Currents at different compartments are not a single local current balance.",
                "No reproduction gate or model parameter is changed by this analysis.",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace")
    parser.add_argument("--reference")
    args = parser.parse_args()
    source = Path(args.input)
    payload = source.read_bytes()
    artifact = yaml.safe_load(payload)
    report = {
        "schema_version": 1,
        "classification": "post-hoc-descriptive-audit-no-new-simulation",
        "source_artifact": str(source),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        **summarize(artifact["mismatch_result"]),
    }
    if args.trace:
        if not args.reference:
            parser.error("--trace requires --reference for event-train verification")
        trace_path = Path(args.trace)
        expected = artifact["mismatch_result"]["relay_trace_sha256"]
        if hashlib.sha256(trace_path.read_bytes()).hexdigest() != expected:
            raise ValueError("trace checksum does not match the recorded result")
        reference = yaml.safe_load(Path(args.reference).read_bytes())
        with np.load(trace_path, allow_pickle=False) as trace:
            report["continuous_trace"] = summarize_trace(trace, artifact, reference)
        report["continuous_trace"].update(
            trace_path=str(trace_path), trace_sha256=expected,
            reference_artifact=args.reference,
        )
    Path(args.output).write_text(yaml.safe_dump(report, sort_keys=False))


def summarize_trace(trace, artifact: dict, reference: dict) -> dict:
    """Describe state ranges only after exact event-train and metadata checks."""
    if artifact["runtime_fingerprint"] != reference["runtime_fingerprint"]:
        raise ValueError("runtime fingerprint differs from the reference")
    actual, previous = artifact["mismatch_result"], reference["mismatch_result"]
    for population in ("relay", "trn", "nonspecific", "category"):
        for suffix in ("spike_indices", "spike_times_ms"):
            field = f"{population}_{suffix}"
            if field == "nonspecific_spike_indices":
                continue  # One nonspecific cell; its result stores times only.
            if actual[field] != previous[field]:
                raise ValueError(f"event train changed: {field}")
    if str(trace["runtime_fingerprint"]) != artifact["runtime_fingerprint"]:
        raise ValueError("trace runtime fingerprint does not match the result")
    if str(trace["condition"]) != "mismatch" or int(trace["schema_version"]) != 1:
        raise ValueError("unexpected trace condition or schema")
    time = np.asarray(trace["time_ms"])
    indices = np.asarray(trace["cell_indices"])
    if time.ndim != 1 or not len(time) or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("trace times must be finite and strictly increasing")
    if indices.ndim != 1 or len(set(indices)) != len(indices):
        raise ValueError("trace cell indices must be unique")
    names = trace["variable_names"].tolist()
    units = dict(zip(names, trace["variable_units"].tolist(), strict=True))
    selected = [name for name in names if name.startswith(("v_", "m_ca_", "h_ca_", "i_ca_"))]
    required = {"v_soma"} | {
        f"{prefix}_{compartment}"
        for prefix in ("v", "m_ca", "h_ca", "i_ca")
        for compartment in ("distal_dendrite", "proximal_dendrite")
    }
    if not required <= set(selected):
        raise ValueError("trace lacks required relay voltage/calcium state")
    for name in selected:
        expected_unit = "mV" if name.startswith("v_") else "pA" if name.startswith("i_") else "dimensionless"
        if units[name] != expected_unit:
            raise ValueError(f"invalid state unit: {name}")
        values = np.asarray(trace[name])
        if values.shape != (len(indices), len(time)) or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid state array: {name}")
    cells = {}
    for row, index in enumerate(indices):
        cells[int(index)] = {
            epoch: {
                name: {"unit": units[name], "min": float(np.min(trace[name][row, mask])),
                       "max": float(np.max(trace[name][row, mask]))}
                for name in selected
            }
            for epoch, mask in (("pre_stimulus", time < 0), ("stimulus", time >= 0))
            if np.any(mask)
        }
    return {
        "event_trains_identical_to_reference": True,
        "monitor_when": str(trace["monitor_when"]),
        "sample_count": len(time),
        "time_range_ms": [float(time[0]), float(time[-1])],
        "cells": cells,
        "causal_calcium_contribution_identified": False,
        "interpretation": "Continuous state ranges are descriptive; no burst label or reproduction promotion is assigned.",
    }


if __name__ == "__main__":
    main()
