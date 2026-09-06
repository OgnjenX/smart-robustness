"""Integrity-check and summarize a declared-input interneuron match trace."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import yaml

EVENT_FIELDS = (
    "relay_spike_indices",
    "relay_spike_times_ms",
    "trn_spike_indices",
    "trn_spike_times_ms",
    "category_spike_indices",
    "category_spike_times_ms",
    "nonspecific_spike_times_ms",
    "layer4_spike_indices",
    "layer4_spike_times_ms",
    "cue_lead_relay_spike_indices",
    "cue_lead_relay_spike_times_ms",
    "cue_lead_trn_spike_indices",
    "cue_lead_trn_spike_times_ms",
    "cue_lead_category_spike_indices",
    "cue_lead_category_spike_times_ms",
    "cue_lead_nonspecific_spike_times_ms",
)
REQUIRED_VARIABLES = (
    "v_soma",
    "v_proximal_dendrite",
    "armed",
    "i_na_soma",
    "i_k_soma",
    "i_axial_inward_soma",
    "i_port_001",
    "i_port_002",
    "i_port_003",
    "port_001_gate",
    "port_002_gate",
    "port_003_gate",
    "i_external_mixed_input",
    "external_mixed_input_input_green",
    "external_mixed_input_input_source_count",
)
EXPECTED_CELLS = (22, 31, 38, 39, 40, 41, 42, 49, 58)
ACTIVE_MATCH_CELLS = (38, 39, 40, 41, 42)


def audit(result, reference, trace, *, trace_sha256):
    match = result["match_result"]
    if not result.get("recorded_match_events_repeat_exactly"):
        raise ValueError("runner did not certify repeated match event trains")
    for field in EVENT_FIELDS:
        if match[field] != reference["match_result"][field]:
            raise ValueError(f"match event train differs: {field}")
    if trace_sha256 != match["interneuron_trace_sha256"]:
        raise ValueError("interneuron trace checksum differs")
    if str(trace["population"]) != "thalamic_interneuron":
        raise ValueError("unexpected trace population")
    if str(trace["condition"]) != "match" or int(trace["schema_version"]) != 1:
        raise ValueError("unexpected trace condition or schema")
    if str(trace["runtime_fingerprint"]) != result["runtime_fingerprint"]:
        raise ValueError("trace runtime differs from result")
    if str(trace["monitor_when"]) != "start":
        raise ValueError("unexpected monitor phase")
    cells = tuple(int(value) for value in trace["cell_indices"])
    if cells != EXPECTED_CELLS:
        raise ValueError("unexpected recorded interneuron cells")
    names = tuple(str(value) for value in trace["variable_names"])
    if set(names) != set(REQUIRED_VARIABLES):
        raise ValueError("interneuron variable coverage differs")
    units = dict(zip(names, (str(value) for value in trace["variable_units"]), strict=True))
    for name in names:
        expected_unit = "mV" if name.startswith("v_") else "pA" if name.startswith("i_") else "dimensionless"
        if units[name] != expected_unit:
            raise ValueError(f"unexpected unit for {name}")
    time = np.asarray(trace["time_ms"], dtype=float)
    if time.ndim != 1 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("invalid trace time axis")
    cue, stimulus = time < 0, time >= 0
    if not cue.any() or not stimulus.any():
        raise ValueError("trace must cover cue and stimulation")
    shape = (len(cells), len(time))
    arrays = {}
    for name in names:
        values = np.asarray(trace[name], dtype=float)
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid state array: {name}")
        arrays[name] = values
    green = arrays["external_mixed_input_input_green"]
    source_count = arrays["external_mixed_input_input_source_count"]
    if np.any(green[:, cue] != 0) or np.any(source_count != 1):
        raise ValueError("cue image or source-count lifecycle differs")
    expected_image = np.asarray([120 if cell in ACTIVE_MATCH_CELLS else 0 for cell in cells])
    if not np.array_equal(green[:, stimulus], np.repeat(expected_image[:, None], stimulus.sum(), axis=1)):
        raise ValueError("stimulus image lifecycle differs")
    summaries = {}
    for row, cell in enumerate(cells):
        summaries[cell] = {
            phase: {
                name: [float(np.min(arrays[name][row, mask])), float(np.max(arrays[name][row, mask]))]
                for name in REQUIRED_VARIABLES
                if name not in {"external_mixed_input_input_green", "external_mixed_input_input_source_count"}
            }
            for phase, mask in (("cue", cue), ("stimulus", stimulus))
        }
    relay_interneuron_ranges = {
        int(cell): [float(low), float(high)]
        for cell, label, low, high in match["relay_driven_current_range_pA_by_index_and_source"]
        if label == "interneuron_gaba"
    }
    emitted = list(zip(match["interneuron_spike_indices"], match["interneuron_spike_times_ms"], strict=True))
    cue_emitted = list(zip(match["cue_lead_interneuron_spike_indices"],
                           match["cue_lead_interneuron_spike_times_ms"], strict=True))
    all_sampled_relay_output_zero = bool(relay_interneuron_ranges) and all(
        low == 0 and high == 0 for low, high in relay_interneuron_ranges.values()
    )
    if emitted:
        pathway_classification = (
            "interneuron_events_present_but_sampled_relay_output_zero"
            if all_sampled_relay_output_zero
            else "interneuron_events_and_sampled_relay_output_present"
        )
    else:
        pathway_classification = "no_interneuron_events_during_stimulus"
    return {
        "schema_version": 1,
        "source_result": result["registration"],
        "trace_sha256": trace_sha256,
        "event_trains_identical_to_reference": True,
        "trace_integrity_checks_pass": True,
        "time_range_ms": [float(time[0]), float(time[-1])],
        "sample_count": len(time),
        "interneuron_cue_events": [[int(i), float(t)] for i, t in cue_emitted],
        "interneuron_stimulus_events": [[int(i), float(t)] for i, t in emitted],
        "sampled_relay_interneuron_current_ranges_pA": relay_interneuron_ranges,
        "pathway_classification": pathway_classification,
        "cells": summaries,
        "interpretation": "State ranges and event timing are descriptive. No causal claim or reproduction promotion is assigned.",
        "baseline_promoted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = yaml.safe_load(Path(args.input).read_text())
    reference = yaml.safe_load(Path(args.reference).read_text())
    path = Path(args.trace)
    digest = sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as trace:
        report = audit(result, reference, trace, trace_sha256=digest)
    with Path(args.output).open("x") as stream:
        yaml.safe_dump(report, stream, sort_keys=False)


if __name__ == "__main__":
    main()
