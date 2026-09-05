"""Verify a calcium-ablation control comparison before describing its outcome."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path

import numpy as np
import yaml


def compare(control, intervention, control_trace, intervention_trace):
    if control.get("relay_calcium_ablated_at_stimulus", False):
        raise ValueError("control must retain relay calcium")
    if not intervention.get("relay_calcium_ablated_at_stimulus", False):
        raise ValueError("intervention must declare relay calcium ablation")
    for population in ("category", "relay", "trn", "nonspecific"):
        field = f"cue_lead_{population}_spike_times_ms"
        if control[field] != intervention[field]:
            raise ValueError(f"cue history changed: {field}")
        if population != "nonspecific":
            field = f"cue_lead_{population}_spike_indices"
            if control[field] != intervention[field]:
                raise ValueError(f"cue history changed: {field}")
    for field in ("time_ms", "cell_indices", "variable_names", "variable_units",
                  "monitor_when", "monitor_order", "runtime_fingerprint", "condition"):
        if not np.array_equal(control_trace[field], intervention_trace[field]):
            raise ValueError(f"trace metadata changed: {field}")
    time = np.asarray(control_trace["time_ms"])
    before, during = time < 0, time >= 0
    if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0) or not before.any() or not during.any():
        raise ValueError("trace must cover finite, ordered cue and sensory samples")
    shape = (len(control_trace["cell_indices"]), len(time))
    names = control_trace["variable_names"].tolist()
    for name in names:
        baseline, changed = np.asarray(control_trace[name]), np.asarray(intervention_trace[name])
        if baseline.shape != shape or changed.shape != shape or not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(changed)):
            raise ValueError(f"invalid state array: {name}")
        if not np.array_equal(baseline[:, before], changed[:, before]):
            raise ValueError(f"pre-stimulus state changed: {name}")
    scope = intervention.get("relay_calcium_ablation_scope", "dendrites_only")
    if scope not in ("dendrites_only", "all_relay_compartments"):
        raise ValueError("unrecognized relay calcium ablation scope")
    currents = ["i_ca_distal_dendrite", "i_ca_proximal_dendrite"]
    if scope == "all_relay_compartments":
        currents.append("i_ca_soma")
    for name in currents:
        if name not in names:
            raise ValueError(f"missing intervention observable: {name}")
        if not np.any(control_trace[name][:, during] != 0):
            raise ValueError("control calcium current is already absent")
        if np.any(intervention_trace[name][:, during] != 0):
            raise ValueError("relay calcium current persists after ablation")
    cells = {}
    for index in (22, 31, 40, 49, 58):
        cells[index] = {
            label: [float(t) for i, t in zip(result["relay_spike_indices"], result["relay_spike_times_ms"], strict=True) if i == index]
            for label, result in (("control_times_ms", control), ("ablation_times_ms", intervention))
        }
    return {
        "pre_stimulus_state_and_cue_events_identical": True,
        "sampled_relay_dendritic_calcium_currents_zero_during_stimulus": True,
        "ablation_scope": (
            scope if scope == "all_relay_compartments"
            else "distal_and_proximal_dendrites_only_somatic_calcium_retained"
        ),
        "verified_zero_current_variables": currents,
        "cells": cells,
        "nonoverlap_cells_still_firing": [i for i in (22, 31, 49, 58) if cells[i]["ablation_times_ms"]],
        "reproduction_eligible": False,
        "scope": "Calibrated network intervention; recurrent effects are included, and prior calcium history is retained.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--intervention", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    artifacts = [yaml.safe_load(Path(p).read_bytes()) for p in (args.control, args.intervention)]
    for field in ("runtime_fingerprint", "profile", "selected_headroom_fraction", "applied_common_weight_factor"):
        if artifacts[0][field] != artifacts[1][field]:
            raise ValueError(f"control configuration changed: {field}")
    with ExitStack() as stack:
        traces = []
        for artifact in artifacts:
            result = artifact["mismatch_result"]
            path = Path(result["relay_trace_path"])
            if sha256(path.read_bytes()).hexdigest() != result["relay_trace_sha256"]:
                raise ValueError("trace checksum differs from result")
            trace = stack.enter_context(np.load(path, allow_pickle=False))
            if str(trace["runtime_fingerprint"]) != artifact["runtime_fingerprint"]:
                raise ValueError("trace runtime differs from result")
            if str(trace["condition"]) != "mismatch" or int(trace["schema_version"]) != 1:
                raise ValueError("unexpected trace condition or schema")
            traces.append(trace)
        report = compare(*(a["mismatch_result"] for a in artifacts), *traces)
    report.update(
        schema_version=1, control_artifact=args.control, intervention_artifact=args.intervention,
        trace_sha256=[a["mismatch_result"]["relay_trace_sha256"] for a in artifacts],
    )
    with Path(args.output).open("x") as stream:
        yaml.safe_dump(report, stream, sort_keys=False)


if __name__ == "__main__":
    main()
