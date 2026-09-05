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
    Path(args.output).write_text(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
