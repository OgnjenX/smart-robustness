"""Classify TRN population volleys in a fixed Figure 7 artifact."""

from __future__ import annotations

import argparse
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _condition_analysis(
    result: dict[str, Any],
    *,
    duration_ms: float,
    bin_width_ms: float,
    active_bin_minimum_events: int,
) -> dict[str, Any]:
    trn_times = np.asarray(result["trn_spike_times_ms"], dtype=float)
    nonspecific_times = np.asarray(result["nonspecific_spike_times_ms"], dtype=float)
    edges = np.arange(0.0, duration_ms + bin_width_ms, bin_width_ms)
    counts, _ = np.histogram(trn_times, bins=edges)
    active_indices = np.flatnonzero(counts >= active_bin_minimum_events)
    groups: list[list[int]] = []
    for index in active_indices:
        if not groups or index != groups[-1][-1] + 1:
            groups.append([int(index)])
        else:
            groups[-1].append(int(index))

    volleys: list[dict[str, Any]] = []
    assigned_events: set[float] = set()
    for volley_index, bins in enumerate(groups):
        first_bin = bins[0]
        last_bin = bins[-1]
        start_ms = float(edges[first_bin])
        end_ms = float(edges[last_bin + 1])
        next_start_ms = (
            float(edges[groups[volley_index + 1][0]])
            if volley_index + 1 < len(groups)
            else duration_ms
        )
        volley_events = trn_times[(trn_times >= start_ms) & (trn_times < end_ms)]
        responses = nonspecific_times[
            (nonspecific_times >= float(np.min(volley_events)))
            & (nonspecific_times < next_start_ms)
        ]
        assigned_events.update(float(value) for value in responses)
        volleys.append(
            {
                "index": volley_index,
                "active_bins": bins,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "first_event_ms": float(np.min(volley_events)),
                "last_event_ms": float(np.max(volley_events)),
                "event_count": len(volley_events),
                "event_time_mean_ms": float(np.mean(volley_events)),
                "nonspecific_response_times_ms": [
                    float(value) for value in responses
                ],
            }
        )
    return {
        "relay_event_count": len(result["relay_spike_times_ms"]),
        "trn_event_count": len(trn_times),
        "nonspecific_event_count": len(nonspecific_times),
        "bin_edges_ms": [float(value) for value in edges],
        "trn_event_counts_by_bin": [int(value) for value in counts],
        "active_bin_indices": [int(value) for value in active_indices],
        "volley_count": len(volleys),
        "volleys": volleys,
        "responding_volley_count": sum(
            bool(volley["nonspecific_response_times_ms"]) for volley in volleys
        ),
        "unmatched_nonspecific_event_times_ms": [
            float(value)
            for value in nonspecific_times
            if float(value) not in assigned_events
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    profile = yaml.safe_load(Path(args.profile).read_text())
    source = yaml.safe_load(Path(profile["source_artifact"]).read_text())
    population_size = int(profile["population_size"])
    minimum_fraction = float(profile["active_bin_minimum_population_fraction"])
    minimum_events = math.ceil(minimum_fraction * population_size)
    analyses = {
        condition: _condition_analysis(
            source[f"{condition}_result"],
            duration_ms=float(profile["duration_ms"]),
            bin_width_ms=float(profile["bin_width_ms"]),
            active_bin_minimum_events=minimum_events,
        )
        for condition in ("match", "mismatch")
    }
    mismatch = analyses["mismatch"]
    classification = (
        "downstream_nonspecific_transfer_loses_two_of_seven_trn_volleys"
        if mismatch["volley_count"] == 7
        and mismatch["responding_volley_count"] == 5
        else "upstream_or_unclassified"
    )
    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "complete-read-only-offline-diagnostic",
        "classification": classification,
        "profile": args.profile,
        "registration_artifact": profile["registration_artifact"],
        "source_artifact": profile["source_artifact"],
        "source_runtime_fingerprint": source["runtime_fingerprint"],
        "parameter_changes": "none",
        "network_rerun": False,
        "active_bin_minimum_event_count": minimum_events,
        "conditions": analyses,
        "reproduced": False,
        "next_gate": profile["next_gate"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(artifact, sort_keys=False))
    print(
        "match volleys/responses="
        f"{analyses['match']['volley_count']}/"
        f"{analyses['match']['responding_volley_count']}; mismatch="
        f"{mismatch['volley_count']}/{mismatch['responding_volley_count']}; "
        f"classification={classification}",
        flush=True,
    )


if __name__ == "__main__":
    main()
