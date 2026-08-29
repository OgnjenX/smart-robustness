"""Audit whether Figure 6 relay rates contain an initialization-latch event."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from smart_robustness.classic_sector import build_first_order_connected_sector
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import run_figure6_learning


def _plain(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    return value


def _pairs(values: tuple[tuple[int, Any], ...]) -> dict[int, Any]:
    return {int(index): value for index, value in values}


def _crossings(values: np.ndarray, level: float, *, upward: bool) -> int:
    if upward:
        return int(np.count_nonzero((values[:-1] <= level) & (values[1:] > level)))
    return int(np.count_nonzero((values[:-1] >= level) & (values[1:] < level)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="configs/calibration/figure6_relay_detector_cycle_audit_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    base_profile = yaml.safe_load(Path(profile["base_profile"]).read_text())
    conventions = runtime_conventions_for_candidate(base_profile["candidate"])
    if conventions.fingerprint != base_profile["runtime_fingerprint"]:
        raise ValueError("base profile runtime fingerprint mismatch")
    indices = tuple(int(value) for value in profile["training"]["active_relay_indices"])

    brian.start_scope()
    brian.defaultclock.dt = float(profile["control"]["dt_ms"]) * brian.ms
    sector = build_first_order_connected_sector(conventions=conventions, brian=brian)
    relay_group = sector.populations["thalamic_relay"].group
    control_spikes = brian.SpikeMonitor(relay_group, name="figure6_control_relay_spikes")
    control_state = brian.StateMonitor(
        relay_group,
        ("spike_detector_voltage", "armed"),
        record=indices,
        name="figure6_control_relay_detector_state",
    )
    sector.network.add(control_spikes, control_state)
    sector.network.run(float(profile["control"]["duration_ms"]) * brian.ms)
    control_voltage = np.asarray(control_state.spike_detector_voltage / brian.mV)
    control_armed = np.asarray(control_state.armed)
    threshold_mV = conventions.spike_event_threshold_mV
    control = {
        "relay_events": int(control_spikes.num_spikes),
        "active_relay_indices": sorted({int(value) for value in control_spikes.i}),
        "detector_voltage_range_mV_by_index": [
            [index, float(np.min(values)), float(np.max(values))]
            for index, values in zip(indices, control_voltage, strict=True)
        ],
        "threshold_upcrossings_by_index": {
            index: _crossings(values, threshold_mV, upward=True)
            for index, values in zip(indices, control_voltage, strict=True)
        },
        "zero_downcrossings_by_index": {
            index: _crossings(values, 0.0, upward=False)
            for index, values in zip(indices, control_voltage, strict=True)
        },
        "arm_transitions_by_index": {
            index: _crossings(values, 0.5, upward=True)
            for index, values in zip(indices, control_armed, strict=True)
        },
        "release_transitions_by_index": {
            index: _crossings(values, 0.5, upward=False)
            for index, values in zip(indices, control_armed, strict=True)
        },
        "final_armed_by_index": {
            index: float(values[-1])
            for index, values in zip(indices, control_armed, strict=True)
        },
    }

    training_run = run_figure6_learning(
        conventions=conventions,
        record_relay_detector_diagnostics=True,
        brian=brian,
    )
    result = training_run.result
    relay_indices = (result.population_spike_indices or {})["thalamic_relay"]
    training_events = {index: relay_indices.count(index) for index in indices}
    training_upcrossings = _pairs(
        result.relay_detector_threshold_upcrossings_by_index
    )
    training_arms = _pairs(result.relay_detector_arm_transitions_by_index)
    training_releases = _pairs(result.relay_detector_release_transitions_by_index)
    control_latched_without_release = bool(
        control["relay_events"] == 0
        and any(value > 0.5 for value in control["final_armed_by_index"].values())
    )
    expected_events = int(profile["training"]["expected_events_per_active_relay"])
    training_cycle_valid = all(
        training_events[index] == expected_events
        and training_upcrossings[index] == expected_events
        and training_arms[index] == expected_events
        and training_releases[index] == expected_events
        for index in indices
    )
    if control_latched_without_release:
        interpretation = "figure6_rate_is_at_risk_of_startup_latch_contamination"
    elif training_cycle_valid:
        interpretation = "figure6_relay_event_train_is_detector-cycle-valid"
    elif any(
        training_events[index] != training_releases[index] for index in indices
    ):
        interpretation = "figure6_event_monitor_or_scheduling_inconsistency"
    else:
        interpretation = "figure6_40hz_claim_requires_reassessment"

    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "diagnostic-complete",
        "profile": args.profile,
        "candidate_fingerprint": base_profile["candidate_fingerprint"],
        "runtime_fingerprint": conventions.fingerprint,
        "control": control,
        "training": {
            "relay_events_by_index": training_events,
            "threshold_upcrossings_by_index": training_upcrossings,
            "arm_transitions_by_index": training_arms,
            "release_transitions_by_index": training_releases,
            "detector_voltage_range_mV_by_index": (
                result.relay_detector_voltage_range_mV_by_index
            ),
            "final_armed_by_index": result.relay_detector_final_armed_by_index,
            "result": result,
        },
        "assessment": {
            "control_latched_without_release": control_latched_without_release,
            "training_cycle_valid": training_cycle_valid,
            "interpretation": interpretation,
        },
        "next_action": (
            "Retain or reopen the Figure 6 source-strength promotion according "
            "to the registered detector-cycle interpretation."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))
    print(
        f"control_relay={control['relay_events']} "
        f"training_events={training_events} interpretation={interpretation}",
        flush=True,
    )


if __name__ == "__main__":
    main()
