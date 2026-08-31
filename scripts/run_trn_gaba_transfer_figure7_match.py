"""Run the Figure 7 match holdout for the calibrated Figure 6 survivor."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure7 import run_figure7_condition


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="configs/calibration/trn_gaba_transfer_figure7_match_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    base_profile = yaml.safe_load(Path(profile["base_profile"]).read_text())
    figure6 = yaml.safe_load(Path(profile["figure6_artifact"]).read_text())
    transfer = profile["trn_to_relay_gaba"]
    if "scales" in transfer:
        projection_scales = {
            str(projection_id): float(scale)
            for projection_id, scale in transfer["scales"].items()
        }
        survivor_label = str(transfer["figure6_survivor_label"])
        if figure6["stage_2_survivor_labels"] != [survivor_label]:
            raise ValueError("match profile is not the sole Figure 6 survivor")
        survivor = next(
            item
            for item in figure6["stage_2_outcomes"]
            if item["label"] == survivor_label
        )
        if survivor["scales"] != projection_scales:
            raise ValueError("match projection scales differ from Figure 6")
    else:
        scale = float(transfer["scale"])
        if figure6["stage_2_survivor_scales"] != [scale]:
            raise ValueError("match scale is not the sole Figure 6 survivor")
        projection_scales = {
            str(projection_id): scale
            for projection_id in transfer["projection_ids"]
        }
    base = runtime_conventions_for_candidate(base_profile["candidate"])
    detector = profile["detector"]
    runtime_overrides = {
        str(key): value for key, value in profile.get("runtime_overrides", {}).items()
    }
    conventions = replace(
        base,
        trn_spike_event_coordinate="absolute_physical",
        trn_spike_event_threshold_mV=float(detector["arm_mV"]),
        trn_spike_event_release_mV=float(detector["release_mV"]),
        trn_spike_event_proximal_blend_fraction=None,
        **runtime_overrides,
    )
    protocol = profile["protocol"]
    result = run_figure7_condition(
        condition=MatchCondition.MATCH,
        top_down_current_pA=float(protocol["top_down_current_pA"]),
        pretrain_with_figure6_episode=True,
        conventions=conventions,
        duration_ms=float(protocol["duration_ms"]),
        dt_ms=float(protocol["dt_ms"]),
        record_relay_diagnostics=bool(protocol["record_relay_diagnostics"]),
        persistent_projection_weight_scales=projection_scales,
        top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
        equilibration_ms=float(protocol["equilibration_ms"]),
        brian=brian,
    )
    expected = tuple(int(value) for value in profile["match_gate"]["relay_active_indices"])
    relay_counts = {index: result.relay_spike_indices.count(index) for index in expected}
    sampled = set(_pairs(result.trn_detector_arm_transitions_by_index))
    event_counts = {
        index: result.trn_spike_indices.count(index) for index in sorted(sampled)
    }
    upcrossings = _pairs(result.trn_detector_threshold_upcrossings_by_index)
    arms = _pairs(result.trn_detector_arm_transitions_by_index)
    releases = _pairs(result.trn_detector_release_transitions_by_index)
    sampled_events_have_fresh_cycles = all(
        event_counts[index] == upcrossings[index] == arms[index] == releases[index]
        for index in sampled
    ) and any(event_counts.values())
    gates = {
        "relay_active_indices": set(result.relay_spike_indices) == set(expected),
        "minimum_relay_events_per_active_index": all(
            count
            >= int(profile["match_gate"]["minimum_relay_events_per_active_index"])
            for count in relay_counts.values()
        ),
        "trn_events": len(result.trn_spike_times_ms) > 0,
        "sampled_trn_events_have_fresh_cycles": sampled_events_have_fresh_cycles,
    }
    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "match-pass" if all(gates.values()) else "match-fail",
        "profile": args.profile,
        "base_candidate_fingerprint": base_profile["candidate_fingerprint"],
        "runtime_fingerprint": conventions.fingerprint,
        "figure6_artifact": profile["figure6_artifact"],
        "holdouts_consulted": ["figure7_match"],
        "projection_weight_scales": projection_scales,
        "result": result,
        "relay_event_counts_by_index": relay_counts,
        "sampled_trn_event_counts_by_index": event_counts,
        "sampled_trn_threshold_upcrossings_by_index": upcrossings,
        "sampled_trn_arm_transitions_by_index": arms,
        "sampled_trn_release_transitions_by_index": releases,
        "gates": gates,
        "assessment": {
            "same_network_match_pass": all(gates.values()),
            "advance_to_mismatch": all(gates.values()),
        },
        "next_gate": profile["next_gate"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))
    print(
        f"relay={len(result.relay_spike_times_ms)} "
        f"active={sorted(set(result.relay_spike_indices))} "
        f"trn={len(result.trn_spike_times_ms)} "
        f"nonspecific={len(result.nonspecific_spike_times_ms)} "
        f"pass={all(gates.values())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
