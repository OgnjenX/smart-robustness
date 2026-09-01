"""Screen scalar TRN-to-relay gain on a registered Figure 7 condition."""

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
from smart_robustness.validation.figure6 import run_figure6_learning
from smart_robustness.validation.figure7 import (
    TopDownCurrentMode,
    expand_figure7_source_expectation_toward_bounds,
    run_figure7_condition,
)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="configs/calibration/figure7_mismatch_gaba_capacity_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    verification_screen_path = profile.get("verification_screen_artifact")
    if verification_screen_path is not None:
        screen = yaml.safe_load(Path(verification_screen_path).read_text())
        if float(screen["selected_gain"]) != float(profile["dimension"]["grid"][0]):
            raise ValueError("verification gain differs from selected screen survivor")
    base_profile = yaml.safe_load(Path(profile["base_profile"]).read_text())
    base = runtime_conventions_for_candidate(base_profile["candidate"])
    detector = profile["detector"]
    conventions = replace(
        base,
        trn_spike_event_coordinate="absolute_physical",
        trn_spike_event_threshold_mV=float(detector["arm_mV"]),
        trn_spike_event_release_mV=float(detector["release_mV"]),
        trn_spike_event_proximal_blend_fraction=None,
        **profile["runtime_overrides"],
    )
    baseline_scales = {
        str(key): float(value)
        for key, value in profile["baseline_trn_to_relay_gaba"]["scales"].items()
    }
    training = run_figure6_learning(
        conventions=conventions,
        projection_weight_scales=baseline_scales,
        brian=brian,
    )
    if training.result.population_spikes["thalamic_relay"] != 20:
        raise ValueError("capacity diagnostic handoff did not reproduce Figure 6")
    learned, applied_factor = expand_figure7_source_expectation_toward_bounds(
        training.learned_weights,
        headroom_fraction=float(profile["learned_state"]["selected_headroom_fraction"]),
        source_index=int(profile["learned_state"]["source_index"]),
    )

    protocol = profile["protocol"]
    condition = MatchCondition(protocol.get("condition", "mismatch"))
    readout = profile["readout"]
    overlap = int(readout.get("overlap_index", 40))
    match_gate = profile.get("match_gate")
    expected_match = (
        tuple(int(index) for index in match_gate["relay_active_indices"])
        if match_gate is not None
        else ()
    )
    outcomes: list[dict[str, Any]] = []
    for gain in profile["dimension"]["grid"]:
        scales = {
            projection_id: baseline * float(gain)
            for projection_id, baseline in baseline_scales.items()
        }
        result = run_figure7_condition(
            condition=condition,
            top_down_current_pA=float(protocol["top_down_current_pA"]),
            learned_weights=learned,
            conventions=conventions,
            duration_ms=float(protocol["duration_ms"]),
            dt_ms=float(protocol["dt_ms"]),
            record_relay_diagnostics=bool(
                protocol.get("record_relay_diagnostics", False)
            ),
            persistent_projection_weight_scales=scales,
            top_down_current_mode=TopDownCurrentMode(
                protocol["top_down_current_mode"]
            ),
            top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
            equilibration_ms=float(protocol["equilibration_ms"]),
            brian=brian,
        )
        active = sorted(set(result.relay_spike_indices))
        gates: dict[str, bool] = {}
        if match_gate is not None:
            source_index = int(profile["learned_state"]["source_index"])
            source_events = tuple(
                time
                for index, time in zip(
                    result.cue_lead_category_spike_indices,
                    result.cue_lead_category_spike_times_ms,
                    strict=True,
                )
                if index == source_index
            )
            relay_counts = {
                index: result.relay_spike_indices.count(index)
                for index in expected_match
            }
            gates = {
                "one_selected_category_event_during_lead": len(source_events) == 1,
                "no_off_source_category_events_during_lead": all(
                    index == source_index
                    for index in result.cue_lead_category_spike_indices
                ),
                "no_relay_events_during_lead": not result.cue_lead_relay_spike_times_ms,
                "current_terminated_on_selected_event": (
                    bool(source_events)
                    and result.top_down_current_termination_time_ms == source_events[0]
                ),
                "relay_active_indices": set(active) == set(expected_match),
                "minimum_relay_events_per_active_index": all(
                    count >= int(match_gate["minimum_relay_events_per_active_index"])
                    for count in relay_counts.values()
                ),
                "trn_events": bool(result.trn_spike_times_ms),
                "figure7_target_duration": result.duration_ms == 100.0,
                "nonspecific_events": (
                    len(result.nonspecific_spike_times_ms)
                    == int(match_gate["nonspecific_events"])
                ),
                "nonspecific_40_hz": (
                    result.nonspecific_rate_hz
                    == float(match_gate["nonspecific_rate_hz"])
                ),
            }
            if protocol.get("record_relay_diagnostics", False):
                sampled = tuple(
                    index for index, _ in result.trn_detector_arm_transitions_by_index
                )
                event_counts = {
                    index: result.trn_spike_indices.count(index) for index in sampled
                }
                upcrossings = dict(
                    result.trn_detector_threshold_upcrossings_by_index
                )
                arms = dict(result.trn_detector_arm_transitions_by_index)
                releases = dict(result.trn_detector_release_transitions_by_index)
                gates["sampled_trn_events_have_fresh_cycles"] = (
                    bool(event_counts)
                    and any(event_counts.values())
                    and all(
                        event_counts[index]
                        == upcrossings[index]
                        == arms[index]
                        == releases[index]
                        for index in event_counts
                    )
                )
        outcomes.append(
            {
                "common_gain": float(gain),
                "scales": scales,
                "relay_events": len(result.relay_spike_times_ms),
                "relay_active_indices": active,
                "relay_event_counts_by_index": {
                    index: result.relay_spike_indices.count(index) for index in active
                },
                "trn_events": len(result.trn_spike_times_ms),
                "nonspecific_events": len(result.nonspecific_spike_times_ms),
                "nonspecific_rate_hz": result.nonspecific_rate_hz,
                "overlap_only": active == [overlap],
                "relay_silent": not active,
                "gates": gates,
                "pass": bool(gates) and all(gates.values()),
                "result": result,
            }
        )
        print(
            f"gain={gain} relay={len(result.relay_spike_times_ms)} "
            f"active={active} trn={len(result.trn_spike_times_ms)} "
            f"ns={len(result.nonspecific_spike_times_ms)}",
            flush=True,
        )

    overlap_only_gains = [
        outcome["common_gain"] for outcome in outcomes if outcome["overlap_only"]
    ]
    match_survivor_gains = [
        outcome["common_gain"] for outcome in outcomes if outcome["pass"]
    ]
    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "complete",
        "profile": args.profile,
        "registration_artifact": profile["registration_artifact"],
        "verification_screen_artifact": verification_screen_path,
        "base_candidate_fingerprint": base_profile["candidate_fingerprint"],
        "runtime_fingerprint": conventions.fingerprint,
        "holdouts_consulted": [f"figure7_{condition.value}"],
        "diagnostic_only": condition is MatchCondition.MISMATCH,
        "handoff_figure6_population_spikes": training.result.population_spikes,
        "applied_common_weight_factor": applied_factor,
        "outcomes": outcomes,
        "overlap_only_gains": overlap_only_gains,
        "match_survivor_gains": match_survivor_gains,
        "selected_gain": min(match_survivor_gains) if match_survivor_gains else None,
        "assessment": {
            "scalar_gain_has_overlap_only_window": bool(overlap_only_gains),
            "advance_to_full_state_match_verification": bool(match_survivor_gains),
            "advance_to_mismatch": bool(
                verification_screen_path and match_survivor_gains
            ),
            "candidate_promoted": False,
            "downstream_holdouts_unlocked": False,
        },
        "next_gate": profile["next_gate"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))


if __name__ == "__main__":
    main()
