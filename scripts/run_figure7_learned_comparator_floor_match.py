"""Screen a preregistered learned-support comparator on Figure 7 match only."""

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


def _write(
    output: Path,
    *,
    profile_path: str,
    profile: dict[str, Any],
    base_profile: dict[str, Any],
    runtime_fingerprint: str,
    training_spikes: dict[str, int],
    applied_factor: float,
    outcomes: list[dict[str, Any]],
    complete: bool,
) -> None:
    transform = profile["dimension"].get("kind", "linear_floor")
    candidate_key = {
        "linear_floor": "comparator_floor",
        "half_max_binary": "support_threshold",
        "top_k_binary": "target_count",
    }[transform]
    survivors = [item[candidate_key] for item in outcomes if item["pass"]]
    selected = max(survivors) if survivors else None
    verification = profile.get("verification_screen_artifact") is not None
    artifact = {
        "schema_version": 1,
        "id": output.stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "complete" if complete else "running",
        "classification": "calibrated-reconstruction-not-recovered-source",
        "profile": profile_path,
        "registration_artifact": profile["registration_artifact"],
        "base_candidate_fingerprint": base_profile["candidate_fingerprint"],
        "runtime_fingerprint": runtime_fingerprint,
        "figure6_artifact": profile["figure6_artifact"],
        "uniform_input_control_artifact": profile["uniform_input_control_artifact"],
        "verification_screen_artifact": profile.get("verification_screen_artifact"),
        "holdouts_consulted": ["figure7_match"],
        "mismatch_consulted": False,
        "handoff_figure6_population_spikes": training_spikes,
        "applied_common_weight_factor": applied_factor,
        "outcomes": outcomes,
        (
            "match_survivor_floors"
            if transform == "linear_floor"
            else "match_survivor_thresholds"
            if transform == "half_max_binary"
            else "match_survivor_target_counts"
        ): survivors,
        (
            "selected_comparator_floor"
            if transform == "linear_floor"
            else "selected_support_threshold"
            if transform == "half_max_binary"
            else "selected_target_count"
        ): selected,
        "assessment": {
            "registered_candidate_count": len(profile["dimension"]["grid"]),
            "completed_candidate_count": len(outcomes),
            "advance_to_independent_match_verification": bool(
                complete and survivors and not verification
            ),
            "advance_to_mismatch": bool(complete and survivors and verification),
            "mismatch_remains_locked": not bool(
                complete and survivors and verification
            ),
        },
        "next_gate": profile["next_gate"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="configs/calibration/figure7_learned_comparator_floor_match_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    base_profile = yaml.safe_load(Path(profile["base_profile"]).read_text())
    verification_path = profile.get("verification_screen_artifact")
    if verification_path is not None:
        screen = yaml.safe_load(Path(verification_path).read_text())
        if screen["selected_target_count"] != profile["dimension"]["grid"][0]:
            raise ValueError("verification target count differs from screen survivor")
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
    scales = {
        str(key): float(value)
        for key, value in profile["trn_to_relay_gaba"]["scales"].items()
    }
    training = run_figure6_learning(
        conventions=conventions,
        projection_weight_scales=scales,
        brian=brian,
    )
    training_spikes = training.result.population_spikes
    if training_spikes["thalamic_relay"] != 20:
        raise ValueError("fresh handoff did not reproduce the Figure 6 relay train")
    learned, applied_factor = expand_figure7_source_expectation_toward_bounds(
        training.learned_weights,
        headroom_fraction=float(profile["learned_state"]["headroom_fraction"]),
        source_index=int(profile["learned_state"]["source_index"]),
    )

    protocol = profile["protocol"]
    gate = profile["stage_1_gate"]
    source_index = int(profile["learned_state"]["source_index"])
    expected = tuple(int(index) for index in gate["relay_active_indices"])
    output = Path(args.output)
    outcomes: list[dict[str, Any]] = []
    transform = profile["dimension"].get("kind", "linear_floor")
    if transform not in {"linear_floor", "half_max_binary", "top_k_binary"}:
        raise ValueError(f"unknown comparator transform: {transform}")
    for candidate in profile["dimension"]["grid"]:
        if transform == "half_max_binary" and float(candidate) != 0.5:
            raise ValueError("half-max comparator threshold must be exactly 0.5")
        result = run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=float(protocol["top_down_current_pA"]),
            learned_weights=learned,
            conventions=conventions,
            duration_ms=float(protocol["duration_ms"]),
            dt_ms=float(protocol["dt_ms"]),
            record_relay_diagnostics=bool(
                protocol.get("record_relay_diagnostics", False)
            ),
            persistent_projection_weight_scales=scales,
            comparator_relay_floor=(
                float(candidate) if transform == "linear_floor" else None
            ),
            comparator_half_max_gate=transform == "half_max_binary",
            comparator_top_k_targets=(
                int(candidate) if transform == "top_k_binary" else None
            ),
            comparator_source_index=source_index,
            top_down_current_mode=TopDownCurrentMode(
                protocol["top_down_current_mode"]
            ),
            top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
            equilibration_ms=float(protocol["equilibration_ms"]),
            brian=brian,
        )
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
            index: result.relay_spike_indices.count(index) for index in expected
        }
        gates = {
            "one_selected_category_event_during_lead": len(source_events) == 1,
            "no_off_source_category_events_during_lead": all(
                index == source_index
                for index in result.cue_lead_category_spike_indices
            ),
            "no_relay_events_during_lead": not result.cue_lead_relay_spike_times_ms,
            "relay_active_indices": set(result.relay_spike_indices) == set(expected),
            "minimum_relay_events_per_active_index": all(
                count >= int(gate["minimum_relay_events_per_active_index"])
                for count in relay_counts.values()
            ),
            "trn_events": bool(result.trn_spike_times_ms),
            "figure7_target_duration": result.duration_ms == 100.0,
            "nonspecific_events": len(result.nonspecific_spike_times_ms)
            == int(gate["nonspecific_events"]),
            "nonspecific_40_hz": result.nonspecific_rate_hz
            == float(gate["nonspecific_rate_hz"]),
            "current_terminated_on_selected_event": (
                bool(source_events)
                and result.top_down_current_termination_time_ms == source_events[0]
            ),
        }
        event_counts: dict[int, int] = {}
        upcrossings: dict[int, int] = {}
        arms: dict[int, int] = {}
        releases: dict[int, int] = {}
        if protocol.get("record_relay_diagnostics", False):
            sampled = tuple(
                index for index, _ in result.trn_detector_arm_transitions_by_index
            )
            event_counts = {
                index: result.trn_spike_indices.count(index) for index in sampled
            }
            upcrossings = dict(result.trn_detector_threshold_upcrossings_by_index)
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
                (
                    "comparator_floor"
                    if transform == "linear_floor"
                    else "support_threshold"
                    if transform == "half_max_binary"
                    else "target_count"
                ): int(candidate)
                if transform == "top_k_binary"
                else float(candidate),
                "result": result,
                "relay_event_counts_by_index": relay_counts,
                "sampled_trn_event_counts_by_index": event_counts,
                "sampled_trn_threshold_upcrossings_by_index": upcrossings,
                "sampled_trn_arm_transitions_by_index": arms,
                "sampled_trn_release_transitions_by_index": releases,
                "gates": gates,
                "pass": all(gates.values()),
            }
        )
        _write(
            output,
            profile_path=args.profile,
            profile=profile,
            base_profile=base_profile,
            runtime_fingerprint=conventions.fingerprint,
            training_spikes=training_spikes,
            applied_factor=applied_factor,
            outcomes=outcomes,
            complete=False,
        )
        print(
            f"{transform}={float(candidate):g} "
            f"relay={len(result.relay_spike_times_ms)} "
            f"trn={len(result.trn_spike_times_ms)} "
            f"nonspecific={len(result.nonspecific_spike_times_ms)} "
            f"pass={all(gates.values())}",
            flush=True,
        )
    _write(
        output,
        profile_path=args.profile,
        profile=profile,
        base_profile=base_profile,
        runtime_fingerprint=conventions.fingerprint,
        training_spikes=training_spikes,
        applied_factor=applied_factor,
        outcomes=outcomes,
        complete=True,
    )


if __name__ == "__main__":
    main()
