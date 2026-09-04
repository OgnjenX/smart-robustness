"""Screen common TRN-to-nonspecific GABA transfer on Figure 6 then match."""

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
from smart_robustness.validation.figure6 import (
    assess_figure6_cortical_recruitment,
    assess_figure6_top_down_timing,
    run_figure6_learning,
)
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
    outcomes: list[dict[str, Any]],
    complete: bool,
) -> None:
    survivors = [item["common_scale"] for item in outcomes if item["pass"]]
    selected = min(survivors) if survivors else None
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
        "figure6_baseline_artifact": profile["figure6_baseline_artifact"],
        "top5_pair_artifact": profile["top5_pair_artifact"],
        "holdouts_consulted": ["figure6", "figure7_match"],
        "mismatch_consulted": False,
        "outcomes": outcomes,
        "match_survivor_common_scales": survivors,
        "selected_common_scale": selected,
        "assessment": {
            "registered_candidate_count": len(
                profile["nonspecific_gaba_transfer"]["common_scale_grid"]
            ),
            "completed_candidate_count": len(outcomes),
            "advance_to_independent_match_verification": bool(
                complete and survivors
            ),
            "advance_to_mismatch": False,
            "mismatch_remains_locked": True,
        },
        "next_gate": profile["next_gate"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default=(
            "configs/calibration/"
            "figure7_top5_nonspecific_gaba_transfer_match_v1.yaml"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
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
    relay_scales = {
        str(key): float(value)
        for key, value in profile["trn_to_relay_gaba"]["scales"].items()
    }
    nonspecific_projection_ids = tuple(
        str(value)
        for value in profile["nonspecific_gaba_transfer"]["projection_ids"]
    )
    protocol = profile["protocol"]
    figure6_gate = profile["figure6_gate"]
    match_gate = profile["match_gate"]
    expected = tuple(int(index) for index in match_gate["relay_active_indices"])
    source_index = int(profile["learned_state"]["source_index"])
    output = Path(args.output)
    outcomes: list[dict[str, Any]] = []

    for raw_scale in profile["nonspecific_gaba_transfer"]["common_scale_grid"]:
        scale = float(raw_scale)
        scales = relay_scales | {
            projection_id: scale for projection_id in nonspecific_projection_ids
        }
        training = run_figure6_learning(
            conventions=conventions,
            projection_weight_scales=scales,
            brian=brian,
        )
        training_result = training.result
        training_indices = training_result.population_spike_indices[
            "thalamic_relay"
        ]
        training_counts = {
            index: training_indices.count(index)
            for index in figure6_gate["relay_active_indices"]
        }
        recruitment = assess_figure6_cortical_recruitment(training_result)
        timing = assess_figure6_top_down_timing(training_result)
        training_gates = {
            "relay_active_indices": set(training_indices)
            == set(figure6_gate["relay_active_indices"]),
            "relay_events_per_active_index": set(training_counts.values())
            == {int(figure6_gate["relay_events_per_active_index"])},
            "relay_events": len(training_indices)
            == int(figure6_gate["relay_events"]),
            "cortical_chain_complete": recruitment.feedforward_chain_complete,
            "causal_learning_pair": timing.causal_pair_in_learning_window,
            "top_down_horizontal_contrast": (
                training_result.top_down_combined.horizontal_orientation_contrast
                > 0
            ),
        }
        outcome: dict[str, Any] = {
            "common_scale": scale,
            "projection_weight_scales": scales,
            "figure6_result": training_result,
            "figure6_relay_event_counts_by_index": training_counts,
            "figure6_cortical_recruitment": recruitment,
            "figure6_top_down_timing": timing,
            "figure6_gates": training_gates,
            "figure6_pass": all(training_gates.values()),
            "match_result": None,
            "match_gates": {},
            "match_pass": False,
            "pass": False,
        }
        if outcome["figure6_pass"]:
            learned, applied_factor = expand_figure7_source_expectation_toward_bounds(
                training.learned_weights,
                headroom_fraction=float(
                    profile["learned_state"]["headroom_fraction"]
                ),
                source_index=source_index,
            )
            match = run_figure7_condition(
                condition=MatchCondition.MATCH,
                top_down_current_pA=float(protocol["top_down_current_pA"]),
                learned_weights=learned,
                conventions=conventions,
                duration_ms=float(protocol["duration_ms"]),
                dt_ms=float(protocol["dt_ms"]),
                record_relay_diagnostics=bool(
                    protocol["record_relay_diagnostics"]
                ),
                persistent_projection_weight_scales=scales,
                comparator_top_k_targets=int(profile["comparator"]["target_count"]),
                comparator_source_index=source_index,
                top_down_current_mode=TopDownCurrentMode(
                    protocol["top_down_current_mode"]
                ),
                top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
                equilibration_ms=float(protocol["equilibration_ms"]),
                brian=brian,
            )
            relay_counts = {
                index: match.relay_spike_indices.count(index) for index in expected
            }
            sampled = tuple(
                index for index, _ in match.trn_detector_arm_transitions_by_index
            )
            event_counts = {
                index: match.trn_spike_indices.count(index) for index in sampled
            }
            upcrossings = dict(match.trn_detector_threshold_upcrossings_by_index)
            arms = dict(match.trn_detector_arm_transitions_by_index)
            releases = dict(match.trn_detector_release_transitions_by_index)
            fresh_cycles = bool(event_counts) and any(event_counts.values()) and all(
                event_counts[index]
                == upcrossings[index]
                == arms[index]
                == releases[index]
                for index in event_counts
            )
            source_events = tuple(
                time
                for index, time in zip(
                    match.cue_lead_category_spike_indices,
                    match.cue_lead_category_spike_times_ms,
                    strict=True,
                )
                if index == source_index
            )
            match_gates = {
                "one_selected_category_event_during_lead": len(source_events) == 1,
                "no_off_source_category_events_during_lead": all(
                    index == source_index
                    for index in match.cue_lead_category_spike_indices
                ),
                "no_relay_events_during_lead": not match.cue_lead_relay_spike_times_ms,
                "relay_active_indices": set(match.relay_spike_indices)
                == set(expected),
                "minimum_relay_events_per_active_index": all(
                    count
                    >= int(match_gate["minimum_relay_events_per_active_index"])
                    for count in relay_counts.values()
                ),
                "trn_events": bool(match.trn_spike_times_ms),
                "nonspecific_events": len(match.nonspecific_spike_times_ms)
                == int(match_gate["nonspecific_events"]),
                "nonspecific_40_hz": match.nonspecific_rate_hz
                == float(match_gate["nonspecific_rate_hz"]),
                "figure7_target_duration": match.duration_ms == 100.0,
                "sampled_trn_events_have_fresh_cycles": fresh_cycles,
            }
            outcome.update(
                {
                    "applied_common_weight_factor": applied_factor,
                    "match_result": match,
                    "match_relay_event_counts_by_index": relay_counts,
                    "sampled_match_trn_event_counts_by_index": event_counts,
                    "sampled_match_trn_threshold_upcrossings_by_index": upcrossings,
                    "sampled_match_trn_arm_transitions_by_index": arms,
                    "sampled_match_trn_release_transitions_by_index": releases,
                    "match_gates": match_gates,
                    "match_pass": all(match_gates.values()),
                    "pass": all(match_gates.values()),
                }
            )
        outcomes.append(outcome)
        _write(
            output,
            profile_path=args.profile,
            profile=profile,
            base_profile=base_profile,
            runtime_fingerprint=conventions.fingerprint,
            outcomes=outcomes,
            complete=False,
        )
        match_result = outcome["match_result"]
        print(
            f"scale={scale:g} figure6_relay={len(training_indices)} "
            f"figure6_pass={outcome['figure6_pass']} "
            f"match_relay={len(match_result.relay_spike_times_ms) if match_result else '-'} "
            f"match_trn={len(match_result.trn_spike_times_ms) if match_result else '-'} "
            f"match_nonspecific={len(match_result.nonspecific_spike_times_ms) if match_result else '-'} "
            f"pass={outcome['pass']}",
            flush=True,
        )

    _write(
        output,
        profile_path=args.profile,
        profile=profile,
        base_profile=base_profile,
        runtime_fingerprint=conventions.fingerprint,
        outcomes=outcomes,
        complete=True,
    )


if __name__ == "__main__":
    main()
