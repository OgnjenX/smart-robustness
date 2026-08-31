"""Screen a bounded common layer-6II-to-TRN gain on Figure 7 match only."""

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
        default="configs/calibration/figure7_corticoreticular_gain_match_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    verification_screen_path = profile.get("verification_screen_artifact")
    if verification_screen_path is not None:
        screen = yaml.safe_load(Path(verification_screen_path).read_text())
        selected = screen.get("selected_candidate")
        if selected is None:
            if screen["selected_gain"] != profile["dimension"]["grid"][0]:
                raise ValueError(
                    "verification gain differs from selected screen survivor"
                )
        elif (
            float(selected["common_gain"]) != float(profile["dimension"]["grid"][0])
            or float(selected["headroom_fraction"])
            != float(profile["learned_state"]["selected_headroom_fraction"])
        ):
            raise ValueError(
                "verification pair differs from selected interaction survivor"
            )
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
    baseline_gaba_scales = {
        str(key): float(value)
        for key, value in profile["baseline_trn_to_relay_gaba"]["scales"].items()
    }
    training = run_figure6_learning(
        conventions=conventions,
        projection_weight_scales=baseline_gaba_scales,
        brian=brian,
    )
    if training.result.population_spikes["thalamic_relay"] != 20:
        raise ValueError("corticoreticular screen did not reproduce Figure 6")
    headroom_grid = tuple(
        float(value)
        for value in profile["learned_state"].get(
            "headroom_grid",
            [profile["learned_state"]["selected_headroom_fraction"]],
        )
    )

    protocol = profile["protocol"]
    gate = profile["match_gate"]
    expected = tuple(int(index) for index in gate["relay_active_indices"])
    source_index = int(profile["learned_state"]["source_index"])
    projection_ids = tuple(str(value) for value in profile["dimension"]["projection_ids"])
    outcomes: list[dict[str, Any]] = []
    for headroom_fraction in headroom_grid:
        learned, applied_factor = expand_figure7_source_expectation_toward_bounds(
            training.learned_weights,
            headroom_fraction=headroom_fraction,
            source_index=source_index,
        )
        for gain in profile["dimension"]["grid"]:
            scales = dict(baseline_gaba_scales)
            scales.update({projection_id: float(gain) for projection_id in projection_ids})
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
            counts = {
                index: result.relay_spike_indices.count(index) for index in expected
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
                "relay_active_indices": set(result.relay_spike_indices) == set(expected),
                "minimum_relay_events_per_active_index": all(
                    count >= int(gate["minimum_relay_events_per_active_index"])
                    for count in counts.values()
                ),
                "trn_events": bool(result.trn_spike_times_ms),
                "figure7_target_duration": result.duration_ms == 100.0,
                "nonspecific_events": (
                    len(result.nonspecific_spike_times_ms)
                    == int(gate["nonspecific_events"])
                ),
                "nonspecific_40_hz": (
                    result.nonspecific_rate_hz == float(gate["nonspecific_rate_hz"])
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
                    "headroom_fraction": headroom_fraction,
                    "applied_common_weight_factor": applied_factor,
                    "common_gain": float(gain),
                    "projection_scales": scales,
                    "relay_event_counts_by_index": counts,
                    "sampled_trn_event_counts_by_index": event_counts,
                    "sampled_trn_threshold_upcrossings_by_index": upcrossings,
                    "sampled_trn_arm_transitions_by_index": arms,
                    "sampled_trn_release_transitions_by_index": releases,
                    "gates": gates,
                    "pass": all(gates.values()),
                    "result": result,
                }
            )
            print(
                f"headroom={headroom_fraction} gain={gain} "
                f"relay={len(result.relay_spike_times_ms)} "
                f"trn={len(result.trn_spike_times_ms)} "
                f"ns={len(result.nonspecific_spike_times_ms)} "
                f"pass={outcomes[-1]['pass']}",
                flush=True,
            )

    survivors = [
        {
            "headroom_fraction": outcome["headroom_fraction"],
            "common_gain": outcome["common_gain"],
        }
        for outcome in outcomes
        if outcome["pass"]
    ]
    single_headroom = len(headroom_grid) == 1
    legacy_survivor_gains = (
        [item["common_gain"] for item in survivors] if single_headroom else None
    )
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
        "holdouts_consulted": ["figure7_match"],
        "mismatch_consulted": False,
        "handoff_figure6_population_spikes": training.result.population_spikes,
        "outcomes": outcomes,
        "match_survivors": survivors,
        "selected_candidate": survivors[0] if survivors else None,
        "applied_common_weight_factor": (
            outcomes[0]["applied_common_weight_factor"] if single_headroom else None
        ),
        "match_survivor_gains": legacy_survivor_gains,
        "selected_gain": (
            min(legacy_survivor_gains) if legacy_survivor_gains else None
        ),
        "assessment": {
            "registered_candidate_count": (
                len(headroom_grid) * len(profile["dimension"]["grid"])
            ),
            "completed_candidate_count": len(outcomes),
            "advance_to_full_state_match_verification": bool(survivors)
            and verification_screen_path is None,
            "advance_to_mismatch": bool(survivors)
            and verification_screen_path is not None,
            "mismatch_remains_locked": not (
                bool(survivors) and verification_screen_path is not None
            ),
        },
        "next_gate": profile["next_gate"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))


if __name__ == "__main__":
    main()
