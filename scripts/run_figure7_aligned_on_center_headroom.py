"""Screen bounded learned on-center headroom on the aligned Figure 7 match."""

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
    outcomes: list[dict[str, Any]],
    complete: bool,
) -> None:
    survivors = [item["headroom_fraction"] for item in outcomes if item["pass"]]
    selected = min(survivors) if survivors else None
    verification = profile.get("verification_screen_artifact") is not None
    artifact = {
        "schema_version": 1,
        "id": output.stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "complete" if complete else "running",
        "profile": profile_path,
        "registration_artifact": profile["registration_artifact"],
        "base_candidate_fingerprint": base_profile["candidate_fingerprint"],
        "runtime_fingerprint": runtime_fingerprint,
        "figure6_artifact": profile["figure6_artifact"],
        "aligned_control_artifact": profile["aligned_control_artifact"],
        "holdouts_consulted": ["figure7_match"],
        "mismatch_consulted": False,
        "handoff_figure6_population_spikes": training_spikes,
        "outcomes": outcomes,
        "stage_1_survivor_headroom_fractions": survivors,
        "selected_headroom_fraction": selected,
        "assessment": {
            "registered_candidate_count": len(profile["dimension"]["grid"]),
            "completed_candidate_count": len(outcomes),
            "advance_to_diagnostic_match": bool(
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
        default="configs/calibration/figure7_aligned_on_center_headroom_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    base_profile = yaml.safe_load(Path(profile["base_profile"]).read_text())
    control = yaml.safe_load(Path(profile["aligned_control_artifact"]).read_text())
    if control["survivor_currents_pA"]:
        raise ValueError("aligned control unexpectedly passed exact match")
    verification_path = profile.get("verification_screen_artifact")
    if verification_path is not None:
        screen = yaml.safe_load(Path(verification_path).read_text())
        if screen["selected_headroom_fraction"] != profile["dimension"]["grid"][0]:
            raise ValueError("verification fraction differs from selected screen survivor")
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
    if training.result.population_spikes["thalamic_relay"] != 20:
        raise ValueError("fresh handoff did not reproduce the Figure 6 relay train")

    protocol = profile["protocol"]
    current_mode = TopDownCurrentMode(protocol["top_down_current_mode"])
    gate = profile["stage_1_gate"]
    expected = tuple(int(index) for index in gate["relay_active_indices"])
    output = Path(args.output)
    outcomes: list[dict[str, Any]] = []
    for fraction in profile["dimension"]["grid"]:
        learned, applied_factor = expand_figure7_source_expectation_toward_bounds(
            training.learned_weights,
            headroom_fraction=float(fraction),
            source_index=int(profile["dimension"]["source_index"]),
        )
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
                current_mode
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
            if index == int(profile["dimension"]["source_index"])
        )
        relay_counts = {
            index: result.relay_spike_indices.count(index) for index in expected
        }
        gates = {
            "one_selected_category_event_during_lead": len(source_events) == 1,
            "no_off_source_category_events_during_lead": all(
                index == int(profile["dimension"]["source_index"])
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
            "nonspecific_events": (
                len(result.nonspecific_spike_times_ms)
                == int(gate["nonspecific_events"])
            ),
            "nonspecific_40_hz": (
                result.nonspecific_rate_hz == float(gate["nonspecific_rate_hz"])
            ),
        }
        if current_mode is TopDownCurrentMode.UNTIL_CUED_CELL_FIRST_EVENT:
            gates["current_terminated_on_selected_event"] = (
                bool(source_events)
                and result.top_down_current_termination_time_ms == source_events[0]
            )
        else:
            gates["sustained_current_protocol"] = (
                result.top_down_current_termination_time_ms is None
            )
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
                "headroom_fraction": float(fraction),
                "applied_common_weight_factor": applied_factor,
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
            training_spikes=training.result.population_spikes,
            outcomes=outcomes,
            complete=False,
        )
        print(
            f"fraction={float(fraction):g} factor={applied_factor:.6g} "
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
        training_spikes=training.result.population_spikes,
        outcomes=outcomes,
        complete=True,
    )


if __name__ == "__main__":
    main()
