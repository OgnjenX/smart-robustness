"""Run the sole locked mismatch for the verified aligned on-center match."""

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
    Figure7ConditionResult,
    TopDownCurrentMode,
    assess_figure7_reproduction,
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


def _scoring_result(raw: dict[str, Any]) -> Figure7ConditionResult:
    return Figure7ConditionResult(
        condition=MatchCondition(raw["condition"]),
        duration_ms=float(raw["duration_ms"]),
        nonspecific_spike_times_ms=tuple(raw["nonspecific_spike_times_ms"]),
        relay_spike_indices=tuple(raw["relay_spike_indices"]),
        relay_spike_times_ms=tuple(raw["relay_spike_times_ms"]),
        trn_spike_indices=tuple(raw["trn_spike_indices"]),
        trn_spike_times_ms=tuple(raw["trn_spike_times_ms"]),
        cue_lead_category_spike_indices=tuple(
            raw["cue_lead_category_spike_indices"]
        ),
        cue_lead_category_spike_times_ms=tuple(
            raw["cue_lead_category_spike_times_ms"]
        ),
        cue_lead_nonspecific_spike_times_ms=tuple(
            raw["cue_lead_nonspecific_spike_times_ms"]
        ),
        cue_lead_trn_spike_indices=tuple(raw["cue_lead_trn_spike_indices"]),
        cue_lead_trn_spike_times_ms=tuple(raw["cue_lead_trn_spike_times_ms"]),
        cue_lead_relay_spike_indices=tuple(raw["cue_lead_relay_spike_indices"]),
        cue_lead_relay_spike_times_ms=tuple(raw["cue_lead_relay_spike_times_ms"]),
        top_down_current_mode=raw["top_down_current_mode"],
        top_down_current_termination_time_ms=raw[
            "top_down_current_termination_time_ms"
        ],
        top_down_cue_lead_ms=float(raw["top_down_cue_lead_ms"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="configs/calibration/figure7_aligned_on_center_mismatch_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--diagnostic-registration")
    parser.add_argument("--pre-event-offsets-ms", type=float, nargs="*", default=[])
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    base_profile = yaml.safe_load(Path(profile["base_profile"]).read_text())
    match_artifact = yaml.safe_load(
        Path(profile["match_verification_artifact"]).read_text()
    )
    if not match_artifact["assessment"].get("advance_to_mismatch", False):
        raise ValueError("verified match does not authorize mismatch")
    selected_fraction = float(profile["learned_state"]["selected_headroom_fraction"])
    match_outcome = match_artifact["outcomes"][0]
    if not match_outcome["pass"]:
        raise ValueError("match artifact does not contain the selected sole survivor")
    if "headroom_fraction" in match_outcome and not np.isclose(
        float(match_outcome["headroom_fraction"]), selected_fraction
    ):
        raise ValueError("match artifact headroom differs from registered mismatch")
    corticoreticular = profile.get("corticoreticular_common_gain")
    if corticoreticular is not None and not np.isclose(
        float(match_outcome["common_gain"]), float(corticoreticular["value"])
    ):
        raise ValueError("match artifact corticoreticular gain differs from registration")
    match = _scoring_result(match_outcome["result"])

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
    training_scales = dict(scales)
    if corticoreticular is not None:
        scales.update(
            {
                str(projection_id): float(corticoreticular["value"])
                for projection_id in corticoreticular["projection_ids"]
            }
        )
    training = run_figure6_learning(
        conventions=conventions,
        projection_weight_scales=training_scales,
        brian=brian,
    )
    if training.result.population_spikes["thalamic_relay"] != 20:
        raise ValueError("fresh mismatch handoff did not reproduce Figure 6")
    learned, applied_factor = expand_figure7_source_expectation_toward_bounds(
        training.learned_weights,
        headroom_fraction=selected_fraction,
        source_index=int(profile["learned_state"]["source_index"]),
    )
    if not np.isclose(
        applied_factor,
        float(profile["learned_state"]["expected_common_weight_factor"]),
    ):
        raise ValueError("fresh mismatch headroom factor differs from registration")

    protocol = profile["protocol"]
    mismatch = run_figure7_condition(
        condition=MatchCondition.MISMATCH,
        top_down_current_pA=float(protocol["top_down_current_pA"]),
        learned_weights=learned,
        conventions=conventions,
        duration_ms=float(protocol["duration_ms"]),
        dt_ms=float(protocol["dt_ms"]),
        record_relay_diagnostics=True,
        relay_pre_event_offsets_ms=tuple(args.pre_event_offsets_ms),
        persistent_projection_weight_scales=scales,
        top_down_current_mode=TopDownCurrentMode(protocol["top_down_current_mode"]),
        top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
        equilibration_ms=float(protocol["equilibration_ms"]),
        brian=brian,
    )
    assessment = assess_figure7_reproduction(match, mismatch)
    sampled = tuple(index for index, _ in mismatch.trn_detector_arm_transitions_by_index)
    event_counts = {index: mismatch.trn_spike_indices.count(index) for index in sampled}
    upcrossings = dict(mismatch.trn_detector_threshold_upcrossings_by_index)
    arms = dict(mismatch.trn_detector_arm_transitions_by_index)
    releases = dict(mismatch.trn_detector_release_transitions_by_index)
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
            mismatch.cue_lead_category_spike_indices,
            mismatch.cue_lead_category_spike_times_ms,
            strict=True,
        )
        if index == int(profile["learned_state"]["source_index"])
    )
    gates = {
        "verified_match_complete": bool(match_outcome["pass"]),
        "one_selected_category_event_during_lead": len(source_events) == 1,
        "no_off_source_category_events_during_lead": all(
            index == int(profile["learned_state"]["source_index"])
            for index in mismatch.cue_lead_category_spike_indices
        ),
        "no_relay_events_during_lead": not mismatch.cue_lead_relay_spike_times_ms,
        "current_terminated_on_selected_event": (
            bool(source_events)
            and mismatch.top_down_current_termination_time_ms == source_events[0]
        ),
        "match_relay_spatial_pattern": assessment.pathway.relay_spatial_match_pass,
        "mismatch_relay_overlap_only": assessment.pathway.relay_mismatch_overlap_pass,
        "match_more_active_relay_cells": assessment.pathway.relay_subset_pass,
        "match_more_trn_events": assessment.pathway.trn_order_pass,
        "mismatch_more_nonspecific_events": assessment.arousal.mismatch_disinhibition_pass,
        "figure7_target_duration": assessment.arousal.target_duration_pass,
        "match_nonspecific_40_hz": assessment.arousal.match_numeric_target_pass,
        "mismatch_nonspecific_70_hz": assessment.arousal.mismatch_numeric_target_pass,
        "sampled_mismatch_trn_events_have_fresh_cycles": fresh_cycles,
    }
    reproduced = all(gates.values()) and assessment.reproduced
    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "figure7-reproduced" if reproduced else "figure7-failed",
        "profile": args.profile,
        "registration_artifact": (
            args.diagnostic_registration or profile["registration_artifact"]
        ),
        "original_holdout_registration_artifact": (
            profile["registration_artifact"]
            if args.diagnostic_registration
            else None
        ),
        "base_candidate_fingerprint": base_profile["candidate_fingerprint"],
        "runtime_fingerprint": conventions.fingerprint,
        "figure6_artifact": profile["figure6_artifact"],
        "match_verification_artifact": profile["match_verification_artifact"],
        "selected_headroom_fraction": selected_fraction,
        "applied_common_weight_factor": applied_factor,
        "corticoreticular_common_gain": corticoreticular,
        "holdouts_consulted": ["figure7_match", "figure7_mismatch"],
        "handoff_figure6_population_spikes": training.result.population_spikes,
        "match_scoring_summary": match,
        "mismatch_result": mismatch,
        "sampled_mismatch_trn_event_counts_by_index": event_counts,
        "sampled_mismatch_trn_threshold_upcrossings_by_index": upcrossings,
        "sampled_mismatch_trn_arm_transitions_by_index": arms,
        "sampled_mismatch_trn_release_transitions_by_index": releases,
        "official_assessment": assessment,
        "gates": gates,
        "reproduced": reproduced,
        "next_gate": profile["next_gate"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))
    print(
        f"match relay/trn/ns={len(match.relay_spike_times_ms)}/"
        f"{len(match.trn_spike_times_ms)}/{len(match.nonspecific_spike_times_ms)} "
        f"mismatch relay/trn/ns={len(mismatch.relay_spike_times_ms)}/"
        f"{len(mismatch.trn_spike_times_ms)}/"
        f"{len(mismatch.nonspecific_spike_times_ms)} reproduced={reproduced}",
        flush=True,
    )


if __name__ == "__main__":
    main()
