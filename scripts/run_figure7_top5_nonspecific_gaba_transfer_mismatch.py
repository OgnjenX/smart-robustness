"""Run the fixed top-five mismatch at verified nonspecific GABA transfer."""

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
    Figure6LearningProtocol,
    assess_figure6_cortical_recruitment,
    assess_figure6_top_down_timing,
    run_figure6_learning,
)
from smart_robustness.validation.figure7 import (
    Figure7ConditionResult,
    TopDownCurrentMode,
    assess_figure7_reproduction,
    expand_figure7_source_expectation_toward_bounds,
    run_figure7_condition,
)

FIGURE6_MONITORED_POPULATIONS = (
    "thalamic_relay",
    "layer4_excitatory_v1",
    "layer23_excitatory_v1",
    "layer5_excitatory_v1",
    "layer6i_excitatory_v1",
    "layer6ii_excitatory_v1",
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
        comparator_source_index=raw.get("comparator_source_index"),
        comparator_transform=raw.get("comparator_transform"),
        comparator_target_count=raw.get("comparator_target_count"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default=(
            "configs/calibration/"
            "figure7_top5_nonspecific_gaba_transfer_mismatch_v1.yaml"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    base_profile = yaml.safe_load(Path(profile["base_profile"]).read_text())
    verified = yaml.safe_load(
        Path(profile["match_verification_artifact"]).read_text()
    )
    if not verified["assessment"].get("advance_to_mismatch", False):
        raise ValueError("verified match does not authorize mismatch")
    match_outcome = verified["outcomes"][0]
    common_scale = float(profile["nonspecific_gaba_transfer"]["common_scale"])
    if match_outcome["common_scale"] != common_scale or not match_outcome["pass"]:
        raise ValueError("registered mismatch differs from verified match")
    match = _scoring_result(match_outcome["match_result"])

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
    scales.update(
        {
            str(projection_id): common_scale
            for projection_id in profile["nonspecific_gaba_transfer"][
                "projection_ids"
            ]
        }
    )
    training = run_figure6_learning(
        conventions=conventions,
        protocol=Figure6LearningProtocol(
            monitored_populations=FIGURE6_MONITORED_POPULATIONS
        ),
        projection_weight_scales=scales,
        brian=brian,
    )
    training_result = training.result
    training_indices = training_result.population_spike_indices["thalamic_relay"]
    training_counts = {index: training_indices.count(index) for index in range(81)}
    training_counts = {index: count for index, count in training_counts.items() if count}
    recruitment = assess_figure6_cortical_recruitment(training_result)
    timing = assess_figure6_top_down_timing(training_result)
    figure6_gates = {
        "relay_active_indices": set(training_indices) == {38, 39, 40, 41, 42},
        "relay_events_per_active_index": set(training_counts.values()) == {4},
        "relay_events": len(training_indices) == 20,
        "cortical_chain_complete": recruitment.feedforward_chain_complete,
        "causal_learning_pair": timing.causal_pair_in_learning_window,
        "top_down_horizontal_contrast": (
            training_result.top_down_combined.horizontal_orientation_contrast > 0
        ),
    }
    if not all(figure6_gates.values()):
        raise ValueError("fresh mismatch handoff failed complete Figure 6")

    learned, applied_factor = expand_figure7_source_expectation_toward_bounds(
        training.learned_weights,
        headroom_fraction=float(profile["learned_state"]["headroom_fraction"]),
        source_index=int(profile["learned_state"]["source_index"]),
    )
    protocol = profile["protocol"]
    mismatch = run_figure7_condition(
        condition=MatchCondition.MISMATCH,
        top_down_current_pA=float(protocol["top_down_current_pA"]),
        learned_weights=learned,
        conventions=conventions,
        duration_ms=float(protocol["duration_ms"]),
        dt_ms=float(protocol["dt_ms"]),
        record_relay_diagnostics=bool(protocol["record_relay_diagnostics"]),
        persistent_projection_weight_scales=scales,
        comparator_top_k_targets=int(profile["comparator"]["target_count"]),
        comparator_source_index=int(profile["learned_state"]["source_index"]),
        top_down_current_mode=TopDownCurrentMode(
            protocol["top_down_current_mode"]
        ),
        top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
        equilibration_ms=float(protocol["equilibration_ms"]),
        brian=brian,
    )
    assessment = assess_figure7_reproduction(match, mismatch)
    sampled = tuple(index for index, _ in mismatch.trn_detector_arm_transitions_by_index)
    event_counts = {
        index: mismatch.trn_spike_indices.count(index) for index in sampled
    }
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
    gates = {
        "figure6_prerequisite": all(figure6_gates.values()),
        "match_relay_spatial_set": assessment.pathway.relay_spatial_match_pass,
        "mismatch_relay_overlap_only": assessment.pathway.relay_mismatch_overlap_pass,
        "match_more_active_relay_cells": assessment.pathway.relay_subset_pass,
        "match_more_trn_events": assessment.pathway.trn_order_pass,
        "match_nonspecific_40_hz": assessment.arousal.match_numeric_target_pass,
        "mismatch_more_nonspecific_events": (
            assessment.arousal.mismatch_disinhibition_pass
        ),
        "mismatch_nonspecific_70_hz": (
            assessment.arousal.mismatch_numeric_target_pass
        ),
        "figure7_target_duration": assessment.arousal.target_duration_pass,
        "sampled_mismatch_trn_events_have_fresh_cycles": fresh_cycles,
    }
    reproduced = all(gates.values())
    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "figure7-reproduced" if reproduced else "figure7-failed",
        "classification": "calibrated-reconstruction-not-recovered-source",
        "profile": args.profile,
        "registration_artifact": profile["registration_artifact"],
        "match_verification_artifact": profile["match_verification_artifact"],
        "base_candidate_fingerprint": base_profile["candidate_fingerprint"],
        "runtime_fingerprint": conventions.fingerprint,
        "nonspecific_gaba_common_scale": common_scale,
        "projection_weight_scales": scales,
        "figure6_result": training_result,
        "figure6_relay_event_counts_by_index": training_counts,
        "figure6_cortical_recruitment": recruitment,
        "figure6_top_down_timing": timing,
        "figure6_gates": figure6_gates,
        "applied_common_weight_factor": applied_factor,
        "match_scoring_summary": match,
        "mismatch_result": mismatch,
        "sampled_mismatch_trn_event_counts_by_index": event_counts,
        "sampled_mismatch_trn_threshold_upcrossings_by_index": upcrossings,
        "sampled_mismatch_trn_arm_transitions_by_index": arms,
        "sampled_mismatch_trn_release_transitions_by_index": releases,
        "assessment": assessment,
        "gates": gates,
        "reproduced": reproduced,
        "next_gate": profile["next_gate"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))
    print(
        f"mismatch relay={len(mismatch.relay_spike_times_ms)} "
        f"active={sorted(set(mismatch.relay_spike_indices))} "
        f"trn={len(mismatch.trn_spike_times_ms)} "
        f"nonspecific={len(mismatch.nonspecific_spike_times_ms)} "
        f"reproduced={reproduced}",
        flush=True,
    )


if __name__ == "__main__":
    main()
