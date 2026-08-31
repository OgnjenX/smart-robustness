"""Complete and score the calibrated same-network Figure 7 pair."""

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
    assess_figure7_reproduction,
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


def _pairs(values: tuple[tuple[int, Any], ...]) -> dict[int, Any]:
    return {int(index): value for index, value in values}


def _scoring_result(raw: dict[str, Any]) -> Figure7ConditionResult:
    return Figure7ConditionResult(
        condition=MatchCondition(raw["condition"]),
        duration_ms=float(raw["duration_ms"]),
        nonspecific_spike_times_ms=tuple(raw["nonspecific_spike_times_ms"]),
        relay_spike_indices=tuple(raw["relay_spike_indices"]),
        relay_spike_times_ms=tuple(raw["relay_spike_times_ms"]),
        trn_spike_indices=tuple(raw["trn_spike_indices"]),
        trn_spike_times_ms=tuple(raw["trn_spike_times_ms"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="configs/calibration/trn_gaba_transfer_figure7_pair_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    base_profile = yaml.safe_load(Path(profile["base_profile"]).read_text())
    match_artifact = yaml.safe_load(Path(profile["match_artifact"]).read_text())
    if not match_artifact["assessment"]["advance_to_mismatch"]:
        raise ValueError("Artifact 203 does not authorize mismatch")
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
    transfer = profile["trn_to_relay_gaba"]
    if "scales" in transfer:
        projection_scales = {
            str(projection_id): float(scale)
            for projection_id, scale in transfer["scales"].items()
        }
    else:
        scale = float(transfer["scale"])
        projection_scales = {
            str(projection_id): scale
            for projection_id in transfer["projection_ids"]
        }
    if projection_scales != match_artifact["projection_weight_scales"]:
        raise ValueError("mismatch projection scales differ from Artifact 203")
    protocol = profile["protocol"]
    learned_state_handoff = protocol.get(
        "learned_state_handoff", "contiguous_same_network"
    )
    if learned_state_handoff != match_artifact.get(
        "learned_state_handoff", "contiguous_same_network"
    ):
        raise ValueError("mismatch learned-state handoff differs from match")
    handoff_training = None
    if learned_state_handoff == "fresh_network_from_figure6_weights":
        handoff_training = run_figure6_learning(
            conventions=conventions,
            projection_weight_scales=projection_scales,
            brian=brian,
        )
        learning_arguments = {
            "learned_weights": handoff_training.learned_weights,
            "pretrain_with_figure6_episode": False,
        }
    elif learned_state_handoff == "contiguous_same_network":
        learning_arguments = {
            "pretrain_with_figure6_episode": True,
        }
    else:
        raise ValueError(f"unsupported learned-state handoff {learned_state_handoff!r}")
    mismatch = run_figure7_condition(
        condition=MatchCondition.MISMATCH,
        top_down_current_pA=float(protocol["top_down_current_pA"]),
        conventions=conventions,
        duration_ms=float(protocol["duration_ms"]),
        dt_ms=float(protocol["dt_ms"]),
        record_relay_diagnostics=bool(protocol["record_relay_diagnostics"]),
        persistent_projection_weight_scales=projection_scales,
        top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
        equilibration_ms=float(protocol["equilibration_ms"]),
        brian=brian,
        **learning_arguments,
    )
    match = _scoring_result(match_artifact["result"])
    assessment = assess_figure7_reproduction(match, mismatch)
    event_counts = {
        index: mismatch.trn_spike_indices.count(index)
        for index, _ in mismatch.trn_detector_arm_transitions_by_index
    }
    upcrossings = _pairs(mismatch.trn_detector_threshold_upcrossings_by_index)
    arms = _pairs(mismatch.trn_detector_arm_transitions_by_index)
    releases = _pairs(mismatch.trn_detector_release_transitions_by_index)
    mismatch_fresh_cycles = all(
        event_counts[index] == upcrossings[index] == arms[index] == releases[index]
        for index in event_counts
    ) and any(event_counts.values())
    gates = {
        "match_relay_spatial_pattern": assessment.pathway.relay_spatial_match_pass,
        "mismatch_relay_overlap_only": assessment.pathway.relay_mismatch_overlap_pass,
        "match_more_active_relay_cells": assessment.pathway.relay_subset_pass,
        "match_more_trn_events": assessment.pathway.trn_order_pass,
        "mismatch_more_nonspecific_events": (
            assessment.arousal.mismatch_disinhibition_pass
        ),
        "sampled_mismatch_trn_events_have_fresh_cycles": mismatch_fresh_cycles,
    }
    reproduced = all(gates.values()) and assessment.reproduced
    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "figure7-reproduced" if reproduced else "figure7-failed",
        "profile": args.profile,
        "base_candidate_fingerprint": base_profile["candidate_fingerprint"],
        "runtime_fingerprint": conventions.fingerprint,
        "figure6_artifact": profile["figure6_artifact"],
        "match_artifact": profile["match_artifact"],
        "holdouts_consulted": ["figure7_match", "figure7_mismatch"],
        "projection_weight_scales": projection_scales,
        "learned_state_handoff": learned_state_handoff,
        "handoff_figure6_population_spikes": (
            None
            if handoff_training is None
            else handoff_training.result.population_spikes
        ),
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
        f"{len(mismatch.nonspecific_spike_times_ms)} "
        f"reproduced={reproduced}",
        flush=True,
    )


if __name__ == "__main__":
    main()
