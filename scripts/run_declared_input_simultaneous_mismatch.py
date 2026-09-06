"""Run one unchanged mismatch to localize the failed simultaneous match."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import Figure6LearningProtocol, run_figure6_learning
from smart_robustness.validation.figure7 import TopDownCurrentMode, run_figure7_condition


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


def _verify_training(result: Any, reference: dict[str, Any]) -> None:
    actual = _plain(result)
    for field in (
        "convention_fingerprint",
        "duration_ms",
        "population_spike_indices",
        "population_spike_times_ms",
        "bottom_up",
        "top_down_wide",
        "top_down_narrow",
    ):
        if actual[field] != reference[field]:
            raise ValueError(f"fresh Figure 6 handoff differs: {field}")


def _fresh_trn_cycles(result: Any) -> bool:
    sampled = tuple(index for index, _ in result.trn_detector_arm_transitions_by_index)
    events = {index: result.trn_spike_indices.count(index) for index in sampled}
    up = dict(result.trn_detector_threshold_upcrossings_by_index)
    arms = dict(result.trn_detector_arm_transitions_by_index)
    releases = dict(result.trn_detector_release_transitions_by_index)
    return bool(events) and any(events.values()) and all(
        events[index] == up.get(index) == arms.get(index) == releases.get(index)
        for index in events
    )


def score_pair(match: dict[str, Any], mismatch: Any) -> dict[str, bool]:
    match_relay = set(match["relay_spike_indices"])
    mismatch_relay = set(mismatch.relay_spike_indices)
    return {
        "match_horizontal_relay_set": match_relay == {38, 39, 40, 41, 42},
        "mismatch_overlap_relay_set": bool(mismatch_relay)
        and mismatch_relay <= {40},
        "match_more_active_relay_cells": len(match_relay) > len(mismatch_relay),
        "match_more_trn_events": len(match["trn_spike_times_ms"])
        > len(mismatch.trn_spike_times_ms),
        "match_nonspecific_40_hz": len(match["nonspecific_spike_times_ms"]) == 4,
        "mismatch_more_nonspecific_events": len(mismatch.nonspecific_spike_times_ms)
        > len(match["nonspecific_spike_times_ms"]),
        "mismatch_nonspecific_70_hz": len(mismatch.nonspecific_spike_times_ms) == 7,
        "figure7_target_duration": match["duration_ms"]
        == mismatch.duration_ms
        == 100.0,
        "sampled_mismatch_trn_events_have_fresh_cycles": _fresh_trn_cycles(mismatch),
        "no_reconstructed_comparator": match["comparator_transform"] in (None, "none")
        and mismatch.comparator_transform in (None, "none"),
        "no_calcium_ablation": not match["relay_calcium_ablated_at_stimulus"]
        and not mismatch.relay_calcium_ablated_at_stimulus,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)

    registration = yaml.safe_load(Path(args.registration).read_text())
    match_artifact = yaml.safe_load(Path(registration["match_result"]).read_text())
    match = match_artifact["match_result"]
    profile = yaml.safe_load(Path(registration["training_profile"]).read_text())
    training_reference = yaml.safe_load(
        Path(registration["training_result"]).read_text()
    )
    base = yaml.safe_load(Path(profile["base_profile"]).read_text())
    conventions = replace(
        runtime_conventions_for_candidate(base["candidate"]),
        **profile["runtime_overrides"],
    )
    if conventions.fingerprint != registration["runtime_fingerprint"]:
        raise ValueError("registered runtime differs")

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    scales = profile["trn_to_relay_gaba"]["scales"]
    training = run_figure6_learning(
        conventions=conventions,
        protocol=Figure6LearningProtocol(
            monitored_populations=tuple(profile["monitored_populations"])
        ),
        projection_weight_scales=scales,
        brian=brian,
    )
    _verify_training(training.result, training_reference["result"])

    protocol = registration["protocol"]
    mismatch = run_figure7_condition(
        condition=MatchCondition.MISMATCH,
        learned_weights=training.learned_weights,
        conventions=conventions,
        persistent_projection_weight_scales=scales,
        top_down_current_pA=float(protocol["top_down_current_pA"]),
        top_down_current_mode=TopDownCurrentMode(protocol["top_down_current_mode"]),
        top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
        duration_ms=float(protocol["duration_ms"]),
        dt_ms=float(protocol["dt_ms"]),
        equilibration_ms=float(protocol["equilibration_ms"]),
        record_relay_diagnostics=True,
        record_interneuron_spikes=True,
        brian=brian,
    )
    gates = score_pair(match, mismatch)
    artifact = {
        "schema_version": 1,
        "registration": args.registration,
        "classification": "post-failure-causal-localization-not-holdout",
        "runtime_fingerprint": conventions.fingerprint,
        "training_repeat_verified": True,
        "weight_handoff": "actual_figure6_weights_no_expansion",
        "applied_common_weight_factor": 1.0,
        "match_result": registration["match_result"],
        "match_result_sha256": registration["match_result_sha256"],
        "match_scoring_summary": {
            "relay_spike_indices": match["relay_spike_indices"],
            "relay_spike_times_ms": match["relay_spike_times_ms"],
            "trn_spike_count": len(match["trn_spike_times_ms"]),
            "nonspecific_spike_times_ms": match["nonspecific_spike_times_ms"],
        },
        "mismatch_result": mismatch,
        "gates": gates,
        "all_official_pair_gates_pass": all(gates.values()),
        "original_smart_reproduced": False,
        "baseline_promoted": False,
        "interpretation_boundary": registration["interpretation_boundary"],
    }
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))
    print(
        f"mismatch relay={len(mismatch.relay_spike_times_ms)} "
        f"active={sorted(set(mismatch.relay_spike_indices))} "
        f"trn={len(mismatch.trn_spike_times_ms)} "
        f"nonspecific={len(mismatch.nonspecific_spike_times_ms)} "
        f"pair_pass={all(gates.values())}",
        flush=True,
    )


if __name__ == "__main__":
    main()

