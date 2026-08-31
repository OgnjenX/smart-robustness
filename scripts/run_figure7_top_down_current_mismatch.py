"""Run the independently rebuilt mismatch for the sole current-grid survivor."""

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
        cue_lead_category_spike_indices=tuple(
            raw.get("cue_lead_category_spike_indices", ())
        ),
        cue_lead_category_spike_times_ms=tuple(
            raw.get("cue_lead_category_spike_times_ms", ())
        ),
        cue_lead_nonspecific_spike_times_ms=tuple(
            raw.get("cue_lead_nonspecific_spike_times_ms", ())
        ),
        cue_lead_trn_spike_indices=tuple(raw.get("cue_lead_trn_spike_indices", ())),
        cue_lead_trn_spike_times_ms=tuple(
            raw.get("cue_lead_trn_spike_times_ms", ())
        ),
        cue_lead_relay_spike_indices=tuple(
            raw.get("cue_lead_relay_spike_indices", ())
        ),
        cue_lead_relay_spike_times_ms=tuple(
            raw.get("cue_lead_relay_spike_times_ms", ())
        ),
        top_down_current_mode=raw.get(
            "top_down_current_mode", TopDownCurrentMode.SUSTAINED_EPOCH.value
        ),
        top_down_current_termination_time_ms=raw.get(
            "top_down_current_termination_time_ms"
        ),
        top_down_cue_lead_ms=float(raw.get("top_down_cue_lead_ms", 0.0)),
        top_down_relay_source_indices=(
            None
            if raw.get("top_down_relay_source_indices") is None
            else tuple(raw["top_down_relay_source_indices"])
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="configs/calibration/figure7_top_down_current_800_fresh_pair_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    base_profile = yaml.safe_load(Path(profile["base_profile"]).read_text())
    match_screen = yaml.safe_load(Path(profile["match_screen_artifact"]).read_text())
    selected_current = float(profile["selected_current_pA"])
    if match_screen["survivor_currents_pA"] != [selected_current]:
        raise ValueError("selected current is not the sole exact match survivor")
    match_outcome = next(
        item
        for item in match_screen["outcomes"]
        if float(item["top_down_current_pA"]) == selected_current
    )
    if not match_outcome["pass"]:
        raise ValueError("selected current did not pass complete match")
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
    training = run_figure6_learning(
        conventions=conventions,
        projection_weight_scales=scales,
        brian=brian,
    )
    if training.result.population_spikes["thalamic_relay"] != 20:
        raise ValueError("fresh handoff did not reproduce the Figure 6 relay train")
    protocol = profile["protocol"]
    current_mode = TopDownCurrentMode(
        protocol.get("top_down_current_mode", TopDownCurrentMode.SUSTAINED_EPOCH)
    )
    relay_source_indices = protocol.get("top_down_relay_source_indices")
    relay_source_indices = (
        None
        if relay_source_indices is None
        else frozenset(int(index) for index in relay_source_indices)
    )
    expected_relay_source_indices = (
        None
        if relay_source_indices is None
        else tuple(sorted(relay_source_indices))
    )
    if match.top_down_relay_source_indices != expected_relay_source_indices:
        raise ValueError("match artifact does not use the registered relay-source mask")
    mismatch = run_figure7_condition(
        condition=MatchCondition.MISMATCH,
        top_down_current_pA=selected_current,
        learned_weights=training.learned_weights,
        conventions=conventions,
        duration_ms=float(protocol["duration_ms"]),
        dt_ms=float(protocol["dt_ms"]),
        record_relay_diagnostics=bool(protocol["record_relay_diagnostics"]),
        persistent_projection_weight_scales=scales,
        top_down_relay_source_indices=relay_source_indices,
        top_down_current_mode=current_mode,
        top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
        equilibration_ms=float(protocol["equilibration_ms"]),
        brian=brian,
    )
    assessment = assess_figure7_reproduction(match, mismatch)
    event_counts = {
        index: mismatch.trn_spike_indices.count(index)
        for index, _ in mismatch.trn_detector_arm_transitions_by_index
    }
    upcrossings = _pairs(mismatch.trn_detector_threshold_upcrossings_by_index)
    arms = _pairs(mismatch.trn_detector_arm_transitions_by_index)
    releases = _pairs(mismatch.trn_detector_release_transitions_by_index)
    fresh_cycles = bool(event_counts) and any(event_counts.values()) and all(
        event_counts[index]
        == upcrossings[index]
        == arms[index]
        == releases[index]
        for index in event_counts
    )
    cue_lead_source_events_ms = tuple(
        time
        for index, time in zip(
            mismatch.cue_lead_category_spike_indices,
            mismatch.cue_lead_category_spike_times_ms,
            strict=True,
        )
        if index == 40
    )
    first_stimulus_cued_event_ms = next(
        (
            time
            for index, time in zip(
                mismatch.category_spike_indices,
                mismatch.category_spike_times_ms,
                strict=True,
            )
            if index == 40
        ),
        None,
    )
    first_cued_event_ms = (
        cue_lead_source_events_ms[0]
        if cue_lead_source_events_ms
        else None
        if first_stimulus_cued_event_ms is None
        else float(protocol["top_down_cue_lead_ms"])
        + first_stimulus_cued_event_ms
    )
    gates = {
        "match_exact_complete_gate": bool(match_outcome["pass"]),
        "match_top_down_relay_source_protocol": (
            match.top_down_relay_source_indices == expected_relay_source_indices
        ),
        "mismatch_top_down_relay_source_protocol": (
            mismatch.top_down_relay_source_indices == expected_relay_source_indices
        ),
        "mismatch_top_down_current_protocol": (
            current_mode is TopDownCurrentMode.SUSTAINED_EPOCH
            or (
                first_cued_event_ms is not None
                and mismatch.top_down_current_termination_time_ms
                == first_cued_event_ms
            )
        ),
        "match_relay_spatial_pattern": assessment.pathway.relay_spatial_match_pass,
        "mismatch_relay_overlap_only": assessment.pathway.relay_mismatch_overlap_pass,
        "match_more_active_relay_cells": assessment.pathway.relay_subset_pass,
        "match_more_trn_events": assessment.pathway.trn_order_pass,
        "mismatch_more_nonspecific_events": (
            assessment.arousal.mismatch_disinhibition_pass
        ),
        "figure7_target_duration": assessment.arousal.target_duration_pass,
        "match_nonspecific_40_hz": assessment.arousal.match_numeric_target_pass,
        "mismatch_nonspecific_70_hz": (
            assessment.arousal.mismatch_numeric_target_pass
        ),
        "sampled_mismatch_trn_events_have_fresh_cycles": fresh_cycles,
    }
    cue_lead_gate = profile.get("cue_lead_gate")
    if cue_lead_gate is not None:
        gates.update(
            {
                "mismatch_one_selected_category_event_during_lead": (
                    len(cue_lead_source_events_ms)
                    == int(cue_lead_gate["selected_category_events"])
                ),
                "mismatch_no_off_source_category_events_during_lead": all(
                    index == 40
                    for index in mismatch.cue_lead_category_spike_indices
                ),
                "mismatch_no_relay_events_during_lead": (
                    len(mismatch.cue_lead_relay_spike_times_ms)
                    == int(cue_lead_gate["relay_events"])
                ),
                "mismatch_no_nonspecific_events_during_lead": (
                    len(mismatch.cue_lead_nonspecific_spike_times_ms)
                    == int(cue_lead_gate["nonspecific_events"])
                ),
            }
        )
    phenotype_reproduced = all(gates.values()) and assessment.reproduced
    diagnostic_only = bool(profile.get("diagnostic_only", False))
    reproduced = phenotype_reproduced and not diagnostic_only
    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": (
            "diagnostic-phenotype-pass"
            if phenotype_reproduced and diagnostic_only
            else "figure7-reproduced"
            if reproduced
            else "figure7-failed"
        ),
        "profile": args.profile,
        "registration_artifact": profile["registration_artifact"],
        "base_candidate_fingerprint": base_profile["candidate_fingerprint"],
        "runtime_fingerprint": conventions.fingerprint,
        "figure6_artifact": profile["figure6_artifact"],
        "match_screen_artifact": profile["match_screen_artifact"],
        "selected_current_pA": selected_current,
        "holdouts_consulted": ["figure7_match", "figure7_mismatch"],
        "learned_state_handoff": protocol["learned_state_handoff"],
        "handoff_figure6_population_spikes": training.result.population_spikes,
        "match_scoring_summary": match,
        "mismatch_result": mismatch,
        "sampled_mismatch_trn_event_counts_by_index": event_counts,
        "sampled_mismatch_trn_threshold_upcrossings_by_index": upcrossings,
        "sampled_mismatch_trn_arm_transitions_by_index": arms,
        "sampled_mismatch_trn_release_transitions_by_index": releases,
        "official_assessment": assessment,
        "gates": gates,
        "diagnostic_only": diagnostic_only,
        "phenotype_reproduced": phenotype_reproduced,
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
        f"phenotype_reproduced={phenotype_reproduced} "
        f"official_reproduced={reproduced}",
        flush=True,
    )


if __name__ == "__main__":
    main()
