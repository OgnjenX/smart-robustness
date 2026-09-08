"""Test one prior TRN-to-nonspecific GABA endpoint under repaired TRN balance."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path

import yaml

from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import (
    Figure6LearningProtocol,
    assess_figure6_cortical_recruitment,
    assess_figure6_top_down_timing,
    run_figure6_learning,
)
from smart_robustness.validation.figure7 import TopDownCurrentMode, run_figure7_condition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    registration = yaml.safe_load(Path(args.registration).read_text())
    profile = yaml.safe_load(Path(registration["profile"]).read_text())
    training_profile = yaml.safe_load(Path(profile["training_profile"]).read_text())
    base_profile = yaml.safe_load(Path(training_profile["base_profile"]).read_text())
    conventions = replace(
        runtime_conventions_for_candidate(base_profile["candidate"]),
        **training_profile["runtime_overrides"],
        **profile["runtime_overrides"],
    )
    if conventions.fingerprint != registration["runtime_fingerprint"]:
        raise ValueError("registration runtime differs from executable profile")

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    scales = {
        str(key): float(value)
        for key, value in training_profile["projection_weight_scales"].items()
    }
    for item in registration["persistent_projection_scales"]:
        projection_id = str(item["projection_id"])
        if projection_id in scales:
            raise ValueError(f"{projection_id} overlaps an existing scale")
        scales[projection_id] = float(item["scale"])
    common_scale = float(registration["nonspecific_gaba_common_scale"])
    for projection_id in registration["nonspecific_gaba_projection_ids"]:
        if projection_id in scales:
            raise ValueError(f"{projection_id} overlaps an existing scale")
        scales[str(projection_id)] = common_scale

    training = run_figure6_learning(
        conventions=conventions,
        protocol=Figure6LearningProtocol(
            monitored_populations=tuple(training_profile["monitored_populations"])
        ),
        projection_weight_scales=scales,
        brian=brian,
    )
    expected = set(profile["figure6_gates"]["relay_active_indices"])
    relay_indices = training.result.population_spike_indices["thalamic_relay"]
    relay_counts = {index: relay_indices.count(index) for index in expected}
    recruitment = assess_figure6_cortical_recruitment(training.result)
    timing = assess_figure6_top_down_timing(training.result)
    figure6_gates = {
        "relay_active_indices": set(relay_indices) == expected,
        "relay_events_per_active_index": set(relay_counts.values()) == {4},
        "relay_events": len(relay_indices) == 20,
        "cortical_chain_complete": recruitment.feedforward_chain_complete,
        "causal_learning_pair": timing.causal_pair_in_learning_window,
        "top_down_horizontal_contrast": (
            training.result.top_down_combined.horizontal_orientation_contrast > 0
        ),
    }
    match_summary = None
    if all(figure6_gates.values()):
        protocol = profile["protocol"]
        comparator = profile["comparator"]
        match = run_figure7_condition(
            condition=MatchCondition.MATCH,
            learned_weights=training.learned_weights,
            conventions=conventions,
            persistent_projection_weight_scales=scales,
            top_down_current_pA=float(protocol["top_down_current_pA"]),
            top_down_current_mode=TopDownCurrentMode(protocol["top_down_current_mode"]),
            top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
            duration_ms=float(protocol["duration_ms"]),
            dt_ms=float(protocol["dt_ms"]),
            equilibration_ms=float(protocol["equilibration_ms"]),
            comparator_top_k_targets=int(comparator["target_count"]),
            comparator_source_index=int(comparator["source_index"]),
            record_relay_diagnostics=True,
            brian=brian,
        )
        trn_counts = Counter(int(index) for index in match.trn_spike_indices)
        central = (22, 31, 38, 39, 40, 41, 42, 49, 58)
        central_total = sum(trn_counts[index] for index in central)
        match_summary = {
            "relay_event_count": len(match.relay_spike_times_ms),
            "relay_active_indices": sorted(set(match.relay_spike_indices)),
            "trn_event_count": len(match.trn_spike_times_ms),
            "trn_central_event_count": central_total,
            "trn_peripheral_event_count": len(match.trn_spike_times_ms) - central_total,
            "nonspecific_event_count": len(match.nonspecific_spike_times_ms),
            "nonspecific_spike_times_ms": list(match.nonspecific_spike_times_ms),
            "nonspecific_trn_gaba_integral_ms": (
                match.nonspecific_trn_gaba_integral_ms
            ),
        }
        match_summary["gates"] = {
            "relay_active_indices": match_summary["relay_active_indices"]
            == profile["figure7_gates"]["match_relay_active_indices"],
            "relay_events": match_summary["relay_event_count"]
            == profile["figure7_gates"]["match_relay_events"],
            "nonspecific_events": match_summary["nonspecific_event_count"]
            == profile["figure7_gates"]["match_nonspecific_events"],
        }

    print(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": registration["result_id"],
                "date": registration["date"],
                "status": "completed-persistent-output-gaba-match",
                "classification": registration["classification"],
                "registration": args.registration,
                "runtime_fingerprint": conventions.fingerprint,
                "persistent_projection_weight_scales": scales,
                "figure6_gates": figure6_gates,
                "match": match_summary,
                "mismatch_consulted": False,
                "original_smart_reproduced": False,
                "baseline_promoted": False,
            },
            sort_keys=False,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
