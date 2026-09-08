"""Verify the sole persistent recurrent-GABA survivor on a fixed Figure 7 pair."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import yaml
from run_figure7_peripheral_trn_summary import (
    CENTRAL,
    PERIPHERAL,
    WHOLE,
    _condition_data,
    _largest_peripheral_spike_excess,
    _order_counts,
    _region_summary,
)

from smart_robustness.protocols import MatchCondition
from smart_robustness.validation import figure7 as figure7_module
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import (
    Figure6LearningProtocol,
    assess_figure6_cortical_recruitment,
    assess_figure6_top_down_timing,
    run_figure6_learning,
)
from smart_robustness.validation.figure7 import TopDownCurrentMode


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
    if figure7_module.FIGURE7_RELAY_DIAGNOSTIC_INDICES != tuple(sorted(CENTRAL)):
        raise ValueError("unexpected default Figure 7 diagnostic index set")
    figure7_module.FIGURE7_RELAY_DIAGNOSTIC_INDICES = WHOLE

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    persistent_scales = {
        str(key): float(value)
        for key, value in training_profile["projection_weight_scales"].items()
    }
    for item in registration["persistent_projection_scales"]:
        projection_id = item["projection_id"]
        if projection_id in persistent_scales:
            raise ValueError(f"{projection_id} overlaps an existing scale")
        persistent_scales[projection_id] = float(item["scale"])

    training = run_figure6_learning(
        conventions=conventions,
        protocol=Figure6LearningProtocol(
            monitored_populations=tuple(training_profile["monitored_populations"])
        ),
        projection_weight_scales=persistent_scales,
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
    if not all(figure6_gates.values()):
        raise RuntimeError(f"fresh Figure 6 prerequisite failed: {figure6_gates}")

    protocol = profile["protocol"]
    comparator = profile["comparator"]
    common = {
        "learned_weights": training.learned_weights,
        "conventions": conventions,
        "persistent_projection_weight_scales": persistent_scales,
        "top_down_current_pA": float(protocol["top_down_current_pA"]),
        "top_down_current_mode": TopDownCurrentMode(protocol["top_down_current_mode"]),
        "top_down_cue_lead_ms": float(protocol["top_down_cue_lead_ms"]),
        "duration_ms": float(protocol["duration_ms"]),
        "dt_ms": float(protocol["dt_ms"]),
        "equilibration_ms": float(protocol["equilibration_ms"]),
        "comparator_top_k_targets": int(comparator["target_count"]),
        "comparator_source_index": int(comparator["source_index"]),
        "record_relay_diagnostics": True,
        "brian": brian,
    }
    match_result = figure7_module.run_figure7_condition(condition=MatchCondition.MATCH, **common)
    mismatch_result = figure7_module.run_figure7_condition(
        condition=MatchCondition.MISMATCH, **common
    )
    match = _condition_data(match_result)
    mismatch = _condition_data(mismatch_result)
    regions = {
        "central": tuple(sorted(CENTRAL)),
        "peripheral": PERIPHERAL,
        "whole": WHOLE,
    }
    event_identity = {
        "match": {
            "relay_events": len(match_result.relay_spike_times_ms),
            "relay_active_indices": sorted(set(match_result.relay_spike_indices)),
            "trn_events": len(match_result.trn_spike_times_ms),
            "nonspecific_events": len(match_result.nonspecific_spike_times_ms),
        },
        "mismatch": {
            "relay_events": len(mismatch_result.relay_spike_times_ms),
            "relay_active_indices": sorted(set(mismatch_result.relay_spike_indices)),
            "trn_events": len(mismatch_result.trn_spike_times_ms),
            "nonspecific_events": len(mismatch_result.nonspecific_spike_times_ms),
        },
    }
    fixed_gates = {
        "match_relay_active_indices": event_identity["match"]["relay_active_indices"]
        == [38, 39, 40, 41, 42],
        "match_relay_events": event_identity["match"]["relay_events"] == 20,
        "mismatch_relay_active_indices": event_identity["mismatch"]["relay_active_indices"] == [40],
        "match_more_active_relay_cells": len(event_identity["match"]["relay_active_indices"])
        > len(event_identity["mismatch"]["relay_active_indices"]),
        "match_more_trn_events": event_identity["match"]["trn_events"]
        > event_identity["mismatch"]["trn_events"],
        "match_nonspecific_events": event_identity["match"]["nonspecific_events"] == 4,
        "mismatch_nonspecific_events": event_identity["mismatch"]["nonspecific_events"] == 7,
    }
    print(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": registration["result_id"],
                "date": registration["date"],
                "status": "completed-persistent-recurrent-gaba-fixed-pair",
                "classification": registration["classification"],
                "registration": args.registration,
                "runtime_fingerprint": conventions.fingerprint,
                "persistent_projection_weight_scales": persistent_scales,
                "figure6_gates": figure6_gates,
                "event_identity": event_identity,
                "fixed_figure7_gates": fixed_gates,
                "region_summaries": {
                    name: {
                        "match": _region_summary(match, indices),
                        "mismatch": _region_summary(mismatch, indices),
                    }
                    for name, indices in regions.items()
                },
                "peripheral_per_cell_order_counts": {
                    field: _order_counts(match, mismatch, PERIPHERAL, field)
                    for field in (
                        "events",
                        "relay_ampa",
                        "layer6ii_ampa",
                        "layer6ii_nmda",
                    )
                },
                "largest_peripheral_spike_excess": (
                    _largest_peripheral_spike_excess(match, mismatch)
                ),
                "nonspecific_trn_gaba_integral_ms": {
                    "match": match_result.nonspecific_trn_gaba_integral_ms,
                    "mismatch": mismatch_result.nonspecific_trn_gaba_integral_ms,
                },
                "model_or_protocol_change": False,
                "instrumentation_change_only": True,
                "original_smart_reproduced": False,
                "baseline_promoted": False,
            },
            sort_keys=False,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
