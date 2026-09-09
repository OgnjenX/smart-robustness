"""Run the fixed Figure 7 consistency pair for the selected p036 endpoint."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from smart_robustness import classic_sector
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
from smart_robustness.validation.figure10_search_cycle_spread import (
    PROJECTION036_ID,
    build_projection036_variance_sector,
)


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
    for item in profile["persistent_projection_scales"]:
        projection_id = str(item["projection_id"])
        if projection_id in scales:
            raise ValueError(f"{projection_id} overlaps an existing scale")
        scales[projection_id] = float(item["scale"])

    topology_observations: list[dict[str, float | int]] = []

    def build_observed_sector(*builder_args, **builder_kwargs):
        sector = build_projection036_variance_sector(*builder_args, **builder_kwargs)
        projection = sector.projections[PROJECTION036_ID]
        weights = np.asarray(projection.w[:], dtype=float)
        targets = np.asarray(projection.j[:], dtype=int)
        topology_observations.append(
            {
                "nonzero_edges": int(np.count_nonzero(weights)),
                "incoming_weight_sum_target40": float(np.sum(weights[targets == 40])),
            }
        )
        return sector

    original_builder = classic_sector.build_first_order_connected_sector
    classic_sector.build_first_order_connected_sector = build_observed_sector
    try:
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
        if not all(figure6_gates.values()):
            raise RuntimeError(f"fresh Figure 6 prerequisite failed: {figure6_gates}")

        protocol = profile["protocol"]
        comparator = profile["comparator"]
        common = {
            "learned_weights": training.learned_weights,
            "conventions": conventions,
            "persistent_projection_weight_scales": scales,
            "top_down_current_pA": float(protocol["top_down_current_pA"]),
            "top_down_current_mode": TopDownCurrentMode(
                protocol["top_down_current_mode"]
            ),
            "top_down_cue_lead_ms": float(protocol["top_down_cue_lead_ms"]),
            "duration_ms": float(protocol["duration_ms"]),
            "dt_ms": float(protocol["dt_ms"]),
            "equilibration_ms": float(protocol["equilibration_ms"]),
            "comparator_top_k_targets": int(comparator["target_count"]),
            "comparator_source_index": int(comparator["source_index"]),
            "record_relay_diagnostics": True,
            "brian": brian,
        }
        match = figure7_module.run_figure7_condition(
            condition=MatchCondition.MATCH, **common
        )
        mismatch = figure7_module.run_figure7_condition(
            condition=MatchCondition.MISMATCH, **common
        )
    finally:
        classic_sector.build_first_order_connected_sector = original_builder

    expected_topology = profile["projection036_topology"]
    topology_gates = {
        "three_sector_builds_observed": len(topology_observations) == 3,
        "nonzero_edges_all_builds": all(
            item["nonzero_edges"] == expected_topology["nonzero_edges"]
            for item in topology_observations
        ),
        "incoming_weight_sum_all_builds": all(
            np.isclose(
                item["incoming_weight_sum_target40"],
                expected_topology["incoming_weight_sum_per_target"],
                rtol=0.0,
                atol=1e-12,
            )
            for item in topology_observations
        ),
    }
    match_active = sorted(set(match.relay_spike_indices))
    mismatch_active = sorted(set(mismatch.relay_spike_indices))
    gates = profile["figure7_gates"]
    event_identity = {
        "match": {
            "relay_events": len(match.relay_spike_times_ms),
            "relay_active_indices": match_active,
            "relay_event_indices": list(match.relay_spike_indices),
            "relay_event_times_ms": list(match.relay_spike_times_ms),
            "trn_events": len(match.trn_spike_times_ms),
            "nonspecific_events": len(match.nonspecific_spike_times_ms),
            "nonspecific_event_times_ms": list(match.nonspecific_spike_times_ms),
            "trn_to_nonspecific_gaba_integral_ms": (
                match.nonspecific_trn_gaba_integral_ms
            ),
        },
        "mismatch": {
            "relay_events": len(mismatch.relay_spike_times_ms),
            "relay_active_indices": mismatch_active,
            "relay_event_indices": list(mismatch.relay_spike_indices),
            "relay_event_times_ms": list(mismatch.relay_spike_times_ms),
            "trn_events": len(mismatch.trn_spike_times_ms),
            "nonspecific_events": len(mismatch.nonspecific_spike_times_ms),
            "nonspecific_event_times_ms": list(mismatch.nonspecific_spike_times_ms),
            "trn_to_nonspecific_gaba_integral_ms": (
                mismatch.nonspecific_trn_gaba_integral_ms
            ),
        },
    }
    figure7_gates = {
        "match_relay_active_indices": match_active
        == gates["match_relay_active_indices"],
        "match_relay_events": len(match.relay_spike_times_ms)
        == gates["match_relay_events"],
        "mismatch_relay_allowed_indices": set(mismatch_active).issubset(
            set(gates["mismatch_relay_allowed_indices"])
        )
        and bool(mismatch_active),
        "match_more_active_relay_cells": len(match_active) > len(mismatch_active),
        "match_more_trn_events": len(match.trn_spike_times_ms)
        > len(mismatch.trn_spike_times_ms),
        "match_more_trn_to_nonspecific_gaba": (
            match.nonspecific_trn_gaba_integral_ms
            > mismatch.nonspecific_trn_gaba_integral_ms
        ),
        "match_nonspecific_events": len(match.nonspecific_spike_times_ms)
        == gates["match_nonspecific_events"],
        "mismatch_nonspecific_events": len(mismatch.nonspecific_spike_times_ms)
        == gates["mismatch_nonspecific_events"],
    }
    reproduced = (
        all(topology_gates.values())
        and all(figure6_gates.values())
        and all(figure7_gates.values())
    )
    print(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": registration["result_id"],
                "date": registration["date"],
                "status": "completed-figure7-projection036-spread-consistency",
                "classification": registration["classification"],
                "registration": args.registration,
                "runtime_fingerprint": conventions.fingerprint,
                "persistent_projection_weight_scales": scales,
                "projection036_topology_observations": topology_observations,
                "projection036_topology_gates": topology_gates,
                "figure6_gates": figure6_gates,
                "event_identity": event_identity,
                "figure7_gates": figure7_gates,
                "figure7_consistency_reproduced": reproduced,
                "parameter_adjustment_after_figure10": False,
                "original_smart_reproduced": False,
                "baseline_promoted": False,
            },
            sort_keys=False,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
