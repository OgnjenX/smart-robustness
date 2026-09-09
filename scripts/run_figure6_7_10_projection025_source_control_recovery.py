"""Run the preregistered joint p025 source-control cross for Figures 6, 7, and 10."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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
    run_figure10_search_cycle_spread_condition,
)

PROJECTION025_ID = "modeldb112923.projection.025"


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
    if PROJECTION025_ID in scales:
        raise ValueError("source-control cross must retain projection 025 at scale 1")

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

        f7_protocol = profile["figure7_protocol"]
        comparator = profile["comparator"]
        figure7_common = {
            "learned_weights": training.learned_weights,
            "conventions": conventions,
            "persistent_projection_weight_scales": scales,
            "top_down_current_pA": float(f7_protocol["top_down_current_pA"]),
            "top_down_current_mode": TopDownCurrentMode(
                f7_protocol["top_down_current_mode"]
            ),
            "top_down_cue_lead_ms": float(f7_protocol["top_down_cue_lead_ms"]),
            "duration_ms": float(f7_protocol["duration_ms"]),
            "dt_ms": float(f7_protocol["dt_ms"]),
            "equilibration_ms": float(f7_protocol["equilibration_ms"]),
            "comparator_top_k_targets": int(comparator["target_count"]),
            "comparator_source_index": int(comparator["source_index"]),
            "record_relay_diagnostics": True,
            "brian": brian,
        }
        match = figure7_module.run_figure7_condition(
            condition=MatchCondition.MATCH, **figure7_common
        )
        mismatch = figure7_module.run_figure7_condition(
            condition=MatchCondition.MISMATCH, **figure7_common
        )
    finally:
        classic_sector.build_first_order_connected_sector = original_builder

    expected_topology = profile["projection036_topology"]
    topology_gates = {
        "three_figure6_figure7_builds_observed": len(topology_observations) == 3,
        "nonzero_edges_all_observed_builds": all(
            item["nonzero_edges"] == expected_topology["nonzero_edges"]
            for item in topology_observations
        ),
        "incoming_weight_sum_all_observed_builds": all(
            np.isclose(
                item["incoming_weight_sum_target40"],
                expected_topology["incoming_weight_sum_per_target"],
                rtol=0.0,
                atol=1e-12,
            )
            for item in topology_observations
        ),
        "projection025_source_scale": PROJECTION025_ID not in scales,
    }

    match_active = sorted(set(match.relay_spike_indices))
    mismatch_active = sorted(set(mismatch.relay_spike_indices))
    f7_gates = profile["figure7_gates"]
    figure7_identity = {
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
        == f7_gates["match_relay_active_indices"],
        "match_relay_events": len(match.relay_spike_times_ms)
        == f7_gates["match_relay_events"],
        "mismatch_relay_allowed_indices": set(mismatch_active).issubset(
            set(f7_gates["mismatch_relay_allowed_indices"])
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
        == f7_gates["match_nonspecific_events"],
        "mismatch_nonspecific_events": len(mismatch.nonspecific_spike_times_ms)
        == f7_gates["mismatch_nonspecific_events"],
    }

    f10_protocol = profile["figure10_protocol"]
    figure10_common = {
        "top_down_current_pA": float(f10_protocol["top_down_current_pA"]),
        "pre_match_duration_ms": float(f10_protocol["pre_match_duration_ms"]),
        "mismatch_duration_ms": float(f10_protocol["mismatch_duration_ms"]),
        "release_after_mismatch_ms": float(
            f10_protocol["release_after_mismatch_ms"]
        ),
        "learned_weights": training.learned_weights,
        "persistent_projection_weight_scales": scales,
        "persistent_projection_delays_ms": {},
        "comparator_top_k_targets": int(comparator["target_count"]),
        "comparator_source_index": int(comparator["source_index"]),
        "top_down_current_mode": f10_protocol["top_down_current_mode"],
        "conventions": conventions,
        "dt_ms": float(f10_protocol["dt_ms"]),
        "brian": brian,
    }
    intact = run_figure10_search_cycle_spread_condition(
        reset_pathway_enabled=True, **figure10_common
    )
    control = run_figure10_search_cycle_spread_condition(
        reset_pathway_enabled=False, **figure10_common
    )
    figure10_gates = {
        "projection036_topology_both_arms": bool(
            intact.projection036_nonzero_edges
            == control.projection036_nonzero_edges
            == expected_topology["nonzero_edges"]
            and np.isclose(
                intact.projection036_incoming_weight_sum,
                expected_topology["incoming_weight_sum_per_target"],
                rtol=0.0,
                atol=1e-12,
            )
            and np.isclose(
                control.projection036_incoming_weight_sum,
                expected_topology["incoming_weight_sum_per_target"],
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "identical_nonempty_pre_reset_state": (
            intact.pre_layer4_events > 0
            and intact.pre_layer4_events == control.pre_layer4_events
            and intact.pre_layer4_active_indices == control.pre_layer4_active_indices
        ),
        "alternatives_quiet_before_release": (
            intact.alternative_events_before_release == 0
            and control.alternative_events_before_release == 0
        ),
        "full_input_delivery": all(
            value > 0
            for result in (intact, control)
            for _, value in result.projection035_post_release_integral_pA_ms
        ),
        "intact_reset_chain": (
            intact.nonspecific_post_events > 0
            and intact.layer5_post_events > 0
            and intact.layer6i_post_events > 0
        ),
        "winner_suppression": intact.winner_post_events
        < control.winner_post_events,
        "alternative_latency": (
            intact.first_alternative_event_ms is not None
            and (
                control.first_alternative_event_ms is None
                or intact.first_alternative_event_ms
                < control.first_alternative_event_ms
            )
        ),
    }
    reproduced = (
        all(topology_gates.values())
        and all(figure6_gates.values())
        and all(figure7_gates.values())
        and all(figure10_gates.values())
    )
    print(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": registration["result_id"],
                "date": registration["date"],
                "status": "completed-projection025-source-control-joint-cross",
                "classification": registration["classification"],
                "registration": args.registration,
                "runtime_fingerprint": conventions.fingerprint,
                "persistent_projection_weight_scales": scales,
                "projection025_runtime_scale": 1.0,
                "projection036_topology_observations": topology_observations,
                "topology_and_source_gates": topology_gates,
                "figure6_gates": figure6_gates,
                "figure7_event_identity": figure7_identity,
                "figure7_gates": figure7_gates,
                "figure10_intact": asdict(intact),
                "figure10_disconnected_control": asdict(control),
                "figure10_gates": figure10_gates,
                "joint_first_order_gates_pass": reproduced,
                "parameter_selected": False,
                "original_smart_reproduced": False,
                "baseline_promoted": False,
            },
            sort_keys=False,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
