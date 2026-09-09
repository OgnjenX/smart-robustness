"""Run the fixed calibrated endpoint on the Figure 15 local synchrony holdout."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import yaml
from run_joint_calibration_projection038_figure10_screen import _figure6_gates, _sha256

from smart_robustness import classic_sector
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import Figure6LearningProtocol, run_figure6_learning
from smart_robustness.validation.figure10_search_cycle_spread import (
    build_projection036_variance_sector,
)
from smart_robustness.validation.figure15 import run_figure15_condition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    registration = yaml.safe_load(Path(args.registration).read_text())
    authorization = Path(registration["authorization"])
    if _sha256(authorization) != registration["authorization_sha256"]:
        raise ValueError("Figure-15 authorization differs from registration")

    profile = yaml.safe_load(Path(registration["profile"]).read_text())
    training_profile = yaml.safe_load(Path(profile["training_profile"]).read_text())
    base_profile = yaml.safe_load(Path(training_profile["base_profile"]).read_text())
    fixed_endpoint = registration["fixed_endpoint"]
    runtime_overrides = {
        **training_profile["runtime_overrides"],
        **profile["runtime_overrides"],
        "nonspecific_dendritic_calcium_density_scale": float(
            fixed_endpoint["nonspecific_t_scale"]
        ),
    }
    conventions = replace(
        runtime_conventions_for_candidate(base_profile["candidate"]),
        **runtime_overrides,
    )
    if conventions.fingerprint != registration["runtime_fingerprint"]:
        raise ValueError("runtime differs from registered Figure-15 endpoint")

    scales = {
        str(key): float(value)
        for key, value in training_profile["projection_weight_scales"].items()
    }
    for item in profile["persistent_projection_scales"]:
        projection_id = str(item["projection_id"])
        if projection_id in scales:
            raise ValueError(f"{projection_id} overlaps an existing scale")
        scales[projection_id] = float(item["scale"])
    for projection_id, scale in registration["additional_global_scales"].items():
        if projection_id in scales:
            raise ValueError(f"{projection_id} unexpectedly present before holdout")
        scales[str(projection_id)] = float(scale)

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    protocol = registration["figure15_protocol"]
    comparator = profile["comparator"]
    figure7_protocol = profile["figure7_protocol"]
    original_builder = classic_sector.build_first_order_connected_sector
    try:
        classic_sector.build_first_order_connected_sector = build_projection036_variance_sector
        training = run_figure6_learning(
            conventions=conventions,
            protocol=Figure6LearningProtocol(
                monitored_populations=tuple(training_profile["monitored_populations"])
            ),
            projection_weight_scales=scales,
            brian=brian,
        )
        figure6_gates = _figure6_gates(training, profile)
        if not all(figure6_gates.values()):
            raise RuntimeError(f"fresh Figure 6 prerequisite failed: {figure6_gates}")

        result = run_figure15_condition(
            top_down_current_pA=float(figure7_protocol["top_down_current_pA"]),
            learned_weights=training.learned_weights,
            conventions=conventions,
            persistent_projection_weight_scales=scales,
            comparator_top_k_targets=int(comparator["target_count"]),
            comparator_source_index=int(comparator["source_index"]),
            top_down_current_mode=figure7_protocol["top_down_current_mode"],
            top_down_cue_lead_ms=float(figure7_protocol["top_down_cue_lead_ms"]),
            equilibration_ms=float(figure7_protocol["equilibration_ms"]),
            first_cell_index=int(protocol["first_cell_index"]),
            second_cell_index=int(protocol["second_cell_index"]),
            duration_ms=float(protocol["duration_ms"]),
            dt_ms=float(protocol["dt_ms"]),
            histogram_bin_ms=float(protocol["histogram_bin_ms"]),
            max_lag_ms=float(protocol["display_max_lag_ms"]),
            target_hz=float(protocol["target_hz"]),
            tolerance_hz=float(protocol["tolerance_hz"]),
            brian=brian,
        )
    finally:
        classic_sector.build_first_order_connected_sector = original_builder

    network = result.network_result
    synchrony = result.synchrony
    assessment = result.assessment
    print(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": registration["result_id"],
                "date": registration["date"],
                "status": "completed-calibrated-figure15-holdout",
                "classification": registration["classification"],
                "registration": args.registration,
                "runtime_fingerprint": conventions.fingerprint,
                "global_projection_weight_scales": scales,
                "figure6_gates": figure6_gates,
                "protocol": protocol,
                "result": {
                    "first_cell_spikes": synchrony.first_spike_count,
                    "second_cell_spikes": synchrony.second_spike_count,
                    "all_layer4_spikes": len(network.layer4_spike_times_ms),
                    "gamma_peak_hz": synchrony.gamma_peak_hz,
                    "relay_events": len(network.relay_spike_times_ms),
                    "trn_events": len(network.trn_spike_times_ms),
                    "nonspecific_events": len(network.nonspecific_spike_times_ms),
                },
                "gates": {
                    "both_cells_have_at_least_two_spikes": assessment.enough_spikes,
                    "peak_within_target_tolerance": (
                        abs(assessment.gamma_peak_hz - assessment.target_hz)
                        <= assessment.tolerance_hz
                    ),
                },
                "figure15_holdout_pass": assessment.reproduced,
                "parameter_selected": False,
                "higher_order_consulted": False,
                "neuron_models_consulted": False,
                "modern_anatomy_consulted": False,
                "original_smart_reproduced": False,
                "baseline_promoted": False,
                "baseline_frozen": False,
            },
            sort_keys=False,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
