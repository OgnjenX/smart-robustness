"""Map Figure 15 frequency across every pair in the stimulated layer-4 row."""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import yaml
from run_joint_calibration_projection038_figure10_screen import _figure6_gates, _sha256

from smart_robustness import classic_sector
from smart_robustness.analysis.figure15 import figure15_layer4_synchrony
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
    registration_path = Path(args.registration)
    registration = yaml.safe_load(registration_path.read_text())
    for path_key, hash_key in (
        ("authorization", "authorization_sha256"),
        ("source_identifiability_audit", "source_identifiability_audit_sha256"),
        ("original_holdout_registration", "original_holdout_registration_sha256"),
        ("original_holdout_result", "original_holdout_result_sha256"),
        ("profile", "profile_sha256"),
        ("network_harness", "network_harness_sha256"),
        ("analysis_harness", "analysis_harness_sha256"),
    ):
        if _sha256(Path(registration[path_key])) != registration[hash_key]:
            raise ValueError(f"{path_key} differs from registration")

    profile = yaml.safe_load(Path(registration["profile"]).read_text())
    training_profile = yaml.safe_load(Path(profile["training_profile"]).read_text())
    base_profile = yaml.safe_load(Path(training_profile["base_profile"]).read_text())
    fixed_endpoint = registration["fixed_endpoint"]
    runtime_overrides = {
        **training_profile["runtime_overrides"],
        **profile["runtime_overrides"],
        "nonspecific_dendritic_calcium_density_scale": float(fixed_endpoint["nonspecific_t_scale"]),
    }
    conventions = replace(
        runtime_conventions_for_candidate(base_profile["candidate"]),
        **runtime_overrides,
    )
    if conventions.fingerprint != registration["runtime_fingerprint"]:
        raise ValueError("runtime differs from registered generation-one endpoint")

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
            raise ValueError(f"{projection_id} unexpectedly present before audit")
        scales[str(projection_id)] = float(scale)

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    protocol = registration["protocol"]
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

        reference_pair = tuple(int(value) for value in protocol["reference_pair"])
        condition = run_figure15_condition(
            top_down_current_pA=float(figure7_protocol["top_down_current_pA"]),
            learned_weights=training.learned_weights,
            conventions=conventions,
            persistent_projection_weight_scales=scales,
            comparator_top_k_targets=int(comparator["target_count"]),
            comparator_source_index=int(comparator["source_index"]),
            top_down_current_mode=figure7_protocol["top_down_current_mode"],
            top_down_cue_lead_ms=float(figure7_protocol["top_down_cue_lead_ms"]),
            equilibration_ms=float(figure7_protocol["equilibration_ms"]),
            first_cell_index=reference_pair[0],
            second_cell_index=reference_pair[1],
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

    network = condition.network_result
    pair_results = []
    for first, second in combinations(protocol["stimulated_row_indices"], 2):
        synchrony = figure15_layer4_synchrony(
            network.layer4_spike_indices,
            network.layer4_spike_times_ms,
            first_cell_index=int(first),
            second_cell_index=int(second),
            duration_ms=float(protocol["duration_ms"]),
            bin_ms=float(protocol["histogram_bin_ms"]),
            max_lag_ms=float(protocol["display_max_lag_ms"]),
        )
        pair_results.append(
            {
                "pair": [int(first), int(second)],
                "lattice_separation": int(second) - int(first),
                "spike_counts": [
                    synchrony.first_spike_count,
                    synchrony.second_spike_count,
                ],
                "gamma_peak_hz": synchrony.gamma_peak_hz,
                "within_39_49_hz": bool(39.0 <= synchrony.gamma_peak_hz <= 49.0),
            }
        )

    reference = condition.synchrony
    reference_reproduced = (
        reference.first_spike_count == registration["reproduction_gates"]["first_spikes"]
        and reference.second_spike_count == registration["reproduction_gates"]["second_spikes"]
        and len(network.layer4_spike_times_ms)
        == registration["reproduction_gates"]["all_layer4_spikes"]
        and abs(reference.gamma_peak_hz - registration["reproduction_gates"]["gamma_peak_hz"])
        <= 1e-12
    )
    if not reference_reproduced:
        raise RuntimeError("immutable Figure 15 reference result did not reproduce")

    print(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": registration["result_id"],
                "date": registration["date"],
                "status": "completed-figure15-stimulated-pair-distribution-audit",
                "classification": registration["classification"],
                "registration": args.registration,
                "runtime_fingerprint": conventions.fingerprint,
                "global_projection_weight_scales": scales,
                "figure6_gates": figure6_gates,
                "reference_result_reproduced": reference_reproduced,
                "protocol": protocol,
                "pair_results": pair_results,
                "raw_layer4_spike_indices": [int(value) for value in network.layer4_spike_indices],
                "raw_layer4_spike_times_ms": [
                    float(value) for value in network.layer4_spike_times_ms
                ],
                "pair_selected": False,
                "parameter_selected": False,
                "generation_two_opened": False,
                "original_holdout_reclassified": False,
                "original_smart_reproduced": False,
                "baseline_frozen": False,
            },
            sort_keys=False,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
