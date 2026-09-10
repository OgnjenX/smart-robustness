"""Replay the fixed Figure 15 endpoint and report a predeclared analysis family."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import yaml
from run_joint_calibration_projection038_figure10_screen import _figure6_gates, _sha256

from smart_robustness import classic_sector
from smart_robustness.analysis.figure15_sensitivity import (
    figure15_analysis_sensitivity,
)
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
        raise ValueError("Figure-15 audit authorization differs from registration")

    original_registration = yaml.safe_load(
        Path(registration["original_holdout_registration"]).read_text()
    )
    profile = yaml.safe_load(Path(original_registration["profile"]).read_text())
    training_profile = yaml.safe_load(Path(profile["training_profile"]).read_text())
    base_profile = yaml.safe_load(Path(training_profile["base_profile"]).read_text())
    fixed_endpoint = original_registration["fixed_endpoint"]
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
    if conventions.fingerprint != original_registration["runtime_fingerprint"]:
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
    for projection_id, scale in original_registration[
        "additional_global_scales"
    ].items():
        if projection_id in scales:
            raise ValueError(f"{projection_id} unexpectedly present before holdout")
        scales[str(projection_id)] = float(scale)

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    protocol = original_registration["figure15_protocol"]
    comparator = profile["comparator"]
    figure7_protocol = profile["figure7_protocol"]
    original_builder = classic_sector.build_first_order_connected_sector
    try:
        classic_sector.build_first_order_connected_sector = (
            build_projection036_variance_sector
        )
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
    indices = np.asarray(network.layer4_spike_indices, dtype=int)
    times_ms = np.asarray(network.layer4_spike_times_ms, dtype=float)
    first_index = int(protocol["first_cell_index"])
    second_index = int(protocol["second_cell_index"])
    first_times = times_ms[indices == first_index]
    second_times = times_ms[indices == second_index]
    method_peaks = figure15_analysis_sensitivity(
        first_times,
        second_times,
        duration_ms=float(protocol["duration_ms"]),
        bin_ms=float(protocol["histogram_bin_ms"]),
        display_max_lag_ms=float(protocol["display_max_lag_ms"]),
        methods_hamming_window_ms=float(
            registration["analysis_family"]["methods_hamming_window_ms"]
        ),
    )
    expected = registration["deterministic_reproduction_gate"]
    reproduction_gates = {
        "first_cell_spikes": int(first_times.size) == int(expected["first_cell_spikes"]),
        "second_cell_spikes": int(second_times.size)
        == int(expected["second_cell_spikes"]),
        "all_layer4_spikes": int(times_ms.size) == int(expected["all_layer4_spikes"]),
        "registered_peak_hz": bool(
            np.isclose(
                result.synchrony.gamma_peak_hz,
                float(expected["registered_peak_hz"]),
                rtol=0.0,
                atol=float(expected["absolute_tolerance"]),
            )
        ),
    }
    print(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": registration["result_id"],
                "date": registration["date"],
                "status": "completed-figure15-analysis-sensitivity-audit",
                "classification": registration["classification"],
                "registration": args.registration,
                "runtime_fingerprint": conventions.fingerprint,
                "global_projection_weight_scales": scales,
                "figure6_gates": figure6_gates,
                "deterministic_reproduction_gates": reproduction_gates,
                "deterministic_reproduction_pass": all(reproduction_gates.values()),
                "fixed_pair": [first_index, second_index],
                "raw_pair_spike_times_ms": {
                    str(first_index): [float(value) for value in first_times],
                    str(second_index): [float(value) for value in second_times],
                },
                "registered_analysis_peak_hz": result.synchrony.gamma_peak_hz,
                "analysis_family": [asdict(outcome) for outcome in method_peaks],
                "selection_performed": False,
                "parameter_changed": False,
                "original_smart_reproduced": False,
                "baseline_frozen": False,
            },
            sort_keys=False,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
