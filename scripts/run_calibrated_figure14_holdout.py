"""Run the fixed calibrated first-order endpoint on the Figure 14 holdout."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import yaml
from run_joint_calibration_projection038_figure10_screen import _figure6_gates, _sha256

from smart_robustness import classic_sector
from smart_robustness.analysis.figure14 import assess_figure14_spectra
from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import Figure6LearningProtocol, run_figure6_learning
from smart_robustness.validation.figure10_search_cycle_spread import (
    build_projection036_variance_sector,
)
from smart_robustness.validation.figure14 import run_figure14_condition


def _condition_summary(result) -> dict[str, object]:
    network = result.network_result
    spectrum = result.spectrum
    return {
        "cortical_spikes": len(network.v1_cortical_spike_times_ms),
        "relay_events": len(network.relay_spike_times_ms),
        "relay_active_indices": sorted(set(network.relay_spike_indices)),
        "trn_events": len(network.trn_spike_times_ms),
        "nonspecific_events": len(network.nonspecific_spike_times_ms),
        "dominant_frequency_hz": spectrum.dominant_frequency_hz,
        "low_power_2_8_hz": spectrum.low_power,
        "middle_caption_power_8_20_hz": spectrum.middle_caption_power,
        "middle_methods_power_8_10_hz": spectrum.middle_methods_power,
        "gamma_power_20_70_hz": spectrum.gamma_power,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    registration = yaml.safe_load(Path(args.registration).read_text())
    authorization = Path(registration["authorization"])
    if _sha256(authorization) != registration["authorization_sha256"]:
        raise ValueError("Figure-14 authorization differs from registration")

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
        raise ValueError("runtime differs from registered Figure-14 endpoint")

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
    protocol = registration["figure14_protocol"]
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

        common = {
            "top_down_current_pA": float(figure7_protocol["top_down_current_pA"]),
            "learned_weights": training.learned_weights,
            "conventions": conventions,
            "persistent_projection_weight_scales": scales,
            "comparator_top_k_targets": int(comparator["target_count"]),
            "comparator_source_index": int(comparator["source_index"]),
            "top_down_current_mode": figure7_protocol["top_down_current_mode"],
            "top_down_cue_lead_ms": float(figure7_protocol["top_down_cue_lead_ms"]),
            "equilibration_ms": float(figure7_protocol["equilibration_ms"]),
            "duration_ms": float(protocol["duration_ms"]),
            "dt_ms": float(protocol["dt_ms"]),
            "histogram_bin_ms": float(protocol["histogram_bin_ms"]),
            "hamming_window_ms": float(protocol["hamming_window_ms"]),
            "brian": brian,
        }
        match = run_figure14_condition(condition=MatchCondition.MATCH, **common)
        mismatch = run_figure14_condition(condition=MatchCondition.MISMATCH, **common)
    finally:
        classic_sector.build_first_order_connected_sector = original_builder

    assessment = assess_figure14_spectra(match.spectrum, mismatch.spectrum)
    print(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": registration["result_id"],
                "date": registration["date"],
                "status": "completed-calibrated-figure14-holdout",
                "classification": registration["classification"],
                "registration": args.registration,
                "runtime_fingerprint": conventions.fingerprint,
                "global_projection_weight_scales": scales,
                "figure6_gates": figure6_gates,
                "protocol": protocol,
                "match": _condition_summary(match),
                "mismatch": _condition_summary(mismatch),
                "gates": {
                    "match_gamma_dominant": assessment.match_gamma_dominant,
                    "mismatch_lower_frequency_dominant": (
                        assessment.mismatch_lower_frequency_dominant
                    ),
                    "mismatch_gamma_reduced": assessment.mismatch_gamma_reduced,
                },
                "figure14_holdout_pass": assessment.reproduced,
                "parameter_selected": False,
                "figure15_consulted": False,
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
