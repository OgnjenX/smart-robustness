"""Run the calibrated endpoint on the full V1-pulvinar-V2 Figure 16 holdout."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from run_joint_calibration_projection038_figure10_screen import _figure6_gates, _sha256

from smart_robustness import classic_sector
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import Figure6LearningProtocol, run_figure6_learning
from smart_robustness.validation.figure10_search_cycle_spread import (
    build_projection036_variance_sector,
)
from smart_robustness.validation.higher_order import (
    Figure16Protocol,
    assess_figure16_candidate,
    run_figure16_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--standalone-directory", required=True)
    args = parser.parse_args()
    registration = yaml.safe_load(Path(args.registration).read_text())
    authorization = Path(registration["authorization"])
    if _sha256(authorization) != registration["authorization_sha256"]:
        raise ValueError("Figure-16 authorization differs from registration")
    standalone_directory = Path(args.standalone_directory).resolve()
    if standalone_directory.exists():
        raise FileExistsError(standalone_directory)

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
        raise ValueError("runtime differs from registered Figure-16 endpoint")

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
    finally:
        classic_sector.build_first_order_connected_sector = original_builder
    figure6_gates = _figure6_gates(training, profile)
    if not all(figure6_gates.values()):
        raise RuntimeError(f"fresh Figure 6 prerequisite failed: {figure6_gates}")

    protocol_values = registration["figure16_protocol"]
    protocol = Figure16Protocol(
        prestimulus_ms=float(protocol_values["prestimulus_ms"]),
        recording_ms=float(protocol_values["recording_ms"]),
        inter_area_delay_ms=float(protocol_values["inter_area_delay_ms"]),
        integration_dt_ms=float(protocol_values["integration_dt_ms"]),
        recording_sample_ms=float(protocol_values["recording_sample_ms"]),
        frequency_bands_hz=tuple(
            tuple(float(value) for value in band)
            for band in protocol_values["frequency_bands_hz"]
        ),
    )
    candidate = run_figure16_candidate(
        learned_weights=training.learned_weights,
        persistent_projection_weight_scales=scales,
        projection036_variance_topology=True,
        protocol=protocol,
        geometry_seed=int(protocol_values["geometry_seed_v1"]),
        conventions=conventions,
        cpp_standalone_directory=standalone_directory,
        brian=brian,
    )
    assessment = assess_figure16_candidate(candidate)
    frequency = assessment.frequency_assessment
    correlations = [
        {
            "band_hz": list(item.band_hz),
            "peak_absolute_normalized": item.peak_absolute_normalized,
            "peak_lag_ms": item.peak_lag_ms,
        }
        for item in assessment.cross_correlations
    ]
    print(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": registration["result_id"],
                "date": registration["date"],
                "status": "completed-calibrated-figure16-holdout",
                "classification": registration["classification"],
                "registration": args.registration,
                "runtime_fingerprint": conventions.fingerprint,
                "global_projection_weight_scales": scales,
                "projection036_spread_convention": "variance",
                "figure6_gates": figure6_gates,
                "protocol": protocol_values,
                "learned_state_provenance": candidate.learned_state_provenance,
                "sample_count": len(candidate.sample_times_ms),
                "sample_time_range_ms": [
                    candidate.sample_times_ms[0],
                    candidate.sample_times_ms[-1],
                ],
                "v1_field_shape": list(candidate.v1_field.potential_uV.shape),
                "v2_field_shape": list(candidate.v2_field.potential_uV.shape),
                "v1_all_finite": bool(np.all(np.isfinite(candidate.v1_field.potential_uV))),
                "v2_all_finite": bool(np.all(np.isfinite(candidate.v2_field.potential_uV))),
                "cross_correlations": correlations,
                "strongest_band_hz": list(frequency.strongest_band_hz),
                "lower_frequency_stronger_than_gamma": (
                    frequency.lower_frequency_stronger_than_gamma
                ),
                "figure16_holdout_pass": (
                    bool(np.all(np.isfinite(candidate.v1_field.potential_uV)))
                    and bool(np.all(np.isfinite(candidate.v2_field.potential_uV)))
                    and frequency.lower_frequency_stronger_than_gamma
                ),
                "parameter_selected": False,
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
