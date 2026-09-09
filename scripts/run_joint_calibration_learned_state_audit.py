"""Audit learned-state changes caused by globally active Figure 10 scales."""

from __future__ import annotations

import argparse
import hashlib
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


def _array_sha256(values: tuple[float, ...]) -> str:
    array = np.asarray(values, dtype="<f8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _map_summary(result) -> dict[str, dict[str, float]]:
    maps = {
        "bottom_up": result.bottom_up,
        "top_down_wide": result.top_down_wide,
        "top_down_narrow": result.top_down_narrow,
        "top_down_combined": result.top_down_combined,
    }
    return {
        name: {
            "horizontal_mean": item.horizontal_mean,
            "vertical_mean": item.vertical_mean,
            "horizontal_orientation_contrast": item.horizontal_orientation_contrast,
        }
        for name, item in maps.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    registration = yaml.safe_load(Path(args.registration).read_text())
    authorization = Path(registration["authorization"])
    if _sha256(authorization) != registration["authorization_sha256"]:
        raise ValueError("learned-state audit authorization differs from registration")

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
        raise ValueError("runtime differs from registered learned-state audit")

    base_scales = {
        str(key): float(value)
        for key, value in training_profile["projection_weight_scales"].items()
    }
    for item in profile["persistent_projection_scales"]:
        projection_id = str(item["projection_id"])
        if projection_id in base_scales:
            raise ValueError(f"{projection_id} overlaps an existing scale")
        base_scales[projection_id] = float(item["scale"])
    endpoint_scales = dict(base_scales)
    for projection_id, value in registration["additional_global_scales"].items():
        if projection_id in endpoint_scales:
            raise ValueError(f"{projection_id} unexpectedly present in base scales")
        endpoint_scales[str(projection_id)] = float(value)

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    protocol = Figure6LearningProtocol(
        monitored_populations=tuple(training_profile["monitored_populations"])
    )
    original_builder = classic_sector.build_first_order_connected_sector
    try:
        classic_sector.build_first_order_connected_sector = build_projection036_variance_sector
        base = run_figure6_learning(
            conventions=conventions,
            protocol=protocol,
            projection_weight_scales=base_scales,
            brian=brian,
        )
        endpoint = run_figure6_learning(
            conventions=conventions,
            protocol=protocol,
            projection_weight_scales=endpoint_scales,
            brian=brian,
        )
    finally:
        classic_sector.build_first_order_connected_sector = original_builder

    if set(base.learned_weights) != set(endpoint.learned_weights):
        raise RuntimeError("learned projection sets differ across audit arms")
    differences = []
    for projection_id in sorted(base.learned_weights):
        base_values = np.asarray(base.learned_weights[projection_id], dtype=float)
        endpoint_values = np.asarray(endpoint.learned_weights[projection_id], dtype=float)
        if base_values.shape != endpoint_values.shape:
            raise RuntimeError(f"learned shape differs for {projection_id}")
        delta = endpoint_values - base_values
        differences.append(
            {
                "projection_id": projection_id,
                "value_count": int(base_values.size),
                "base_sha256_float64_le": _array_sha256(base.learned_weights[projection_id]),
                "endpoint_sha256_float64_le": _array_sha256(
                    endpoint.learned_weights[projection_id]
                ),
                "exact_changed_values": int(np.count_nonzero(delta)),
                "max_abs_difference": float(np.max(np.abs(delta))),
                "l1_difference": float(np.sum(np.abs(delta))),
                "l2_difference": float(np.linalg.norm(delta)),
            }
        )

    print(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": registration["result_id"],
                "date": registration["date"],
                "status": "completed-learned-state-audit",
                "classification": registration["classification"],
                "registration": args.registration,
                "runtime_fingerprint": conventions.fingerprint,
                "base_training_scales": base_scales,
                "endpoint_training_scales": endpoint_scales,
                "base_figure6_gates": _figure6_gates(base, profile),
                "endpoint_figure6_gates": _figure6_gates(endpoint, profile),
                "base_population_spikes": base.result.population_spikes,
                "endpoint_population_spikes": endpoint.result.population_spikes,
                "base_maps": _map_summary(base.result),
                "endpoint_maps": _map_summary(endpoint.result),
                "learned_weight_differences": differences,
                "any_exact_learned_weight_difference": any(
                    item["exact_changed_values"] > 0 for item in differences
                ),
                "figure7_consulted": False,
                "figure10_consulted": False,
                "holdouts_consulted": False,
                "parameter_selected": False,
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
