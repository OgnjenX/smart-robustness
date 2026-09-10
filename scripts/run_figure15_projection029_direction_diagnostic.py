"""Run the preregistered projection-029 Figure 15 conductance diagnostic."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import yaml
from run_figure15_projection037_direction_diagnostic import (
    _isi_summary,
    _load_generation_one,
)
from run_joint_calibration_projection038_figure10_screen import _figure6_gates, _sha256

from smart_robustness import classic_sector
from smart_robustness.analysis.figure15_sensitivity import (
    direct_cross_spectrum_gamma_peak,
)
from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import Figure6LearningProtocol, run_figure6_learning
from smart_robustness.validation.figure7 import run_figure7_condition
from smart_robustness.validation.figure10_search_cycle_spread import (
    build_projection036_variance_sector,
)

PROJECTION029_ID = "modeldb112923.projection.029"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    registration = yaml.safe_load(Path(args.registration).read_text())
    for path_key, hash_key in (
        ("authorization", "authorization_sha256"),
        ("conductance_scaling_implementation", "conductance_scaling_implementation_sha256"),
        ("source_audit", "source_audit_sha256"),
        ("original_holdout_registration", "original_holdout_registration_sha256"),
        ("figure7_runtime", "figure7_runtime_sha256"),
        ("script", "script_sha256"),
        ("analysis", "analysis_sha256"),
    ):
        if _sha256(Path(registration[path_key])) != registration[hash_key]:
            raise ValueError(f"{path_key} differs from registration")

    holdout, profile, training_profile, base_profile = _load_generation_one(registration)
    runtime_overrides = {
        **training_profile["runtime_overrides"],
        **profile["runtime_overrides"],
        "nonspecific_dendritic_calcium_density_scale": float(
            holdout["fixed_endpoint"]["nonspecific_t_scale"]
        ),
    }
    conventions = replace(
        runtime_conventions_for_candidate(base_profile["candidate"]),
        **runtime_overrides,
    )
    if conventions.fingerprint != holdout["runtime_fingerprint"]:
        raise ValueError("runtime differs from immutable generation-one endpoint")

    scales = {
        str(key): float(value)
        for key, value in training_profile["projection_weight_scales"].items()
    }
    for item in profile["persistent_projection_scales"]:
        scales[str(item["projection_id"])] = float(item["scale"])
    for projection_id, scale in holdout["additional_global_scales"].items():
        scales[str(projection_id)] = float(scale)
    if PROJECTION029_ID in scales:
        raise ValueError("electrical projection 029 entered chemical weight scales")

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    diagnostic = registration["diagnostic"]
    comparator = profile["comparator"]
    figure7_protocol = profile["figure7_protocol"]
    original_builder = classic_sector.build_first_order_connected_sector
    outcomes = []
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

        for scale in diagnostic["projection029_conductance_scales"]:
            result = run_figure7_condition(
                condition=MatchCondition.MATCH,
                top_down_current_pA=float(figure7_protocol["top_down_current_pA"]),
                learned_weights=training.learned_weights,
                conventions=conventions,
                persistent_projection_weight_scales=scales,
                persistent_projection_conductance_scales={
                    PROJECTION029_ID: float(scale)
                },
                comparator_top_k_targets=int(comparator["target_count"]),
                comparator_source_index=int(comparator["source_index"]),
                top_down_current_mode=figure7_protocol["top_down_current_mode"],
                top_down_cue_lead_ms=float(figure7_protocol["top_down_cue_lead_ms"]),
                equilibration_ms=float(figure7_protocol["equilibration_ms"]),
                duration_ms=float(diagnostic["duration_ms"]),
                dt_ms=float(diagnostic["dt_ms"]),
                record_interneuron_spikes=True,
                brian=brian,
            )
            indices = np.asarray(result.layer4_spike_indices, dtype=int)
            times_ms = np.asarray(result.layer4_spike_times_ms, dtype=float)
            inhibitory_indices = np.asarray(result.interneuron_spike_indices, dtype=int)
            inhibitory_times_ms = np.asarray(result.interneuron_spike_times_ms, dtype=float)
            first_index, second_index = (int(value) for value in diagnostic["pair"])
            first_times = times_ms[indices == first_index]
            second_times = times_ms[indices == second_index]
            peak = direct_cross_spectrum_gamma_peak(
                first_times,
                second_times,
                duration_ms=float(diagnostic["duration_ms"]),
                bin_ms=float(diagnostic["bin_ms"]),
            )
            outcomes.append(
                {
                    "projection029_conductance_scale": float(scale),
                    "first_cell_spikes": int(first_times.size),
                    "second_cell_spikes": int(second_times.size),
                    "all_layer4_excitatory_spikes": int(times_ms.size),
                    "layer4_inhibitory_events": [
                        [int(index), float(time)]
                        for index, time in zip(
                            inhibitory_indices, inhibitory_times_ms, strict=True
                        )
                    ],
                    "direct_cross_spectrum": asdict(peak),
                    "first_cell_isi": _isi_summary(first_times),
                    "second_cell_isi": _isi_summary(second_times),
                    "raw_pair_spike_times_ms": {
                        str(first_index): [float(value) for value in first_times],
                        str(second_index): [float(value) for value in second_times],
                    },
                }
            )
    finally:
        classic_sector.build_first_order_connected_sector = original_builder

    print(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": registration["result_id"],
                "date": registration["date"],
                "status": "completed-projection029-direction-diagnostic",
                "classification": registration["classification"],
                "registration": args.registration,
                "runtime_fingerprint": conventions.fingerprint,
                "generation_one_projection_weight_scales": scales,
                "figure6_gates": figure6_gates,
                "learned_state_reused_across_arms": True,
                "outcomes": outcomes,
                "selection_performed": False,
                "generation_two_opened": False,
                "neuron_models_consulted": False,
                "modern_anatomy_consulted": False,
                "original_smart_reproduced": False,
                "baseline_frozen": False,
            },
            sort_keys=False,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
