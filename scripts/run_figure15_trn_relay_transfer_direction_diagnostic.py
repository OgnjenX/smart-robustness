"""Run the preregistered Figure 15 joint TRN-to-relay transfer diagnostic."""

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
from smart_robustness.analysis.figure15_phase import (
    population_gamma_spectrum,
    population_peak_lag,
)
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

TRN_RELAY_PROJECTION_IDS = (
    "modeldb112923.projection.000",
    "modeldb112923.projection.001",
    "modeldb112923.projection.004",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    registration = yaml.safe_load(Path(args.registration).read_text())
    for path_key, hash_key in (
        ("authorization", "authorization_sha256"),
        ("source_audit", "source_audit_sha256"),
        ("original_holdout_registration", "original_holdout_registration_sha256"),
        ("figure7_runtime", "figure7_runtime_sha256"),
        ("phase_analysis", "phase_analysis_sha256"),
        ("pair_analysis", "pair_analysis_sha256"),
        ("script", "script_sha256"),
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
    expected = registration["fixed_generation_one"]["trn_to_relay_scales"]
    for projection_id in TRN_RELAY_PROJECTION_IDS:
        if scales[projection_id] != float(expected[projection_id]):
            raise ValueError(f"{projection_id} differs from registered endpoint")

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

        for factor in diagnostic["common_factors"]:
            arm_scales = dict(scales)
            for projection_id in TRN_RELAY_PROJECTION_IDS:
                arm_scales[projection_id] *= float(factor)
            result = run_figure7_condition(
                condition=MatchCondition.MATCH,
                top_down_current_pA=float(figure7_protocol["top_down_current_pA"]),
                learned_weights=training.learned_weights,
                conventions=conventions,
                persistent_projection_weight_scales=arm_scales,
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
            layer4_indices = np.asarray(result.layer4_spike_indices, dtype=int)
            layer4_times = np.asarray(result.layer4_spike_times_ms, dtype=float)
            relay_indices = np.asarray(result.relay_spike_indices, dtype=int)
            relay_times = np.asarray(result.relay_spike_times_ms, dtype=float)
            trn_indices = np.asarray(result.trn_spike_indices, dtype=int)
            trn_times = np.asarray(result.trn_spike_times_ms, dtype=float)
            inhibitory_indices = np.asarray(result.interneuron_spike_indices, dtype=int)
            inhibitory_times = np.asarray(result.interneuron_spike_times_ms, dtype=float)
            first_index, second_index = (int(value) for value in diagnostic["pair"])
            first_times = layer4_times[layer4_indices == first_index]
            second_times = layer4_times[layer4_indices == second_index]
            duration_ms = float(diagnostic["duration_ms"])
            bin_ms = float(diagnostic["bin_ms"])
            outcomes.append(
                {
                    "common_factor": float(factor),
                    "effective_scales": {
                        projection_id: arm_scales[projection_id]
                        for projection_id in TRN_RELAY_PROJECTION_IDS
                    },
                    "event_counts": {
                        "thalamic_relay": int(relay_times.size),
                        "trn": int(trn_times.size),
                        "layer4_excitatory": int(layer4_times.size),
                        "layer4_inhibitory": int(inhibitory_times.size),
                    },
                    "active_cell_counts": {
                        "thalamic_relay": int(np.unique(relay_indices).size),
                        "trn": int(np.unique(trn_indices).size),
                        "layer4_excitatory": int(np.unique(layer4_indices).size),
                        "layer4_inhibitory": int(np.unique(inhibitory_indices).size),
                    },
                    "population_spectra": {
                        "thalamic_relay": asdict(
                            population_gamma_spectrum(
                                relay_times,
                                spike_indices=relay_indices,
                                duration_ms=duration_ms,
                                bin_ms=bin_ms,
                            )
                        ),
                        "trn": asdict(
                            population_gamma_spectrum(
                                trn_times,
                                spike_indices=trn_indices,
                                duration_ms=duration_ms,
                                bin_ms=bin_ms,
                            )
                        ),
                        "layer4_excitatory": asdict(
                            population_gamma_spectrum(
                                layer4_times,
                                spike_indices=layer4_indices,
                                duration_ms=duration_ms,
                                bin_ms=bin_ms,
                            )
                        ),
                    },
                    "relay_to_layer4_excitatory_lag": asdict(
                        population_peak_lag(
                            relay_times,
                            layer4_times,
                            duration_ms=duration_ms,
                            bin_ms=bin_ms,
                        )
                    ),
                    "fixed_pair_direct_cross_spectrum": asdict(
                        direct_cross_spectrum_gamma_peak(
                            first_times,
                            second_times,
                            duration_ms=duration_ms,
                            bin_ms=bin_ms,
                        )
                    ),
                    "first_cell_isi": _isi_summary(first_times),
                    "second_cell_isi": _isi_summary(second_times),
                    "raw_pair_spike_times_ms": {
                        str(first_index): [float(value) for value in first_times],
                        str(second_index): [float(value) for value in second_times],
                    },
                    "raw_relay_events": [
                        [int(index), float(time)]
                        for index, time in zip(relay_indices, relay_times, strict=True)
                    ],
                    "raw_layer4_inhibitory_events": [
                        [int(index), float(time)]
                        for index, time in zip(inhibitory_indices, inhibitory_times, strict=True)
                    ],
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
                "status": "completed-trn-relay-transfer-direction-diagnostic",
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
