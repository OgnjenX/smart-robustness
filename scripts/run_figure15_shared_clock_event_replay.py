"""Replay Figure 15 once and archive event streams for shared-clock diagnosis."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import yaml
from run_joint_calibration_projection038_figure10_screen import _figure6_gates, _sha256

from smart_robustness import classic_sector
from smart_robustness.analysis.figure15_phase import (
    population_gamma_spectrum,
    population_peak_lag,
)
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import Figure6LearningProtocol, run_figure6_learning
from smart_robustness.validation.figure10_search_cycle_spread import (
    build_projection036_variance_sector,
)
from smart_robustness.validation.figure15 import run_figure15_condition


def _events(indices: tuple[int, ...], times: tuple[float, ...]) -> dict[str, list]:
    return {
        "indices": [int(value) for value in indices],
        "times_ms": [float(value) for value in times],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    registration = yaml.safe_load(Path(args.registration).read_text())
    for path_key, hash_key in (
        ("authorization", "authorization_sha256"),
        ("source_audit", "source_audit_sha256"),
        ("prior_event_result", "prior_event_result_sha256"),
        ("profile", "profile_sha256"),
        ("network_harness", "network_harness_sha256"),
        ("analysis", "analysis_sha256"),
    ):
        if _sha256(Path(registration[path_key])) != registration[hash_key]:
            raise ValueError(f"{path_key} differs from registration")

    prior = yaml.safe_load(Path(registration["prior_event_result"]).read_text())
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
            raise ValueError(f"{projection_id} unexpectedly present before replay")
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
            first_cell_index=int(protocol["reference_pair"][0]),
            second_cell_index=int(protocol["reference_pair"][1]),
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
    layer4_exact = (
        list(network.layer4_spike_indices) == prior["raw_layer4_spike_indices"]
        and list(network.layer4_spike_times_ms) == prior["raw_layer4_spike_times_ms"]
    )
    if not layer4_exact:
        raise RuntimeError("layer-4 event stream differs from registered prior result")

    duration_ms = float(protocol["duration_ms"])
    bin_ms = float(protocol["histogram_bin_ms"])
    populations = {
        "thalamic_relay": (
            network.relay_spike_indices,
            network.relay_spike_times_ms,
        ),
        "layer4_excitatory": (
            network.layer4_spike_indices,
            network.layer4_spike_times_ms,
        ),
        "layer4_inhibitory": (
            network.interneuron_spike_indices,
            network.interneuron_spike_times_ms,
        ),
        "trn": (network.trn_spike_indices, network.trn_spike_times_ms),
        "layer6ii_category": (
            network.category_spike_indices,
            network.category_spike_times_ms,
        ),
    }
    spectra = {
        name: asdict(
            population_gamma_spectrum(
                times,
                spike_indices=indices,
                duration_ms=duration_ms,
                bin_ms=bin_ms,
            )
        )
        for name, (indices, times) in populations.items()
    }
    lags = {
        "relay_to_layer4_excitatory": asdict(
            population_peak_lag(
                network.relay_spike_times_ms,
                network.layer4_spike_times_ms,
                duration_ms=duration_ms,
                bin_ms=bin_ms,
            )
        ),
        "relay_to_layer4_inhibitory": asdict(
            population_peak_lag(
                network.relay_spike_times_ms,
                network.interneuron_spike_times_ms,
                duration_ms=duration_ms,
                bin_ms=bin_ms,
            )
        ),
        "layer4_inhibitory_to_excitatory": asdict(
            population_peak_lag(
                network.interneuron_spike_times_ms,
                network.layer4_spike_times_ms,
                duration_ms=duration_ms,
                bin_ms=bin_ms,
            )
        ),
    }
    learned_projection035 = np.asarray(
        training.learned_weights["modeldb112923.projection.035"], dtype=float
    )

    print(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": registration["result_id"],
                "date": registration["date"],
                "status": "completed-figure15-shared-clock-event-replay",
                "classification": registration["classification"],
                "registration": args.registration,
                "runtime_fingerprint": conventions.fingerprint,
                "figure6_gates": figure6_gates,
                "layer4_prior_event_stream_exact": layer4_exact,
                "population_spectra": spectra,
                "near_zero_population_lags": lags,
                "learned_projection035_summary": {
                    "edge_count": int(learned_projection035.size),
                    "minimum": float(np.min(learned_projection035)),
                    "maximum": float(np.max(learned_projection035)),
                    "mean": float(np.mean(learned_projection035)),
                    "nonzero_count": int(np.count_nonzero(learned_projection035)),
                },
                "event_streams": {
                    name: _events(indices, times) for name, (indices, times) in populations.items()
                },
                "nonspecific_event_times_ms": [
                    float(value) for value in network.nonspecific_spike_times_ms
                ],
                "parameter_selected": False,
                "generation_two_opened": False,
                "causal_origin_claimed": False,
                "original_smart_reproduced": False,
                "baseline_frozen": False,
            },
            sort_keys=False,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
