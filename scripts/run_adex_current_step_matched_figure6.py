"""Run the registered current-step-matched AdEx arm on frozen Figure 6."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from smart_robustness import classic_sector
from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.models.adex import (
    load_adex_parameter_map,
    make_somatic_adex_factory,
)
from smart_robustness.validation.figure6 import (
    Figure6LearningProtocol,
    assess_figure6_cortical_recruitment,
    assess_figure6_top_down_timing,
    run_figure6_learning,
)
from smart_robustness.validation.figure10_search_cycle_spread import (
    build_projection036_variance_sector,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="configs/baselines/classic_smart_calibrated_v1.yaml",
    )
    parser.add_argument("--arm-label", default="somatic_adex_current_step_matched")
    parser.add_argument(
        "--result-status", default="completed-current-step-matched-adex-figure6"
    )
    parser.add_argument(
        "--parameters",
        default="configs/models/adex_current_step_matched_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = load_frozen_classic_baseline(args.baseline)
    raw_manifest = yaml.safe_load(Path(args.baseline).read_text())
    profile = yaml.safe_load(
        (
            baseline.repository_root
            / raw_manifest["implementation"]["profile"]["path"]
        ).read_text()
    )
    training_profile = yaml.safe_load(
        (baseline.repository_root / profile["training_profile"]).read_text()
    )
    expected = set(profile["figure6_gates"]["relay_active_indices"])
    factory = make_somatic_adex_factory(load_adex_parameter_map(args.parameters))

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    original_builder = classic_sector.build_first_order_connected_sector
    try:
        classic_sector.build_first_order_connected_sector = (
            build_projection036_variance_sector
        )
        training = run_figure6_learning(
            conventions=baseline.runtime_conventions(),
            protocol=Figure6LearningProtocol(
                monitored_populations=tuple(training_profile["monitored_populations"])
            ),
            projection_weight_scales=dict(baseline.projection_weight_scales),
            population_factory=factory,
            brian=brian,
        )
    finally:
        classic_sector.build_first_order_connected_sector = original_builder

    relay_indices = training.result.population_spike_indices["thalamic_relay"]
    relay_counts = {index: relay_indices.count(index) for index in expected}
    recruitment = assess_figure6_cortical_recruitment(training.result)
    timing = assess_figure6_top_down_timing(training.result)
    gates = {
        "relay_active_indices": set(relay_indices) == expected,
        "relay_events_per_active_index": set(relay_counts.values()) == {4},
        "relay_events": len(relay_indices) == 20,
        "cortical_chain_complete": recruitment.feedforward_chain_complete,
        "causal_learning_pair": timing.causal_pair_in_learning_window,
        "top_down_horizontal_contrast": (
            training.result.top_down_combined.horizontal_orientation_contrast > 0
        ),
    }
    output = {
        "schema_version": 1,
        "status": args.result_status,
        "baseline_manifest": args.baseline,
        "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
        "runtime_fingerprint": baseline.runtime_fingerprint,
        "parameter_manifest": args.parameters,
        "neuron_model": args.arm_label,
        "network_outcome_used_for_parameter_selection": False,
        "figure6_gates": gates,
        "figure6_all_gates_pass": all(gates.values()),
        "population_spike_counts": training.result.population_spikes,
        "relay_active_indices": sorted(set(relay_indices)),
        "relay_event_count": len(relay_indices),
        "top_down_horizontal_contrast": (
            training.result.top_down_combined.horizontal_orientation_contrast
        ),
        "learned_weights": training.learned_weights,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(output, sort_keys=False))


if __name__ == "__main__":
    main()
