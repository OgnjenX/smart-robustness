"""Run preregistered Figure 6 control and conserved cortical point-HH arms."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from smart_robustness import classic_sector
from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.models.point_hh import (
    create_conserved_cortical_point_hh_population,
)
from smart_robustness.models.selective import (
    CORTICAL_CELL_CLASSES,
    make_selective_population_factory,
)
from smart_robustness.validation.figure6 import (
    BOTTOM_UP_PROJECTION_ID,
    TOP_DOWN_NARROW_PROJECTION_ID,
    TOP_DOWN_WIDE_PROJECTION_ID,
    Figure6LearningProtocol,
    assess_figure6_cortical_recruitment,
    assess_figure6_top_down_timing,
    run_figure6_learning,
)
from smart_robustness.validation.figure10_search_cycle_spread import (
    build_projection036_variance_sector,
)


def _trial(training, expected: set[int], repetition: int) -> dict[str, object]:
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
    return {
        "repetition": repetition,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "population_spike_counts": training.result.population_spikes,
        "population_spike_indices": training.result.population_spike_indices,
        "population_spike_times_ms": training.result.population_spike_times_ms,
        "relay_active_indices": sorted(set(relay_indices)),
        "relay_event_count": len(relay_indices),
        "bottom_up_weights": training.learned_weights[BOTTOM_UP_PROJECTION_ID],
        "top_down_wide_weights": training.learned_weights[
            TOP_DOWN_WIDE_PROJECTION_ID
        ],
        "top_down_narrow_weights": training.learned_weights[
            TOP_DOWN_NARROW_PROJECTION_ID
        ],
        "top_down_horizontal_contrast": (
            training.result.top_down_combined.horizontal_orientation_contrast
        ),
    }


def _without_repetition(trial: dict[str, object]) -> dict[str, object]:
    return {name: value for name, value in trial.items() if name != "repetition"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="configs/baselines/classic_smart_calibrated_v1.yaml",
    )
    parser.add_argument(
        "--study",
        default="configs/robustness/cortical_point_hh_conserved_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = load_frozen_classic_baseline(args.baseline)
    study = yaml.safe_load(Path(args.study).read_text())
    exact_reruns = int(study["execution"]["exact_reruns_per_arm"])
    if exact_reruns != 2:
        raise ValueError("the registered study requires exactly two reruns per arm")
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
    protocol = Figure6LearningProtocol(
        monitored_populations=tuple(training_profile["monitored_populations"])
    )
    factories = {
        "classic_control": None,
        "cortical_point_hh_conserved": make_selective_population_factory(
            create_conserved_cortical_point_hh_population,
            CORTICAL_CELL_CLASSES,
        ),
    }
    if tuple(factories) != tuple(study["execution"]["run_order"]):
        raise ValueError("runner arm order differs from the registered order")

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arms: dict[str, dict[str, object]] = {}
    original_builder = classic_sector.build_first_order_connected_sector
    try:
        classic_sector.build_first_order_connected_sector = (
            build_projection036_variance_sector
        )
        for arm_name, factory in factories.items():
            trials = []
            for repetition in range(exact_reruns):
                training = run_figure6_learning(
                    conventions=baseline.runtime_conventions(),
                    protocol=protocol,
                    projection_weight_scales=dict(baseline.projection_weight_scales),
                    population_factory=factory,
                    brian=brian,
                )
                trials.append(_trial(training, expected, repetition))
                exact_repeat = len(trials) < 2 or (
                    _without_repetition(trials[0])
                    == _without_repetition(trials[-1])
                )
                arms[arm_name] = {
                    "trials": trials,
                    "exact_repeat": exact_repeat,
                    "all_trials_pass": bool(
                        len(trials) == exact_reruns
                        and exact_repeat
                        and all(trial["all_gates_pass"] for trial in trials)
                    ),
                }
                output_path.write_text(
                    yaml.safe_dump(
                        {
                            "schema_version": 1,
                            "status": "running-cortical-point-hh-figure6",
                            "baseline_manifest": args.baseline,
                            "baseline_manifest_fingerprint": (
                                baseline.manifest_fingerprint
                            ),
                            "runtime_fingerprint": baseline.runtime_fingerprint,
                            "study": args.study,
                            "network_outcome_used_for_parameter_selection": False,
                            "arms": arms,
                        },
                        sort_keys=False,
                    )
                )
    finally:
        classic_sector.build_first_order_connected_sector = original_builder

    classic_pass = bool(arms["classic_control"]["all_trials_pass"])
    point_pass = bool(
        arms["cortical_point_hh_conserved"]["all_trials_pass"]
    )
    output_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "status": "completed-cortical-point-hh-figure6",
                "baseline_manifest": args.baseline,
                "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
                "runtime_fingerprint": baseline.runtime_fingerprint,
                "study": args.study,
                "network_outcome_used_for_parameter_selection": False,
                "classic_sentinel_pass": classic_pass,
                "point_hh_figure6_pass": point_pass,
                "stage_3_authorized": bool(classic_pass and point_pass),
                "arms": arms,
            },
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
