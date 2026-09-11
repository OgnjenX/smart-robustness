"""Run the registered cortical localization arms on the frozen Figure 6 gate."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from smart_robustness import classic_sector
from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.models.adex import load_adex_parameter_map, make_somatic_adex_factory
from smart_robustness.models.gif import load_gif_parameter_map, make_somatic_gif_factory
from smart_robustness.models.selective import (
    CORTICAL_CELL_CLASSES,
    CORTICAL_EXCITATORY_CELL_CLASSES,
    CORTICAL_INHIBITORY_CELL_CLASSES,
    make_selective_population_factory,
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


def _trial(training, expected: set[int], seed: int | None) -> dict[str, object]:
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
        "seed": seed,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "population_spike_counts": training.result.population_spikes,
        "relay_active_indices": sorted(set(relay_indices)),
        "relay_event_count": len(relay_indices),
        "top_down_horizontal_contrast": (
            training.result.top_down_combined.horizontal_orientation_contrast
        ),
    }


def _summary(
    trials: list[dict[str, object]],
    rule: dict[str, object],
    *,
    stochastic: bool,
) -> dict[str, object]:
    gate_names = tuple(trials[0]["gates"]) if trials else ()
    counts = {
        name: sum(bool(trial["gates"][name]) for trial in trials)  # type: ignore[index]
        for name in gate_names
    }
    all_gate_count = sum(bool(trial["all_gates_pass"]) for trial in trials)
    if stochastic:
        complete = len(trials) == int(rule["stochastic_gif_total_trials"])
        progression = bool(
            complete
            and all_gate_count >= int(rule["stochastic_gif_minimum_successful_trials"])
            and all(
                value
                >= int(rule["stochastic_gif_each_individual_gate_minimum_trials"])
                for value in counts.values()
            )
        )
    else:
        complete = len(trials) == 1
        progression = bool(
            complete
            and all_gate_count == 1
            and all(value == 1 for value in counts.values())
        )
    return {
        "completed_trials": len(trials),
        "all_gate_success_count": all_gate_count,
        "individual_gate_success_counts": counts,
        "figure6_progression_pass": progression,
    }


def _factories() -> dict[str, object]:
    adex = make_somatic_adex_factory(
        load_adex_parameter_map("configs/models/adex_current_step_matched_v1.yaml")
    )
    gif = make_somatic_gif_factory(
        load_gif_parameter_map("configs/models/gif_current_step_matched_v1.yaml")
    )
    return {
        "classic_control": None,
        "adex_cortical_all": make_selective_population_factory(adex, CORTICAL_CELL_CLASSES),
        "adex_cortical_excitatory": make_selective_population_factory(
            adex, CORTICAL_EXCITATORY_CELL_CLASSES
        ),
        "adex_cortical_inhibitory": make_selective_population_factory(
            adex, CORTICAL_INHIBITORY_CELL_CLASSES
        ),
        "gif_cortical_all": make_selective_population_factory(gif, CORTICAL_CELL_CLASSES),
        "gif_cortical_excitatory": make_selective_population_factory(
            gif, CORTICAL_EXCITATORY_CELL_CLASSES
        ),
        "gif_cortical_inhibitory": make_selective_population_factory(
            gif, CORTICAL_INHIBITORY_CELL_CLASSES
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", default="configs/baselines/classic_smart_calibrated_v1.yaml"
    )
    parser.add_argument(
        "--study", default="configs/robustness/neuron_model_localization_v1.yaml"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--arms", nargs="*", default=None)
    args = parser.parse_args()

    baseline = load_frozen_classic_baseline(args.baseline)
    study = yaml.safe_load(Path(args.study).read_text())
    seeds = tuple(int(seed) for seed in study["gif_dynamics_seed_ensemble"]["seeds"])
    rule = study["figure6_progression_rule"]
    if len(seeds) != int(rule["stochastic_gif_total_trials"]):
        raise ValueError("registered seed count and Figure 6 trial count differ")

    raw_manifest = yaml.safe_load(Path(args.baseline).read_text())
    profile = yaml.safe_load(
        (baseline.repository_root / raw_manifest["implementation"]["profile"]["path"]).read_text()
    )
    training_profile = yaml.safe_load(
        (baseline.repository_root / profile["training_profile"]).read_text()
    )
    expected = set(profile["figure6_gates"]["relay_active_indices"])
    protocol = Figure6LearningProtocol(
        monitored_populations=tuple(training_profile["monitored_populations"])
    )
    factories = _factories()
    if args.arms:
        unknown = set(args.arms) - set(factories)
        if unknown:
            raise ValueError(f"unknown arms: {sorted(unknown)}")
        factories = {name: factories[name] for name in args.arms}

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arms: dict[str, dict[str, object]] = {
        name: {
            "trials": [],
            "summary": _summary([], rule, stochastic=name.startswith("gif_")),
        }
        for name in factories
    }
    original_builder = classic_sector.build_first_order_connected_sector
    try:
        classic_sector.build_first_order_connected_sector = build_projection036_variance_sector
        for arm_name, factory in factories.items():
            trials: list[dict[str, object]] = arms[arm_name]["trials"]  # type: ignore[assignment]
            stochastic = arm_name.startswith("gif_")
            arm_seeds: tuple[int | None, ...] = seeds if stochastic else (None,)
            for seed in arm_seeds:
                training = run_figure6_learning(
                    conventions=baseline.runtime_conventions(),
                    protocol=protocol,
                    projection_weight_scales=dict(baseline.projection_weight_scales),
                    population_factory=factory,
                    dynamics_seed=seed,
                    brian=brian,
                )
                trials.append(_trial(training, expected, seed))
                arms[arm_name]["summary"] = _summary(
                    trials, rule, stochastic=stochastic
                )
                output_path.write_text(
                    yaml.safe_dump(
                        {
                            "schema_version": 1,
                            "status": "running-neuron-model-localization",
                            "baseline_manifest": args.baseline,
                            "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
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

    output_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "status": "completed-neuron-model-localization",
                "baseline_manifest": args.baseline,
                "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
                "runtime_fingerprint": baseline.runtime_fingerprint,
                "study": args.study,
                "network_outcome_used_for_parameter_selection": False,
                "arms": arms,
            },
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
