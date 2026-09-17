"""Run preregistered Figure 6 control and expanded layer-5 arms."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from smart_robustness import classic_sector
from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.models.expanded_layer5 import (
    LAYER5_CELL_CLASS,
    create_expanded_layer5_population,
)
from smart_robustness.models.selective import make_selective_population_factory
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

ARM_ORDER = ("classic_control", "expanded_layer5_branched")


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


def _arm_summary(trials: list[dict[str, object]], exact_reruns: int):
    exact_repeat = len(trials) < 2 or all(
        _without_repetition(trial) == _without_repetition(trials[0])
        for trial in trials[1:]
    )
    return {
        "trials": trials,
        "exact_repeat": exact_repeat,
        "all_trials_pass": bool(
            len(trials) == exact_reruns
            and exact_repeat
            and all(trial["all_gates_pass"] for trial in trials)
        ),
    }


def _write_progress(
    output_path: Path,
    *,
    status: str,
    baseline,
    baseline_path: str,
    study_path: str,
    arms: dict[str, dict[str, object]],
    complete: bool,
) -> None:
    classic_pass = bool(arms.get("classic_control", {}).get("all_trials_pass", False))
    expanded_pass = bool(
        arms.get("expanded_layer5_branched", {}).get("all_trials_pass", False)
    )
    payload = {
        "schema_version": 1,
        "status": status,
        "baseline_manifest": baseline_path,
        "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
        "runtime_fingerprint": baseline.runtime_fingerprint,
        "study": study_path,
        "network_outcome_used_for_parameter_selection": False,
        "arms": arms,
    }
    if complete:
        payload.update(
            {
                "classic_sentinel_pass": classic_pass,
                "expanded_layer5_figure6_pass": expanded_pass,
                "stage_3_authorized": bool(classic_pass and expanded_pass),
            }
        )
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _load_resumable_arms(
    output_path: Path,
    *,
    baseline,
    baseline_path: str,
    study_path: str,
) -> dict[str, dict[str, object]]:
    if not output_path.exists():
        return {}
    partial = yaml.safe_load(output_path.read_text())
    expected_identity = {
        "baseline_manifest": baseline_path,
        "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
        "runtime_fingerprint": baseline.runtime_fingerprint,
        "study": study_path,
        "network_outcome_used_for_parameter_selection": False,
    }
    if any(partial.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("existing Figure 6 result does not match this sealed run")
    arms = partial.get("arms", {})
    if not isinstance(arms, dict) or set(arms) - set(ARM_ORDER):
        raise ValueError("existing Figure 6 result contains unknown arms")
    return arms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="configs/baselines/classic_smart_calibrated_v1.yaml",
    )
    parser.add_argument(
        "--study",
        default="configs/robustness/expanded_layer5_branched_v1.yaml",
    )
    parser.add_argument(
        "--precheck",
        default=(
            "docs/validation-results/mechanism-expanded-layer5-precheck-933.yaml"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    precheck = yaml.safe_load(Path(args.precheck).read_text())
    if precheck["decision"] != {
        "stage_1_pass": True,
        "figure6_authorized": True,
        "later_network_stages_authorized": False,
    }:
        raise ValueError("Artifact 933 does not authorize Figure 6")
    baseline = load_frozen_classic_baseline(args.baseline)
    study = yaml.safe_load(Path(args.study).read_text())
    exact_reruns = int(study["execution"]["exact_reruns_per_arm"])
    if exact_reruns != 2:
        raise ValueError("the registered study requires exactly two reruns per arm")
    if tuple(study["execution"]["run_order"]) != ARM_ORDER:
        raise ValueError("runner arm order differs from the registered order")
    raw_manifest = yaml.safe_load(Path(args.baseline).read_text())
    profile = yaml.safe_load(
        (
            baseline.repository_root / raw_manifest["implementation"]["profile"]["path"]
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
        "expanded_layer5_branched": make_selective_population_factory(
            create_expanded_layer5_population,
            {LAYER5_CELL_CLASS},
        ),
    }

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arms = _load_resumable_arms(
        output_path,
        baseline=baseline,
        baseline_path=args.baseline,
        study_path=args.study,
    )
    original_builder = classic_sector.build_first_order_connected_sector
    try:
        classic_sector.build_first_order_connected_sector = (
            build_projection036_variance_sector
        )
        for arm_name, factory in factories.items():
            saved_trials = arms.get(arm_name, {}).get("trials", [])
            if not isinstance(saved_trials, list) or len(saved_trials) > exact_reruns:
                raise ValueError(f"invalid saved trial count for {arm_name}")
            trials = list(saved_trials)
            for repetition in range(len(trials), exact_reruns):
                training = run_figure6_learning(
                    conventions=baseline.runtime_conventions(),
                    protocol=protocol,
                    projection_weight_scales=dict(baseline.projection_weight_scales),
                    population_factory=factory,
                    brian=brian,
                )
                trials.append(_trial(training, expected, repetition))
                arms[arm_name] = _arm_summary(trials, exact_reruns)
                _write_progress(
                    output_path,
                    status="running-expanded-layer5-figure6",
                    baseline=baseline,
                    baseline_path=args.baseline,
                    study_path=args.study,
                    arms=arms,
                    complete=False,
                )
    finally:
        classic_sector.build_first_order_connected_sector = original_builder

    _write_progress(
        output_path,
        status="completed-expanded-layer5-figure6",
        baseline=baseline,
        baseline_path=args.baseline,
        study_path=args.study,
        arms=arms,
        complete=True,
    )


if __name__ == "__main__":
    main()
