"""Run the sealed Figure 6 direct visual-thalamus to L5 strength grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from smart_robustness import classic_sector
from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.models.direct_visual_thalamus_layer5 import (
    make_direct_visual_thalamus_layer5_sector_builder,
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

ARM_RATIOS = {
    "ratio_0": 0.0,
    "ratio_0p03125": 0.03125,
    "ratio_0p0625": 0.0625,
    "ratio_0p125": 0.125,
    "ratio_0p25": 0.25,
    "ratio_0p5": 0.5,
    "ratio_1": 1.0,
}
ARM_ORDER = tuple(ARM_RATIOS)


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


def _arm_summary(
    ratio: float,
    trials: list[dict[str, object]],
    exact_reruns: int,
) -> dict[str, object]:
    exact_repeat = len(trials) < 2 or all(
        _without_repetition(trial) == _without_repetition(trials[0])
        for trial in trials[1:]
    )
    return {
        "strength_ratio": ratio,
        "trials": trials,
        "exact_repeat": exact_repeat,
        "all_trials_pass": bool(
            len(trials) == exact_reruns
            and exact_repeat
            and all(trial["all_gates_pass"] for trial in trials)
        ),
    }


def _identity(args, baseline) -> dict[str, object]:
    return {
        "baseline_manifest": args.baseline,
        "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
        "runtime_fingerprint": baseline.runtime_fingerprint,
        "study": args.study,
        "precheck": args.precheck,
        "network_outcome_used_for_parameter_selection": False,
    }


def _load_arms(output: Path, *, identity: dict[str, object]) -> dict[str, object]:
    if not output.exists():
        return {}
    partial = yaml.safe_load(output.read_text())
    if any(partial.get(key) != value for key, value in identity.items()):
        raise ValueError("existing Figure 6 result does not match this sealed run")
    arms = partial.get("arms", {})
    if not isinstance(arms, dict) or set(arms) - set(ARM_ORDER):
        raise ValueError("existing Figure 6 result contains unknown arms")
    return arms


def _write(
    output: Path,
    *,
    status: str,
    identity: dict[str, object],
    arms: dict[str, object],
    complete: bool,
) -> None:
    payload = {
        "schema_version": 1,
        "status": status,
        **identity,
        "arms": arms,
    }
    if complete:
        passing = [
            ARM_RATIOS[name]
            for name in ARM_ORDER
            if arms[name]["all_trials_pass"]
        ]
        payload.update(
            {
                "all_exact_repeats": all(arms[name]["exact_repeat"] for name in ARM_ORDER),
                "figure6_passing_ratios": passing,
                "stage3_authorized_ratios": passing,
                "primary_source_consistent_passing_ratios": [
                    ratio for ratio in passing if 0.0 < ratio < 1.0
                ],
            }
        )
    output.write_text(yaml.safe_dump(payload, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="configs/baselines/classic_smart_calibrated_v1.yaml",
    )
    parser.add_argument(
        "--study",
        default="configs/robustness/direct_visual_thalamus_layer5_v1.yaml",
    )
    parser.add_argument(
        "--precheck",
        default=(
            "docs/validation-results/"
            "post2008-direct-visual-thalamus-layer5-precheck-939.yaml"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    precheck = yaml.safe_load(Path(args.precheck).read_text())
    if precheck["decision"]["stage_1_pass"] is not True or precheck["decision"][
        "figure6_grid_authorized"
    ] is not True:
        raise ValueError("Artifact 939 does not authorize Figure 6")
    baseline = load_frozen_classic_baseline(args.baseline)
    study = yaml.safe_load(Path(args.study).read_text())
    if tuple(study["execution"]["run_order"]) != ARM_ORDER:
        raise ValueError("runner arm order differs from the sealed study")
    ratios = tuple(float(value) for value in study["independent_variable"]["values"])
    if ratios != tuple(ARM_RATIOS.values()):
        raise ValueError("runner ratios differ from the sealed study")
    exact_reruns = int(study["execution"]["exact_reruns_per_arm"])
    if exact_reruns != 2:
        raise ValueError("the sealed study requires exactly two reruns per arm")

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

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = _identity(args, baseline)
    arms = _load_arms(output, identity=identity)
    original_builder = classic_sector.build_first_order_connected_sector
    try:
        for arm_index, arm_name in enumerate(ARM_ORDER):
            if any(name in arms for name in ARM_ORDER[arm_index + 1 :]) and arm_name not in arms:
                raise ValueError("resume result violates the sealed arm order")
            ratio = ARM_RATIOS[arm_name]
            trials = list(arms.get(arm_name, {}).get("trials", []))
            if len(trials) > exact_reruns:
                raise ValueError(f"invalid saved trial count for {arm_name}")
            classic_sector.build_first_order_connected_sector = (
                make_direct_visual_thalamus_layer5_sector_builder(
                    strength_ratio=ratio,
                    base_builder=build_projection036_variance_sector,
                )
            )
            for repetition in range(len(trials), exact_reruns):
                training = run_figure6_learning(
                    conventions=baseline.runtime_conventions(),
                    protocol=protocol,
                    projection_weight_scales=dict(baseline.projection_weight_scales),
                    brian=brian,
                )
                trials.append(_trial(training, expected, repetition))
                arms[arm_name] = _arm_summary(ratio, trials, exact_reruns)
                _write(
                    output,
                    status="running-direct-visual-thalamus-layer5-figure6",
                    identity=identity,
                    arms=arms,
                    complete=False,
                )
    finally:
        classic_sector.build_first_order_connected_sector = original_builder

    _write(
        output,
        status="completed-direct-visual-thalamus-layer5-figure6",
        identity=identity,
        arms=arms,
        complete=True,
    )


if __name__ == "__main__":
    main()
