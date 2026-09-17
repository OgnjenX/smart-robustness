"""Run registered Figure 7 and Figure 10 expanded layer-5 comparisons."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml
from run_layer5_distal_nak_stage3 import (
    _run_stage3_repetition,
    _sha256,
    _without_repetition,
)

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.models.expanded_layer5 import (
    LAYER5_CELL_CLASS,
    create_expanded_layer5_population,
)
from smart_robustness.models.selective import make_selective_population_factory

ARM_ORDER = ("classic_control", "expanded_layer5_branched")


def _canonical(value: object) -> object:
    return yaml.safe_load(yaml.safe_dump(value, sort_keys=True))


def _checkpoint(
    *,
    status: str,
    args,
    baseline,
    assessment,
    arms: dict[str, dict[str, object]],
    resume_input_sha256: str | None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": status,
        "baseline_manifest": args.baseline,
        "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
        "runtime_fingerprint": baseline.runtime_fingerprint,
        "study": args.study,
        "figure6_result_sha256": assessment["raw_result"]["sha256"],
        "network_outcome_used_for_parameter_selection": False,
        "resume_input_sha256": resume_input_sha256,
        "arms": arms,
    }


def _validate_resume(raw, *, args, baseline, assessment) -> None:
    if raw.get("status") != "running-expanded-layer5-stage3":
        raise ValueError("resume input is not a running expanded-L5 stage-3 result")
    expected = {
        "baseline_manifest": args.baseline,
        "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
        "runtime_fingerprint": baseline.runtime_fingerprint,
        "study": args.study,
        "figure6_result_sha256": assessment["raw_result"]["sha256"],
        "network_outcome_used_for_parameter_selection": False,
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ValueError(f"resume input differs at {key}")
    arms = raw.get("arms")
    if not isinstance(arms, dict):
        raise TypeError("resume input arms must be a mapping")
    unknown = set(arms) - set(ARM_ORDER)
    if unknown:
        raise ValueError(f"resume input contains unknown arms: {sorted(unknown)}")


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
        "--figure6-result",
        default="results/expanded-layer5-figure6-934.yaml",
    )
    parser.add_argument(
        "--figure6-assessment",
        default=(
            "docs/validation-results/"
            "mechanism-expanded-layer5-figure6-assessment-934.yaml"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-sha256")
    args = parser.parse_args()

    baseline = load_frozen_classic_baseline(args.baseline)
    study = yaml.safe_load(Path(args.study).read_text())
    assessment = yaml.safe_load(Path(args.figure6_assessment).read_text())
    figure6_path = Path(args.figure6_result)
    if _sha256(figure6_path) != assessment["raw_result"]["sha256"]:
        raise ValueError("Figure 6 result differs from its sealed assessment")
    if not assessment["decision"]["stage3_match_mismatch_and_reset_authorized"]:
        raise ValueError("Figure 6 assessment does not authorize stage 3")
    figure6 = yaml.safe_load(figure6_path.read_text())
    exact_reruns = int(study["execution"]["exact_reruns_per_arm"])
    if exact_reruns != 2:
        raise ValueError("the registered study requires exactly two reruns per arm")
    profile_path = baseline.repository_root / yaml.safe_load(
        Path(args.baseline).read_text()
    )["implementation"]["profile"]["path"]
    profile = yaml.safe_load(profile_path.read_text())
    factories = {
        "classic_control": None,
        "expanded_layer5_branched": make_selective_population_factory(
            create_expanded_layer5_population,
            {LAYER5_CELL_CLASS},
        ),
    }
    if tuple(factories) != ARM_ORDER or tuple(study["execution"]["run_order"]) != ARM_ORDER:
        raise ValueError("runner arm order differs from the registered order")

    output_path = Path(args.output)
    resume_input_sha256 = None
    if args.resume:
        if not args.resume_sha256:
            raise ValueError("--resume requires --resume-sha256")
        encoded = output_path.read_bytes()
        resume_input_sha256 = hashlib.sha256(encoded).hexdigest()
        if resume_input_sha256 != args.resume_sha256:
            raise ValueError("resume result hash differs from the registered input")
        running = yaml.safe_load(encoded)
        _validate_resume(
            running,
            args=args,
            baseline=baseline,
            assessment=assessment,
        )
        arms: dict[str, dict[str, object]] = running["arms"]
    else:
        if args.resume_sha256:
            raise ValueError("--resume-sha256 requires --resume")
        if output_path.exists():
            raise FileExistsError("refusing to overwrite an existing stage-3 result")
        arms = {}

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scales = dict(baseline.projection_weight_scales)
    for arm_index, (arm_name, population_factory) in enumerate(factories.items()):
        if any(
            later_name in arms for later_name in ARM_ORDER[arm_index + 1 :]
        ) and arm_name not in arms:
            raise ValueError("resume result violates the registered arm order")
        source_trials = figure6["arms"][arm_name]["trials"]
        if len(source_trials) != exact_reruns:
            raise ValueError(f"{arm_name} does not contain both sealed Figure 6 runs")
        learned_weight_keys = (
            "bottom_up_weights",
            "top_down_wide_weights",
            "top_down_narrow_weights",
        )
        if any(
            source_trials[0][key] != source_trials[1][key]
            for key in learned_weight_keys
        ):
            raise ValueError(f"{arm_name} Figure 6 learned weights are not exact")
        learned_weights = {
            "modeldb112923.projection.035": source_trials[0]["bottom_up_weights"],
            "modeldb112923.projection.005": source_trials[0][
                "top_down_wide_weights"
            ],
            "modeldb112923.projection.007": source_trials[0][
                "top_down_narrow_weights"
            ],
        }
        outcomes = list(arms.get(arm_name, {}).get("outcomes", []))
        if len(outcomes) > exact_reruns or [
            outcome["repetition"] for outcome in outcomes
        ] != list(range(len(outcomes))):
            raise ValueError(f"{arm_name} resume repetitions are invalid")
        for repetition in range(len(outcomes), exact_reruns):
            outcomes.append(
                _run_stage3_repetition(
                    population_factory=population_factory,
                    learned_weights=learned_weights,
                    conventions=baseline.runtime_conventions(),
                    scales=scales,
                    profile=profile,
                    repetition=repetition,
                    brian=brian,
                )
            )
            exact_repeat = len(outcomes) < 2 or (
                _canonical(_without_repetition(outcomes[0]))
                == _canonical(_without_repetition(outcomes[-1]))
            )
            arms[arm_name] = {
                "outcomes": outcomes,
                "exact_repeat": exact_repeat,
                "all_repetitions_pass": bool(
                    len(outcomes) == exact_reruns
                    and exact_repeat
                    and all(item["stage3_pass"] for item in outcomes)
                ),
            }
            output_path.write_text(
                yaml.safe_dump(
                    _checkpoint(
                        status="running-expanded-layer5-stage3",
                        args=args,
                        baseline=baseline,
                        assessment=assessment,
                        arms=arms,
                        resume_input_sha256=resume_input_sha256,
                    ),
                    sort_keys=False,
                )
            )

    classic_pass = bool(arms["classic_control"]["all_repetitions_pass"])
    expanded_pass = bool(
        arms["expanded_layer5_branched"]["all_repetitions_pass"]
    )
    completed = _checkpoint(
        status="completed-expanded-layer5-stage3",
        args=args,
        baseline=baseline,
        assessment=assessment,
        arms=arms,
        resume_input_sha256=resume_input_sha256,
    )
    completed.update(
        {
            "classic_stage3_pass": classic_pass,
            "expanded_layer5_stage3_pass": expanded_pass,
            "stage4_authorized": bool(classic_pass and expanded_pass),
        }
    )
    output_path.write_text(yaml.safe_dump(completed, sort_keys=False))


if __name__ == "__main__":
    main()
