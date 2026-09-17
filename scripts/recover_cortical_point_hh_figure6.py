"""Recover only the interrupted second point-HH Figure 6 repeat."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml
from run_cortical_point_hh_figure6 import _trial, _without_repetition

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
    Figure6LearningProtocol,
    run_figure6_learning,
)
from smart_robustness.validation.figure10_search_cycle_spread import (
    build_projection036_variance_sector,
)

REGISTERED_PARTIAL_SHA256 = (
    "36b97bb8d2c8bae573ff00b473268b9b78f59d598b3a5fb63510a5ac4d74e446"
)


def _validate_partial(raw: dict[str, object], baseline, args) -> None:
    if raw.get("status") != "running-cortical-point-hh-figure6":
        raise ValueError("recovery requires the registered running partial result")
    if raw.get("baseline_manifest") != args.baseline:
        raise ValueError("partial baseline path differs from recovery baseline")
    if raw.get("study") != args.study:
        raise ValueError("partial study path differs from recovery study")
    if raw.get("baseline_manifest_fingerprint") != baseline.manifest_fingerprint:
        raise ValueError("partial baseline fingerprint differs")
    if raw.get("runtime_fingerprint") != baseline.runtime_fingerprint:
        raise ValueError("partial runtime fingerprint differs")
    if raw.get("network_outcome_used_for_parameter_selection") is not False:
        raise ValueError("partial result violates the no-selection rule")
    arms = raw.get("arms")
    if not isinstance(arms, dict) or set(arms) != {
        "classic_control",
        "cortical_point_hh_conserved",
    }:
        raise ValueError("partial result does not contain the registered arms")
    classic = arms["classic_control"]
    point = arms["cortical_point_hh_conserved"]
    if (
        len(classic["trials"]) != 2
        or classic["exact_repeat"] is not True
        or classic["all_trials_pass"] is not True
    ):
        raise ValueError("partial classic sentinel is not complete and passing")
    if (
        len(point["trials"]) != 1
        or point["trials"][0]["repetition"] != 0
        or point["trials"][0]["all_gates_pass"] is not True
        or point["all_trials_pass"] is not False
    ):
        raise ValueError("partial point-HH arm is not the registered one-repeat state")


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

    output_path = Path(args.output)
    partial_bytes = output_path.read_bytes()
    partial_sha256 = hashlib.sha256(partial_bytes).hexdigest()
    if partial_sha256 != REGISTERED_PARTIAL_SHA256:
        raise ValueError("partial result hash differs from the registered recovery input")
    raw = yaml.safe_load(partial_bytes)
    baseline = load_frozen_classic_baseline(args.baseline)
    _validate_partial(raw, baseline, args)

    study = yaml.safe_load(Path(args.study).read_text())
    if int(study["execution"]["exact_reruns_per_arm"]) != 2:
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
    factory = make_selective_population_factory(
        create_conserved_cortical_point_hh_population,
        CORTICAL_CELL_CLASSES,
    )

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    original_builder = classic_sector.build_first_order_connected_sector
    try:
        classic_sector.build_first_order_connected_sector = (
            build_projection036_variance_sector
        )
        training = run_figure6_learning(
            conventions=baseline.runtime_conventions(),
            protocol=protocol,
            projection_weight_scales=dict(baseline.projection_weight_scales),
            population_factory=factory,
            brian=brian,
        )
    finally:
        classic_sector.build_first_order_connected_sector = original_builder

    arms = raw["arms"]
    point = arms["cortical_point_hh_conserved"]
    point["trials"].append(_trial(training, expected, repetition=1))
    point["exact_repeat"] = (
        _without_repetition(point["trials"][0])
        == _without_repetition(point["trials"][1])
    )
    point["all_trials_pass"] = bool(
        point["exact_repeat"]
        and all(trial["all_gates_pass"] for trial in point["trials"])
    )
    classic_pass = bool(arms["classic_control"]["all_trials_pass"])
    point_pass = bool(point["all_trials_pass"])
    raw.update(
        {
            "status": "completed-cortical-point-hh-figure6",
            "classic_sentinel_pass": classic_pass,
            "point_hh_figure6_pass": point_pass,
            "stage_3_authorized": bool(classic_pass and point_pass),
            "execution_recovery": {
                "registration": (
                    "docs/validation-results/"
                    "mechanism-cortical-point-hh-figure6-recovery-registration-929r.yaml"
                ),
                "partial_sha256": partial_sha256,
                "preserved_completed_trials": 3,
                "executed_missing_arm": "cortical_point_hh_conserved",
                "executed_missing_repetition": 1,
            },
        }
    )
    output_path.write_text(yaml.safe_dump(raw, sort_keys=False))


if __name__ == "__main__":
    main()
