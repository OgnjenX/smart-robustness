"""Merge complete, non-overlapping GIF localization shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from run_neuron_model_localization_figure6 import _summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="configs/robustness/neuron_model_localization_v1.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    study = yaml.safe_load(Path(args.study).read_text())
    expected_seeds = tuple(int(seed) for seed in study["gif_dynamics_seed_ensemble"]["seeds"])
    rule = study["figure6_progression_rule"]
    common: dict[str, object] | None = None
    arms: dict[str, dict[str, object]] = {}
    for input_name in args.inputs:
        payload = yaml.safe_load(Path(input_name).read_text())
        if payload["status"] not in {
            "completed-neuron-model-localization",
            "completed-neuron-model-localization-shard",
            "running-neuron-model-localization",
        }:
            raise ValueError(f"incomplete localization result: {input_name}")
        identity = {
            "baseline_manifest": payload["baseline_manifest"],
            "baseline_manifest_fingerprint": payload["baseline_manifest_fingerprint"],
            "runtime_fingerprint": payload["runtime_fingerprint"],
            "study": payload["study"],
        }
        if common is None:
            common = identity
        elif identity != common:
            raise ValueError("localization shards do not share one frozen execution identity")
        for arm_name, arm in payload["arms"].items():
            if not arm_name.startswith("gif_"):
                raise ValueError("the shard merger accepts stochastic GIF arms only")
            target = arms.setdefault(arm_name, {"trial_records": []})
            target["trial_records"].extend(  # type: ignore[union-attr]
                (trial, payload["status"]) for trial in arm["trials"]
            )

    for arm_name, arm in arms.items():
        records = arm.pop("trial_records")
        by_seed = {}
        for trial, source_status in records:
            seed = int(trial["seed"])
            if seed in by_seed:
                previous, previous_status = by_seed[seed]
                if trial == previous:
                    continue
                completed = source_status != "running-neuron-model-localization"
                previous_completed = (
                    previous_status != "running-neuron-model-localization"
                )
                if completed == previous_completed:
                    raise ValueError(
                        f"non-identical equal-status duplicate GIF seed in {arm_name}: {seed}"
                    )
                if not completed:
                    continue
            by_seed[seed] = (trial, source_status)
        seeds = list(by_seed)
        if set(seeds) != set(expected_seeds):
            missing = sorted(set(expected_seeds) - set(seeds))
            extra = sorted(set(seeds) - set(expected_seeds))
            raise ValueError(f"incomplete GIF seed ensemble for {arm_name}: missing={missing}, extra={extra}")
        arm["trials"] = [by_seed[seed][0] for seed in sorted(by_seed)]
        arm["summary"] = _summary(arm["trials"], rule, stochastic=True)

    if common is None:
        raise ValueError("no localization shards supplied")
    Path(args.output).write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "status": "completed-neuron-model-localization-merged",
                **common,
                "network_outcome_used_for_parameter_selection": False,
                "arms": arms,
            },
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
