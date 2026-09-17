"""Correct tuple/list bookkeeping after the hash-locked Figure 6 recovery."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml

REGISTERED_INPUT_SHA256 = (
    "7b791652e30a8650760a1a17a96b792ae40a964687781e95276d62f96e71f62f"
)


def _without_repetition(trial: dict[str, object]) -> dict[str, object]:
    return {name: value for name, value in trial.items() if name != "repetition"}


def _canonical(value: object) -> object:
    """Normalize exactly as the persisted safe-YAML result is represented."""

    return yaml.safe_load(yaml.safe_dump(value, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    result_path = Path(args.result)
    encoded = result_path.read_bytes()
    input_sha256 = hashlib.sha256(encoded).hexdigest()
    if input_sha256 != REGISTERED_INPUT_SHA256:
        raise ValueError("completed recovery result hash differs from registration")
    raw = yaml.safe_load(encoded)
    if raw.get("status") != "completed-cortical-point-hh-figure6":
        raise ValueError("correction requires the completed recovered result")
    arms = raw["arms"]
    classic = arms["classic_control"]
    point = arms["cortical_point_hh_conserved"]
    if (
        len(classic["trials"]) != 2
        or classic["exact_repeat"] is not True
        or classic["all_trials_pass"] is not True
    ):
        raise ValueError("classic sentinel is not complete, exact, and passing")
    if (
        len(point["trials"]) != 2
        or point["exact_repeat"] is not False
        or point["all_trials_pass"] is not False
        or not all(trial["all_gates_pass"] for trial in point["trials"])
    ):
        raise ValueError("point-HH result is not the registered correction state")
    first = _canonical(_without_repetition(point["trials"][0]))
    second = _canonical(_without_repetition(point["trials"][1]))
    if first != second:
        raise ValueError("canonical point-HH repeats are not exactly equal")

    point["exact_repeat"] = True
    point["all_trials_pass"] = True
    raw["point_hh_figure6_pass"] = True
    raw["stage_3_authorized"] = True
    raw["execution_recovery"]["exactness_correction"] = {
        "registration": (
            "docs/validation-results/"
            "mechanism-cortical-point-hh-figure6-exactness-correction-929c.yaml"
        ),
        "input_sha256": input_sha256,
        "comparison": "exact canonical safe-YAML equality excluding repetition",
        "trial_fields_changed": 0,
        "gates_changed": 0,
    }
    result_path.write_text(yaml.safe_dump(raw, sort_keys=False))


if __name__ == "__main__":
    main()
