"""Independently assess all fourteen Stage-4B outcomes and OS provenance."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import run_l5_sst_like_stage4b_continuation as runner
import yaml
from verify_l5_sst_like_stage4b import _check_outcome

from smart_robustness.validation.active_apical_recording import file_sha256


def check_point(point: dict, protocol: dict) -> dict:
    if len(point["outcomes"]) != 2:
        raise ValueError("point lacks two repetitions")
    summaries = [_check_outcome(outcome, protocol) for outcome in point["outcomes"]]
    normalized = []
    for outcome in point["outcomes"]:
        copy = deepcopy(outcome)
        copy.pop("repetition", None)
        normalized.append(copy)
    exact = normalized[0] == normalized[1]
    passed = all(outcome["figure14_pass"] is True for outcome in point["outcomes"])
    classification = (
        "engineering_stop" if not exact else "figure14_survival" if passed else "figure14_failure"
    )
    if (
        point.get("exact_repeat") is not exact
        or point.get("both_figure14_gate_sets_pass") is not passed
        or point.get("classification") != classification
    ):
        raise ValueError("classification contradicts independent comparison")
    if not exact:
        raise ValueError("exact repeat failed")
    return {
        "point_id": point["point_id"],
        "classification": classification,
        "repetitions": summaries,
    }


def verify_result(path: Path, seal_path: Path = runner.SEAL) -> dict:
    runner.verify_seal(seal_path)
    registration, source, _baseline, learning, _profile = runner.load_inputs()
    if path.resolve() != Path(registration["result"]).resolve():
        raise ValueError("result path changed")
    payload = yaml.safe_load(path.read_text())
    runner.validate_payload(payload, source, registration, seal_path)
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "completed-stage4b-continuation"
    ):
        raise ValueError("continuation is not terminal")
    if payload.get("all_points_reported") is not True or len(payload["points"]) != 7:
        raise ValueError("continuation point inventory incomplete")
    evidence = []
    for index, point in enumerate(payload["points"]):
        if point["stage4a_classification"] != learning["points"][index]["classification"]:
            raise ValueError("source learning classification changed")
        evidence.append(check_point(point, source["identity"]["protocol"]))
    counts = dict(sorted(Counter(p["classification"] for p in evidence).items()))
    if payload["classification_counts"] != counts:
        raise ValueError("classification counts changed")
    return {
        "raw_result_sha256": file_sha256(path),
        "completed_points": 7,
        "completed_repetitions": 14,
        "all_points_exact_repeat": True,
        "original_ten_outcomes_preserved": True,
        "per_repetition_environments_verified": True,
        "classification_counts": counts,
        "point_evidence": evidence,
        "independent_verification_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_result(args.result), indent=2))


if __name__ == "__main__":
    main()
