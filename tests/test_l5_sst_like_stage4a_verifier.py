from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_l5_sst_like_stage4a.py"


def load_verifier():
    spec = importlib.util.spec_from_file_location("l5_sst_stage4a_verifier", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def completed_payload(verifier) -> dict:
    points = []
    for registered in verifier.runner.registered_points():
        outcomes = [
            {
                "repetition": repetition,
                "figure6": {"pass": True, "gates": {"learning": True}},
                "figure7": {"pass": True, "gates": {"match": True}},
                "figure10": {"pass": True, "gates": {"reset": True}},
                "progression_pass": True,
            }
            for repetition in range(2)
        ]
        points.append(
            {
                **registered,
                "outcomes": outcomes,
                "exact_repeat": True,
                "all_progression_gates_pass": True,
                "classification": "learning_first_order_survival",
            }
        )
    return {
        "status": "completed-l5-sst-like-stage4a",
        "identity": {},
        "points": points,
        "all_points_reported": True,
        "classification_counts": {"learning_first_order_survival": 7},
        "network_execution": True,
        "frozen_baseline_modified": False,
    }


def install_fakes(monkeypatch, verifier, payload: dict) -> None:
    seal = {
        "stage4_registration_sha256": "registration",
        "stage3_assessment_sha256": "assessment",
        "baseline_manifest_fingerprint": "baseline",
        "runtime_fingerprint": "runtime",
    }
    monkeypatch.setattr(verifier.runner, "verify_seal", lambda path: seal)
    monkeypatch.setattr(verifier, "file_sha256", lambda path: f"hash:{Path(path).name}")

    expected_in_order = [
        {
            name: point[name]
            for name in (
                "exact_repeat",
                "all_progression_gates_pass",
                "classification",
            )
        }
        for point in payload["points"]
    ]
    calls = iter(expected_in_order)
    monkeypatch.setattr(verifier.runner, "classify_point", lambda outcomes: next(calls))


def write_payload(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "result.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def test_verifier_recomputes_complete_registered_result(monkeypatch, tmp_path: Path) -> None:
    verifier = load_verifier()
    payload = completed_payload(verifier)
    install_fakes(monkeypatch, verifier, payload)
    payload["identity"] = {
        "seal_sha256": "hash:seal.yaml",
        "stage4_registration_sha256": "registration",
        "stage3_assessment_sha256": "assessment",
        "baseline_manifest_fingerprint": "baseline",
        "runtime_fingerprint": "runtime",
        "registered_points": verifier.runner.registered_points(),
    }
    evidence = verifier.verify_result(
        write_payload(tmp_path, payload), tmp_path / "seal.yaml"
    )
    assert evidence["independent_verification_passed"] is True
    assert evidence["completed_points"] == 7
    assert evidence["completed_repetitions"] == 14
    assert evidence["classification_counts"] == {
        "learning_first_order_survival": 7
    }
    assert evidence["all_points_exact_repeat"] is True
    assert evidence["all_points_survive_stage4a"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(status="running"), "not terminal"),
        (lambda payload: payload["points"].pop(), "point inventory is incomplete"),
        (
            lambda payload: payload["points"][0].update(exact_repeat=False),
            "stored point assessment mismatch",
        ),
        (
            lambda payload: payload.update(classification_counts={"failure": 7}),
            "classification counts mismatch",
        ),
    ],
)
def test_verifier_rejects_incomplete_or_inconsistent_results(
    monkeypatch, tmp_path: Path, mutation, message: str
) -> None:
    verifier = load_verifier()
    payload = completed_payload(verifier)
    install_fakes(monkeypatch, verifier, payload)
    payload["identity"] = {
        "seal_sha256": "hash:seal.yaml",
        "stage4_registration_sha256": "registration",
        "stage3_assessment_sha256": "assessment",
        "baseline_manifest_fingerprint": "baseline",
        "runtime_fingerprint": "runtime",
        "registered_points": verifier.runner.registered_points(),
    }
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        verifier.verify_result(write_payload(tmp_path, payload), tmp_path / "seal.yaml")
