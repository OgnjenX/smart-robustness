from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_l5_sst_like_stage4a.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("l5_sst_stage4a_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def outcome(*, repetition: int, passed: bool = True, marker: int = 1) -> dict:
    return {
        "repetition": repetition,
        "figure6": {"pass": passed, "marker": marker},
        "figure7": {"pass": passed},
        "figure10": {"pass": passed} if passed else None,
        "progression_pass": passed,
    }


def test_registered_points_are_exactly_the_seven_stage3_survivors() -> None:
    runner = load_runner()
    points = runner.registered_points()
    assert [point["point_id"] for point in points] == list(runner.ELIGIBLE_POINT_IDS)
    assert [point["stage4a_index"] for point in points] == list(range(7))
    stage3 = {point["point_id"]: point for point in runner.stage3_runner.registered_points()}
    assert all(
        point["delay_ms"] == stage3[point["point_id"]]["delay_ms"]
        and point["resource_fraction"]
        == stage3[point["point_id"]]["resource_fraction"]
        and point["total_conductance_nS"]
        == stage3[point["point_id"]]["total_conductance_nS"]
        for point in points
    )


def test_point_builder_injects_only_registered_coordinate(monkeypatch) -> None:
    runner = load_runner()
    captured = {}

    def fake_wrapper(*, base_builder, total_conductance_nS, delay_ms):
        captured.update(
            base_builder=base_builder,
            total_conductance_nS=total_conductance_nS,
            delay_ms=delay_ms,
        )
        return "builder"

    monkeypatch.setattr(runner, "make_l5_sst_like_sector_builder", fake_wrapper)
    point = runner.registered_points()[2]
    assert runner._point_builder(point) == "builder"
    assert captured == {
        "base_builder": runner.build_projection036_variance_sector,
        "total_conductance_nS": point["total_conductance_nS"],
        "delay_ms": point["delay_ms"],
    }


def test_classification_requires_exact_complete_progression() -> None:
    runner = load_runner()
    passed = runner.classify_point([outcome(repetition=0), outcome(repetition=1)])
    failed_gate = runner.classify_point(
        [outcome(repetition=0, passed=False), outcome(repetition=1, passed=False)]
    )
    failed_repeat = runner.classify_point(
        [outcome(repetition=0), outcome(repetition=1, marker=2)]
    )
    assert passed == {
        "exact_repeat": True,
        "all_progression_gates_pass": True,
        "classification": "learning_first_order_survival",
    }
    assert failed_gate["classification"] == "failure"
    assert failed_repeat["exact_repeat"] is False
    assert failed_repeat["classification"] == "failure"
    with pytest.raises(ValueError, match="exactly two"):
        runner.classify_point([outcome(repetition=0)])


def test_checkpoint_validation_rejects_reorder_and_skips() -> None:
    runner = load_runner()
    points = runner.registered_points()
    first = {
        **points[0],
        "outcomes": [outcome(repetition=0), outcome(repetition=1)],
        "classification": "learning_first_order_survival",
    }
    second = {
        **points[1],
        "outcomes": [outcome(repetition=0)],
        "classification": None,
    }
    runner.validate_checkpoint_points([first, second])
    with pytest.raises(ValueError, match="order or identity"):
        runner.validate_checkpoint_points([second])
    malformed = dict(second)
    malformed["outcomes"] = [outcome(repetition=1)]
    with pytest.raises(ValueError, match="repetition order"):
        runner.validate_checkpoint_points([first, malformed])
    incomplete = {
        **points[0],
        "outcomes": [outcome(repetition=0)],
        "classification": None,
    }
    complete_second = {
        **points[1],
        "outcomes": [outcome(repetition=0), outcome(repetition=1)],
        "classification": "learning_first_order_survival",
    }
    with pytest.raises(ValueError, match="incomplete point"):
        runner.validate_checkpoint_points([incomplete, complete_second])


def test_runner_rejects_unsealed_status(tmp_path: Path) -> None:
    runner = load_runner()
    seal = tmp_path / "seal.yaml"
    seal.write_text(yaml.safe_dump({"status": "draft", "files": {}}))
    with pytest.raises(ValueError, match="execution seal"):
        runner.verify_seal(seal)
