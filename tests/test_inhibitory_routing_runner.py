"""Exact-null inhibitory-routing runner seal and summary checks."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


def _runner():
    path = Path(__file__).parents[1] / "scripts/run_inhibitory_routing_null.py"
    spec = importlib.util.spec_from_file_location("inhibitory_routing_runner", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_seal_requires_exact_status(tmp_path) -> None:
    runner = _runner()
    seal = tmp_path / "seal.yaml"
    seal.write_text(yaml.safe_dump({"status": "draft", "files": {}}))
    with pytest.raises(ValueError, match="behavioral execution seal"):
        runner.verify_seal(seal)


def test_seal_checks_every_file(tmp_path) -> None:
    runner = _runner()
    source = tmp_path / "source"
    source.write_text("changed")
    seal = tmp_path / "seal.yaml"
    seal.write_text(
        yaml.safe_dump(
            {
                "status": "sealed-before-null-behavioral-outcomes",
                "files": {str(source): "0" * 64},
            }
        )
    )
    with pytest.raises(ValueError, match="sealed file changed"):
        runner.verify_seal(seal)


def test_arm_summary_requires_two_exact_passing_repetitions() -> None:
    runner = _runner()
    outcomes = [
        {"repetition": 0, "behavioral_pass": True},
        {"repetition": 1, "behavioral_pass": True},
    ]
    summary = runner._arm_summary(outcomes)
    assert summary["exact_repeat"]
    assert summary["all_behavioral_gates_pass"]


def test_arm_summary_rejects_nonexact_repetition() -> None:
    runner = _runner()
    outcomes = [
        {"repetition": 0, "behavioral_pass": True, "value": 1},
        {"repetition": 1, "behavioral_pass": True, "value": 2},
    ]
    summary = runner._arm_summary(outcomes)
    assert not summary["exact_repeat"]
    assert not summary["all_behavioral_gates_pass"]
