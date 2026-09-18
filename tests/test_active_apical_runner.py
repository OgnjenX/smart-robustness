"""Test runner safeguards without executing biological response simulations."""

import runpy
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def runner():
    return runpy.run_path(str(Path(__file__).parents[1] / "scripts/run_active_apical_isolated.py"))


def test_atomic_checkpoint_replacement(runner, tmp_path):
    path = tmp_path / "checkpoint.yaml"
    runner["checkpoint"](path, {"stage": 1})
    runner["checkpoint"](path, {"stage": 2})
    assert yaml.safe_load(path.read_text()) == {"stage": 2}
    assert not list(tmp_path.glob(".apical-checkpoint-*"))


def test_unsealed_execution_rejected(runner, tmp_path):
    path = tmp_path / "seal.yaml"
    path.write_text(yaml.safe_dump({"status": "draft"}))
    with pytest.raises(ValueError, match="seal"):
        runner["verify_seal"](path)


def test_seal_detects_changed_file_and_missing_runner(runner, tmp_path):
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    seal = tmp_path / "seal.yaml"
    data = {
        "status": "sealed-before-isolated-outcomes",
        "files": {str(source): runner["file_sha256"](source)},
    }
    seal.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="runner"):
        runner["verify_seal"](seal)
    source.write_text("value = 2\n")
    with pytest.raises(ValueError, match="changed"):
        runner["verify_seal"](seal)
