"""Runner identity and seal tests; no scientific execution."""

import importlib.util
from pathlib import Path

import pytest
import yaml


def _runner():
    path = Path(__file__).parents[1] / "scripts/run_pulvinar_relay_isolated.py"
    spec = importlib.util.spec_from_file_location("pulvinar_relay_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expected_grid_is_complete_and_unique():
    runner = _runner()
    keys = runner.expected_run_keys()
    assert len(keys) == 20
    assert "frequency0.5-dt0.005-repeat1" in keys
    assert "frequency20-dt0.01-repeat0" in keys


def test_seal_rejects_wrong_status(tmp_path):
    runner = _runner()
    seal = tmp_path / "seal.yaml"
    seal.write_text(yaml.safe_dump({"status": "draft", "files": {}}))
    with pytest.raises(ValueError, match="execution seal"):
        runner.verify_seal(seal)


def test_seal_rejects_changed_file(tmp_path):
    runner = _runner()
    source = tmp_path / "source.py"
    source.write_text("one\n")
    seal = tmp_path / "seal.yaml"
    seal.write_text(yaml.safe_dump({
        "status": "sealed-before-isolated-relay-outcomes",
        "files": {
            str(source): "0" * 64,
            "scripts/run_pulvinar_relay_isolated.py": "0" * 64,
        },
    }))
    with pytest.raises(ValueError, match="sealed file changed"):
        runner.verify_seal(seal)
