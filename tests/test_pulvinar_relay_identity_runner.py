"""Source-identity runner seal checks only."""

import importlib.util
from pathlib import Path

import pytest
import yaml


def _runner():
    path = Path(__file__).parents[1] / "scripts/run_pulvinar_relay_source_identity.py"
    spec = importlib.util.spec_from_file_location("relay_identity_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seal_requires_exact_status(tmp_path):
    runner = _runner()
    seal = tmp_path / "seal.yaml"
    seal.write_text(yaml.safe_dump({"status": "draft", "files": {}}))
    with pytest.raises(ValueError, match="execution seal"):
        runner.verify_seal(seal)


def test_seal_checks_every_file(tmp_path):
    runner = _runner()
    source = tmp_path / "source"
    source.write_text("changed")
    seal = tmp_path / "seal.yaml"
    seal.write_text(yaml.safe_dump({
        "status": "sealed-before-source-identity-outcomes",
        "files": {str(source): "0" * 64},
    }))
    with pytest.raises(ValueError, match="sealed file changed"):
        runner.verify_seal(seal)
