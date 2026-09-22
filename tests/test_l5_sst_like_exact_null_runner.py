from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_l5_sst_like_exact_null.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("l5_sst_like_null_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_structural_noop_proof_covers_every_registered_delay() -> None:
    runner = load_runner()
    proof = runner.structural_noop_proof()
    assert proof == {
        "resource_nS": 0.0,
        "registered_delay_labels_ms": [1.0, 3.0, 7.0],
        "base_builder_calls": 3,
        "same_return_identity": True,
        "args_preserved": True,
        "kwargs_preserved": True,
        "population_factory_not_injected": True,
        "port_created": False,
        "synapse_created": False,
        "all_gates_pass": True,
    }


def test_archived_legacy_assessment_records_four_exact_passes() -> None:
    path = (
        ROOT
        / "docs/validation-results/post2008-inhibitory-routing-null-assessment-974.yaml"
    )
    assessment = yaml.safe_load(path.read_text())
    execution = assessment["execution"]

    assert assessment["raw_result_sha256"] == (
        "4ad5dc017a0112dbad1434846c4542caa2bc48e2d3a6ae784499924a5d3b133e"
    )
    assert execution["completed_runs"] == 4
    assert execution["exact_within_arm_repeats"] == {
        "unwrapped_control": True,
        "wrapped_legacy_aggregate": True,
    }
    assert execution["exact_wrapped_vs_unwrapped"] is True
    assert execution["all_behavioral_gates_passed"] == {
        "unwrapped_control": True,
        "wrapped_legacy_aggregate": True,
    }


def test_runner_rejects_unsealed_status(tmp_path: Path) -> None:
    runner = load_runner()
    seal = tmp_path / "seal.yaml"
    seal.write_text(yaml.safe_dump({"status": "draft", "files": {}}))
    with pytest.raises(ValueError, match="execution seal"):
        runner.verify_seal(seal)
