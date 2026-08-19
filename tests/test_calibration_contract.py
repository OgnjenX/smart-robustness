from pathlib import Path

import pytest
import yaml

from smart_robustness.validation.calibration import load_calibration_contract

ROOT = Path(__file__).parents[1]
CONTRACT_PATH = ROOT / "configs/calibration/classic_uncertainty_space.yaml"


def test_classic_calibration_contract_separates_training_and_holdout() -> None:
    contract = load_calibration_contract(CONTRACT_PATH)
    assert contract.base_tag == "classic-smart-source-constrained-v0.1.0"
    assert set(contract.training_targets).isdisjoint(contract.holdout_targets)
    assert len(contract.dimensions) == 8
    assert contract.fingerprint == load_calibration_contract(CONTRACT_PATH).fingerprint


def test_candidate_fingerprint_requires_complete_admissible_values() -> None:
    contract = load_calibration_contract(CONTRACT_PATH)
    candidate = {
        dimension.name: (
            dimension.values[0] if dimension.kind == "categorical" else dimension.grid[0]
        )
        for dimension in contract.dimensions
    }
    assert contract.candidate_fingerprint(candidate) == contract.candidate_fingerprint(candidate)
    with pytest.raises(ValueError, match="missing"):
        contract.candidate_fingerprint({})
    candidate["top_down_current_pA"] = 2000
    with pytest.raises(ValueError, match="outside declared bounds"):
        contract.candidate_fingerprint(candidate)


def test_contract_rejects_holdout_leakage(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONTRACT_PATH.read_text())
    raw["training_targets"].append(raw["holdout_targets"][0])
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="targets overlap"):
        load_calibration_contract(path)


def test_contract_rejects_published_free_parameter(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONTRACT_PATH.read_text())
    raw["dimensions"]["top_down_current_pA"]["status"] = "published"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="not calibration-admissible"):
        load_calibration_contract(path)
