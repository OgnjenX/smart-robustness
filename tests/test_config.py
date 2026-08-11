from pathlib import Path

from smart_robustness.config import load_config


def test_reference_configs_are_valid_and_distinct() -> None:
    root = Path(__file__).parents[1]
    match = load_config(root / "configs/match.yaml")
    mismatch = load_config(root / "configs/mismatch.yaml")
    assert match.condition == "match"
    assert mismatch.condition == "mismatch"
    assert match.fingerprint != mismatch.fingerprint


def test_config_fingerprint_is_stable() -> None:
    path = Path(__file__).parents[1] / "configs/match.yaml"
    assert load_config(path).fingerprint == load_config(path).fingerprint
