"""The OS bridge must detect scientific differences and forbid moving targets."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


def load_bridge():
    path = Path(__file__).parents[1] / "scripts/run_l5_sst_like_stage4b_os_bridge.py"
    spec = importlib.util.spec_from_file_location("sst_os_bridge", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_replay_point_and_environment_are_fixed():
    bridge = load_bridge()
    registration = yaml.safe_load(bridge.REGISTRATION.read_text())
    bridge.validate_registration(registration, registration["new_environment"])
    with pytest.raises(ValueError, match="environment"):
        bridge.validate_registration(registration, registration["old_environment"])
    registration["point_index"] = 4
    with pytest.raises(ValueError, match="selection"):
        bridge.validate_registration(registration, registration["new_environment"])


def test_exact_replay_rejects_count_difference_even_with_same_spectral_gates(monkeypatch):
    bridge = load_bridge()
    checked = []
    monkeypatch.setattr(bridge, "_check_outcome", lambda outcome, protocol: checked.append(outcome))
    expected = {"relay_events": 3, "gates": {"gamma": True}}
    actual = {"relay_events": 4, "gates": {"gamma": True}}
    assert bridge.exact_replay(expected, expected.copy(), {})
    assert not bridge.exact_replay(expected, actual, {})
    assert len(checked) == 4


def test_unsealed_replay_is_rejected(tmp_path):
    bridge = load_bridge()
    seal = tmp_path / "seal.yaml"
    seal.write_text("status: unsealed\n")
    with pytest.raises(ValueError, match="not sealed"):
        bridge.verify_bridge_seal(seal)
