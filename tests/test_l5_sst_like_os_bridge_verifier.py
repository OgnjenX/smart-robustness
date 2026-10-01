"""Reject falsely reported OS compatibility and changed replay protocols."""

from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest


def load_verifier():
    path = Path(__file__).parents[1] / "scripts/verify_l5_sst_like_stage4b_os_bridge.py"
    spec = importlib.util.spec_from_file_location("sst_os_bridge_verifier", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


@pytest.mark.parametrize("change", ["spikes", "events"])
def test_verifier_detects_false_exact_replay(monkeypatch, change):
    verifier = load_verifier()
    checked = []
    monkeypatch.setattr(
        verifier, "_check_outcome", lambda outcome, protocol: checked.append(outcome)
    )
    expected = {"spikes": [1.0, 2.0], "events": 2}
    result = {
        "outcome": deepcopy(expected),
        "status": "completed-os-bridge-replay",
        "exact_replay": True,
        "continuation_authorized": False,
        "frozen_baseline_modified": False,
        "protocol": {},
    }
    assert verifier.compare_outcomes(result, expected, {})["exact_replay"]
    result["outcome"][change] = [1.0, 2.001] if change == "spikes" else 3
    with pytest.raises(ValueError, match="contradicts"):
        verifier.compare_outcomes(result, expected, {})
    result.update(status="engineering-stop-replay-disagreement", exact_replay=False)
    assert not verifier.compare_outcomes(result, expected, {})["exact_replay"]
    assert len(checked) == 6


def test_verifier_forbids_replay_self_promotion():
    verifier = load_verifier()
    with pytest.raises(ValueError, match="authorize"):
        verifier.compare_outcomes({"continuation_authorized": True}, {}, {})


def test_verifier_rejects_changed_protocol():
    verifier = load_verifier()
    result = {
        "continuation_authorized": False,
        "frozen_baseline_modified": False,
        "protocol": {"dt_ms": 0.02},
    }
    with pytest.raises(ValueError, match="protocol"):
        verifier.compare_outcomes(result, {}, {"dt_ms": 0.01})
