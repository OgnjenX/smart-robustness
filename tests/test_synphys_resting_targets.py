"""Rested-state coverage never substitutes nominal holding or picks records."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("resting_targets", SCRIPTS / "extract_synphys_resting_targets.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    return [{"ic_amp": 0.001, "ic_latency": 0.001, "ic_rise_time": 0.002,
             "ic_decay_tau": 0.02, "ic_nrmse": 0.1}], [{"avg_baseline_potential": -0.067}]


def test_finite_measured_baseline_required():
    r, b = fixture()
    assert module.coverage(r, b, True)
    assert not module.coverage(r, [], True)
    assert not module.coverage(r, b * 2, True)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -0.2, 0.1])
def test_invalid_baseline_not_replaced(value):
    r, b = fixture()
    b[0]["avg_baseline_potential"] = value
    assert not module.coverage(r, b, True)


def test_no_best_fit_selection_or_sign_repair():
    r, b = fixture()
    assert not module.coverage(r * 2, b, True)
    assert not module.coverage(r, b, False)
