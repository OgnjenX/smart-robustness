"""Synthetic independent scorer parity and failure controls."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from smart_robustness.validation.physiology_recordings import experimental_events, score_group

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC = importlib.util.spec_from_file_location(
        "independent_physiology_scoring", SCRIPTS / "verify_sst_vip_physiology_scoring.py"
    )
    MODULE = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.remove(str(SCRIPTS))


def recording():
    response = np.full(2000, -0.06)
    response[[200, 400, 600, 800]] = 0.03
    return {
        "response": response,
        "rate": 10000.0,
        "start": 0,
        "count": 2000,
        "stimulus_sha256": "synthetic-fixed-stimulus",
    }


def test_independent_passing_group_matches_registered_implementation():
    r = recording()
    events = experimental_events(r)
    actual = MODULE.group_score([r, r], [events, events], [True, True])
    assert actual == score_group([r, r], [events, events], [True, True])
    assert actual["passed"]


@pytest.mark.parametrize("scenario", ["singleton", "silent", "bad-parity", "spike-mismatch"])
def test_independent_failed_and_inconclusive_groups(scenario):
    r = recording()
    repeats, parity = [r, r], [True, True]
    predictions = [experimental_events(r), experimental_events(r)]
    if scenario == "singleton":
        repeats, predictions, parity = repeats[:1], predictions[:1], parity[:1]
    elif scenario == "silent":
        r["response"][:] = -0.06
        predictions = [[], []]
    elif scenario == "bad-parity":
        parity[1] = False
    else:
        predictions = [[], []]
    actual = MODULE.group_score(repeats, predictions, parity)
    assert actual == score_group(repeats, predictions, parity)
    assert not actual["passed"]


def test_epoch_clipping_and_nonfinite_prediction():
    r = recording()
    r.update(start=200, count=500)
    np.testing.assert_array_equal(MODULE.within_epoch([0.019, 0.02, 0.069, 0.07], r), [0.02, 0.069])
    with pytest.raises(ValueError, match="finite"):
        MODULE.group_score([r, r], [[np.nan], []], [True, True])
