"""Synthetic positive-pulse ratios and missing-evidence/selection controls."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('recording_quality', scripts/'measure_patchseq_recording_quality.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def pulses():
    v, i = np.zeros(4000), np.zeros(4000)
    for start, stop in ((100, 200), (1000, 2000)):
        v[start:stop], i[start:stop] = .005, 50e-12
        i[start] = 500e-12
    return v, i, 10000.


def test_source_indices_stable_and_peak_resistance():
    out = module.measure('breakin', *pulses())
    assert out['status'] == 'measured'
    assert out['input_MOhm'] == pytest.approx(100.)
    assert out['access_MOhm'] == pytest.approx(10.)
    assert out['pulses'][0]['up_index'] == 999
    assert out['pulses'][0]['peak_index'] == 1000
    assert module.measure('seal', *pulses())['seal_GOhm'] == pytest.approx(.1)


@pytest.mark.parametrize('kind', ['negative', 'zero', 'nan', 'only-test-pulse', 'short-window'])
def test_unsupported_measurements_unknown_not_default(kind):
    v, i, rate = pulses()
    if kind == 'negative':
        v, i = -v, -i
    elif kind == 'zero':
        i[:] = 0
    elif kind == 'nan':
        i[0] = np.nan
    elif kind == 'only-test-pulse':
        v[1000:], i[1000:] = 0, 0
    else:
        rate = 100
    assert module.measure('breakin', v, i, rate)['status'] == 'unknown'


def good_trials():
    return [{'protocol': 'bath', 'sweep_number': 1, 'measurement': {'status': 'measured', 'electrode_zero_pA': 0.}},
            {'protocol': 'seal', 'sweep_number': 2, 'measurement': {'status': 'measured', 'seal_GOhm': 2.}},
            {'protocol': 'breakin', 'sweep_number': 3, 'measurement': {'status': 'measured', 'input_MOhm': 100., 'access_MOhm': 10.}}]


def test_missing_blowout_is_unassessed_and_later_invalid_trial_not_replaced():
    trials = good_trials()
    out = module.score(trials)
    assert out['mandatory_measurement_gate'] == 'pass' and out['complete_cell_health_established'] is False
    assert out['blowout_state'].startswith('unassessed')
    trials.append({'protocol': 'breakin', 'sweep_number': 4, 'measurement': {'status': 'unknown', 'reason': 'bad'}})
    assert module.score(trials)['mandatory_measurement_gate'] == 'unknown'
    assert module.score(trials)['selected_sweeps']['breakin'] == 4


def test_required_measurement_missing_or_threshold_fail():
    assert module.score(good_trials()[:2])['mandatory_measurement_gate'] == 'unknown'
    trials = good_trials()
    trials[2]['measurement']['access_MOhm'] = 21.
    assert module.score(trials)['mandatory_measurement_gate'] == 'fail'
