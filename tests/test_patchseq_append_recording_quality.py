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
    spec = importlib.util.spec_from_file_location('recording_quality', scripts/'measure_patchseq_append_recording_quality.py')
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


def test_unsupported_metadata_never_reads_arrays_or_charges_budget(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('unsupported health pair arrays were read')
    monkeypatch.setattr(module, 'read_array', forbidden)
    pair = {'protocol': 'bath', 'metadata_pair_pass': False, 'failures': ['series-metadata-failure']}
    budget = [0]
    result = module.measure_pair(None, pair, {}, None, budget, 1024)
    assert result == {'status': 'unknown', 'reason': 'unsupported-health-pair-metadata',
                      'failures': ['series-metadata-failure']}
    assert budget == [0]


def test_unsupported_latest_trial_is_not_removed_from_selection():
    trials = good_trials()
    unknown = {'status': 'unknown', 'reason': 'unsupported-health-pair-metadata', 'failures': ['series-metadata-failure']}
    trials.append({'protocol': 'bath', 'sweep_number': 4, 'measurement': unknown})
    result = module.score(trials)
    assert result['mandatory_measurement_gate'] == 'unknown'
    assert result['selected_sweeps']['bath'] == 4
    trials[-1]['sweep_number'] = 0
    assert module.score(trials)['selected_sweeps']['bath'] == 1
    assert module.score(trials)['mandatory_measurement_gate'] == 'pass'


def test_supported_pair_uses_exact_conversions_and_original_measurement(monkeypatch):
    pair = {'protocol': 'bath', 'metadata_pair_pass': True, 'series_paths': ['r', 's']}
    series = {'r': {'role': 'response', 'path': 'r', 'shape': [1000], 'conversion': 2e-12, 'rate': 10000.},
              's': {'role': 'stimulus', 'path': 's', 'shape': [1000], 'conversion': .001, 'rate': 10000.}}
    calls = []
    def decoder(file, item, h5py, budget, limit):
        calls.append(item['path'])
        budget[0] += 8000
        return np.ones(1000)
    monkeypatch.setattr(module, 'read_array', decoder)
    budget = [0]
    result = module.measure_pair(None, pair, series, None, budget, 20000)
    assert result['electrode_zero_pA'] == pytest.approx(2.)
    assert calls == ['r', 's'] and budget == [16000]
