"""Source-index fsum equivalence and exact gate corruption rejection."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from test_patchseq_append_recording_quality import good_trials, pulses
from test_patchseq_append_recording_quality import module as collector

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('independent_quality', scripts/'verify_patchseq_append_recording_quality.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


@pytest.mark.parametrize('protocol', ['bath', 'seal', 'breakin'])
@pytest.mark.parametrize('seed', range(5))
def test_independent_means_ratios_and_peak_indices(protocol, seed):
    v, i, rate = pulses()
    i += np.random.default_rng(seed).normal(0, 1e-14, len(i))
    module.compare(collector.measure(protocol, v, i, rate), module.reconstruct(protocol, v, i, rate))


def test_latest_invalid_trial_not_replaced_and_missing_blowout_explicit():
    trials = good_trials()
    module.compare(collector.score(trials), module.gate(trials))
    trials.append({'protocol': 'seal', 'sweep_number': 20, 'measurement': {'status': 'unknown'}})
    module.compare(collector.score(trials), module.gate(trials))
    assert module.gate(trials)['mandatory_measurement_gate'] == 'unknown'


@pytest.mark.parametrize('mutation', ['value', 'index', 'missing', 'gate'])
def test_corruption_controls(mutation):
    actual = module.reconstruct('breakin', *pulses())
    saved = copy.deepcopy(actual)
    if mutation == 'value':
        saved['access_MOhm'] += .001
    elif mutation == 'index':
        saved['pulses'][0]['peak_index'] += 1
    elif mutation == 'missing':
        del saved['pulses']
    else:
        saved, actual = {'mandatory_measurement_gate': 'pass'}, {'mandatory_measurement_gate': 'unknown'}
    with pytest.raises(ValueError):
        module.compare(saved, actual)
