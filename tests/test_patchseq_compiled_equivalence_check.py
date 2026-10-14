"""Synthetic equivalence checker retains mismatches and failures without retuning."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

scripts = Path(__file__).parents[1]/'scripts'
for name in ('verify_patchseq_whole_file_integrity', 'patchseq_hh_independent_replay',
             'patchseq_hh_compiled_prototype', 'check_patchseq_compiled_equivalence'):
    spec = importlib.util.spec_from_file_location(name, scripts/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
mod = sys.modules['check_patchseq_compiled_equivalence']


def test_matching_traces(monkeypatch):
    monkeypatch.setattr(mod, 'integrate', lambda *args: np.zeros((1, 2)))
    result = mod.compare_case(lambda *args: np.zeros((1, 2)), {'leak_reversal_mV': -65}, None, 50000)
    assert result['passed'] and result['maximum_voltage_difference_mV'] == 0


def test_mismatches_retained(monkeypatch):
    monkeypatch.setattr(mod, 'integrate', lambda *args: np.zeros((1, 2)))
    result = mod.compare_case(lambda *args: np.ones((1, 2)), {'leak_reversal_mV': -65}, None, 50000)
    assert not result['passed'] and result['maximum_voltage_difference_mV'] == 1


def test_failure_agreement_is_not_zero_voltage(monkeypatch):
    def fails(*args):
        raise ValueError('synthetic numerical failure')
    monkeypatch.setattr(mod, 'integrate', fails)
    result = mod.compare_case(fails, {'leak_reversal_mV': -65}, None, 50000)
    assert result['passed'] and result['maximum_voltage_difference_mV'] is None
    result = mod.compare_case(lambda *args: np.zeros((1, 2)), {'leak_reversal_mV': -65}, None, 50000)
    assert not result['passed']
