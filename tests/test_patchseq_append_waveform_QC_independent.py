"""Synthetic arithmetic equivalence and corruption controls, not biological fits."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from test_patchseq_append_waveform_QC import module as collector
from test_patchseq_append_waveform_QC import step

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('independent_wave_QC', scripts/'verify_patchseq_append_waveform_QC.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


@pytest.mark.parametrize('seed', range(10))
@pytest.mark.parametrize('noise', [0., 0.03, 0.15, 0.6])
def test_independent_fsum_matches_synthetic_classifications(seed, noise):
    v, i, rate = step()
    v += np.random.default_rng(seed).normal(0, noise, len(v))
    if seed % 2:
        v[-5001:] += 1.2
    module.compare(collector.qc(v, i, rate), module.reconstruct(v, i, rate))


@pytest.mark.parametrize('mutation', ['nan', 'ramp', 'early', 'short', 'truncated', 'test-pulse', 'infinite-rate'])
def test_independent_nonpasses_and_test_pulse_match(mutation):
    v, i, rate = step()
    if mutation == 'nan':
        v[10] = np.nan
    elif mutation == 'ramp':
        i[10000:20000] = np.linspace(20, 120, 10000)
    elif mutation == 'early':
        i[1000:10000] = 120
    elif mutation == 'short':
        i[14000:] = 20
    elif mutation == 'truncated':
        v, i = v[:22000], i[:22000]
    elif mutation == 'test-pulse':
        i[1000:1100] = -20
    else:
        rate = np.inf
    module.compare(collector.qc(v, i, rate), module.reconstruct(v, i, rate))


@pytest.mark.parametrize('mutation', ['metric', 'pass', 'failure', 'index', 'missing'])
def test_corrupted_outcomes_rejected(mutation):
    truth = module.reconstruct(*step())
    saved = copy.deepcopy(truth)
    if mutation == 'metric':
        saved['metrics']['pre_noise_RMS_mV'] += 1e-6
    elif mutation == 'pass':
        saved['sweep_QC_pass'] = False
    elif mutation == 'failure':
        saved['failures'].append('invented')
    elif mutation == 'index':
        saved['step_start_index'] += 1
    else:
        del saved['metrics']
    with pytest.raises(ValueError):
        module.compare(saved, truth)


def test_numeric_tolerance_does_not_relax_discrete_gate():
    module.compare(0.07+1e-12, 0.07)
    with pytest.raises(ValueError, match='discrete'):
        module.compare(True, False)
    with pytest.raises(ValueError, match='type'):
        module.compare(1, True)


def test_independent_reader_rejects_soft_links_and_undefined_instrument(tmp_path, monkeypatch):
    h5py = pytest.importorskip('h5py')
    source = tmp_path/'fixture.h5'
    with h5py.File(source, 'w') as file:
        file['series/bias_current'] = np.nan
        file['series/gain'] = h5py.SoftLink('/never-read')
    original = h5py.Dataset.__getitem__
    def guard(node, key):
        assert node.name == '/series/bias_current'
        return original(node, key)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', guard)
    with h5py.File(source, 'r') as file:
        values = module.scalar_metadata(file, '/series', h5py)
        assert values['bias_current']['undefined'] == 'nonfinite'
        assert values['gain']['undefined'] == 'non-hard-or-missing-path'
        assert values['bridge_balance']['present'] is False
