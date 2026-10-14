"""Synthetic arrays test sealed QC and enforce bounded HDF5 access."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('patchseq_qc', scripts/'run_patchseq_append_waveform_QC.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def step():
    v = np.full(30000, -65.)
    i = np.full(30000, 20.)
    i[10000:20000] = 120.
    return v, i, 10000.


def test_clean_step_preserves_nonzero_command_bias():
    result = module.qc(*step())
    assert result['sweep_QC_pass'] is True
    assert result['command_baseline_pA'] == 20.
    assert result['command_step_pA'] == 120.
    assert result['command_delta_pA'] == 100.
    assert result['step_duration_s'] == 1.


@pytest.mark.parametrize('kind', ['nan', 'inf', 'drift', 'noise', 'ramp', 'truncated', 'short'])
def test_invalid_sweeps_not_promoted(kind):
    v, i, rate = step()
    if kind == 'nan':
        i[50] = np.nan
    elif kind == 'inf':
        v[50] = np.inf
    elif kind == 'drift':
        v[-5001:] += 1.01
    elif kind == 'noise':
        v[4999:9999] += np.arange(5000)%2
    elif kind == 'ramp':
        i[10000:20000] = np.linspace(20, 120, 10000)
    elif kind == 'truncated':
        v, i = v[:23000], i[:23000]
    else:
        i[14000:] = 20
    assert module.qc(v, i, rate)['sweep_QC_pass'] is False


def test_early_test_pulse_allowed_but_not_in_baseline_window():
    v, i, rate = step()
    i[1000:1100] = -20.
    assert module.qc(v, i, rate)['sweep_QC_pass'] is True
    i[1000:1100] = 20
    i[6000:6100] = -20.
    assert module.qc(v, i, rate)['sweep_QC_pass'] is False


def test_nonhard_path_and_unregistered_dataset_not_read(tmp_path, monkeypatch):
    h5py = pytest.importorskip('h5py')
    source = tmp_path/'synthetic.h5'
    with h5py.File(source, 'w') as file:
        file['series/data'] = np.arange(10.)
        file['outside'] = np.arange(5.)
        file['soft'] = h5py.SoftLink('/series')
        file['external'] = h5py.ExternalLink('/never-open.h5', '/')
    original = h5py.Dataset.__getitem__
    def guarded(node, key):
        assert node.name == '/series/data'
        return original(node, key)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', guarded)
    with h5py.File(source, 'r') as file:
        budget = [0]
        values = module.read_array(file, {'path': '/series', 'data_shape': [10]}, h5py, budget, 80)
        assert np.array_equal(values, np.arange(10.)) and budget == [80]
        with pytest.raises(ValueError, match='budget'):
            module.read_array(file, {'path': '/series', 'data_shape': [10]}, h5py, budget, 80)
        for path in ('/soft/data', '/external/data'):
            with pytest.raises(ValueError, match='non-hard'):
                module.hard_node(file, path, h5py)


def test_instrument_nan_not_imputed_and_soft_link_not_followed(tmp_path):
    h5py = pytest.importorskip('h5py')
    source = tmp_path/'synthetic.h5'
    with h5py.File(source, 'w') as file:
        file['series/bias_current'] = np.nan
        file['series/gain'] = h5py.SoftLink('/no-such-value')
    with h5py.File(source, 'r') as file:
        out = module.instruments(file, '/series', h5py)
        assert out['bias_current']['undefined'] == 'nonfinite'
        assert 'value' not in out['bias_current']
        assert out['gain']['undefined'] == 'non-hard-or-missing-path'
        assert out['bridge_balance']['present'] is False


def test_virtual_external_oversized_and_boolean_storage_rejected_before_read(tmp_path, monkeypatch):
    h5py = pytest.importorskip('h5py')
    source = tmp_path/'synthetic.h5'
    with h5py.File(source, 'w', libver='latest') as file:
        layout = h5py.VirtualLayout(shape=(10,), dtype='float64')
        layout[:] = h5py.VirtualSource('/never-open.h5', 'data', shape=(10,))
        file.create_virtual_dataset('virtual/data', layout)
        file.create_dataset('external/data', shape=(10,), dtype='float64', external=[('never-open.bin', 0, 80)])
        file.create_dataset('oversized/data', shape=(10000001,), dtype='float32', chunks=(100,))
        file['boolean/data'] = [True, False]
    monkeypatch.setattr(h5py.Dataset, '__getitem__', lambda *args: pytest.fail('forbidden array read'))
    with h5py.File(source, 'r') as file:
        for path, size in [('virtual', 10), ('external', 10), ('oversized', 10000001), ('boolean', 2)]:
            with pytest.raises(ValueError):
                module.read_array(file, {'path': '/'+path, 'data_shape': [size]}, h5py, [0], 2**34)
