"""Optional compiler provenance and replay controls; no scientific data used."""

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

scripts = Path(__file__).parents[1]/'scripts'
for name in ('patchseq_hh_independent_replay', 'patchseq_hh_compiled_prototype',
             'patchseq_hh_compiled_driver'):
    spec = importlib.util.spec_from_file_location(name, scripts/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
mod = sys.modules['patchseq_hh_compiled_driver']
pytestmark = pytest.mark.skipif(shutil.which('clang++') is None, reason='optional compiler unavailable')


def params():
    return {'capacitance_pF': 100., 'leak_nS': 10., 'leak_reversal_mV': -65.,
            'sodium_nS': 0., 'potassium_nS': 0., 'm_current_nS': 0.,
            'threshold_shift_mV': -60., 'm_tau_max_ms': 1000.}


def test_retained_binary_and_provenance_resume(tmp_path):
    with mod.compiled_driver(tmp_path/'build') as (replay, receipt):
        v = replay([params()], np.full(100, 30.), 50000, 20, .02)
        np.testing.assert_allclose(v[0], -60-5*np.exp(-(.02+np.arange(100)/50000)/.01), rtol=0, atol=1e-9)
        assert receipt['kernel_sha256'] == mod.KERNEL_SHA256
        assert receipt['binary_sha256'] == mod.digest(receipt['binary_path'])
        assert '-fno-fast-math' in receipt['flags'] and '-ffp-contract=off' in receipt['flags']
    with mod.compiled_driver(tmp_path/'build') as (_, resumed):
        assert resumed == receipt


def test_partial_build_not_overwritten(tmp_path):
    (tmp_path/'kernel.so').write_bytes(b'partial')
    with pytest.raises(ValueError, match='partial'):
        mod.prepare_build(tmp_path)
    assert (tmp_path/'kernel.so').read_bytes() == b'partial'


def test_changed_binary_rejected_without_rebuild(tmp_path):
    library, _ = mod.prepare_build(tmp_path)
    library.write_bytes(b'changed')
    with pytest.raises(ValueError, match='provenance'):
        mod.prepare_build(tmp_path)
    assert library.read_bytes() == b'changed'


def test_unknown_kernel_rejected_before_build(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, 'KERNEL_SHA256', 'not-the-tested-source')
    with pytest.raises(ValueError, match='source'):
        mod.prepare_build(tmp_path/'uncreated')
    assert not (tmp_path/'uncreated').exists()
