"""Synthetic header bounds and strict value-path whitelist."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('notebook_keys', scripts/'read_patchseq_notebook_keys.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def test_only_key_cells_read_and_duplicates_preserved(tmp_path, monkeypatch):
    h5py = pytest.importorskip('h5py')
    with h5py.File(tmp_path/'fixture.h5', 'w') as file:
        path = '/general/labnotebook/device/numericalKeys'
        file[path] = np.array([[b'Hold', b'Hold'], [b'pA', b'pA'], [b'raw', b'raw']], dtype='S4')
        file['general/labnotebook/device/numericalValues'] = np.ones((3, 2))
        original = h5py.Dataset.__getitem__
        def guard(node, key):
            assert node.name == path and len(key) == 2
            return original(node, key)
        monkeypatch.setattr(h5py.Dataset, '__getitem__', guard)
        budget = [0]
        result = module.read_header(file, path, h5py, budget)
        assert [v['decoded_text'] for v in result['rows'][0]] == ['Hold', 'Hold']
        assert module.read_header(file, '/general/labnotebook/device/numericalValues', h5py, budget)['undefined'] == 'unregistered-value-path'


@pytest.mark.parametrize('kind', ['shape', 'filter', 'soft', 'numeric'])
def test_bad_storage_rejected_before_values(tmp_path, monkeypatch, kind):
    h5py = pytest.importorskip('h5py')
    with h5py.File(tmp_path/'fixture.h5', 'w') as file:
        path = '/general/labnotebook/device/textualKeys'
        if kind == 'soft':
            file[path] = h5py.SoftLink('/never-read')
        else:
            file.create_dataset(path, shape=(3, 513 if kind == 'shape' else 2),
                                dtype='float64' if kind == 'numeric' else 'S2',
                                compression='gzip' if kind == 'filter' else None)
        monkeypatch.setattr(h5py.Dataset, '__getitem__', lambda *args: pytest.fail('forbidden read'))
        assert 'undefined' in module.read_header(file, path, h5py, [0])


def test_decoded_cell_and_file_budgets(tmp_path):
    h5py = pytest.importorskip('h5py')
    with h5py.File(tmp_path/'fixture.h5', 'w') as file:
        path = '/general/labnotebook/device/numericalKeys'
        file[path] = np.full((3, 1), b'x'*513, dtype='S513')
        assert 'undefined' in module.read_header(file, path, h5py, [0])
        del file[path]
        file[path] = np.full((3, 1), b'xx', dtype='S2')
        assert 'undefined' in module.read_header(file, path, h5py, [65535])
