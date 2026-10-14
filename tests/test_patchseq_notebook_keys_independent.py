"""Header corruptions reject without notebook-value access."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from test_patchseq_notebook_keys_continuation import module as collector

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('independent_keys', scripts/'verify_patchseq_notebook_keys.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


@pytest.mark.parametrize('mutation', [None, 'unit', 'filter', 'path', 'column'])
def test_independent_reconstruction_rejects_corrupted_headers(tmp_path, monkeypatch, mutation):
    h5py = pytest.importorskip('h5py')
    with h5py.File(tmp_path/'fixture.h5', 'w') as file:
        path = '/general/labnotebook/device/numericalKeys'
        file.create_dataset(path, data=np.array([[b'Hold'], [b'pA'], [b'raw']], dtype='S4'), compression='gzip', shuffle=True)
        original = h5py.Dataset.__getitem__
        def guard(node, key):
            assert node.name == path and len(key) == 2
            return original(node, key)
        monkeypatch.setattr(h5py.Dataset, '__getitem__', guard)
        header = copy.deepcopy(collector.read(file, path, h5py, [0]))
        if mutation == 'unit':
            header['rows'][1][0]['decoded_text'] = 'A'
        elif mutation == 'filter':
            header['filter_IDs'] = [32000]
        elif mutation == 'path':
            header['path'] = path.replace('numericalKeys', 'numericalValues')
        elif mutation == 'column':
            header['rows'][0].append(header['rows'][0][0])
        if mutation is None:
            assert module.verify(file, header, h5py) == 9
        else:
            with pytest.raises(ValueError):
                module.verify(file, header, h5py)
