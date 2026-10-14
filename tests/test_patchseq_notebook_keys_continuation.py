"""Explicit standard-filter whitelist, not arbitrary plugin decompression."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('key_continuation', scripts/'continue_patchseq_notebook_keys.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def test_shuffle_deflate_header_read_only(tmp_path, monkeypatch):
    h5py = pytest.importorskip('h5py')
    with h5py.File(tmp_path/'fixture.h5', 'w') as file:
        path = '/general/labnotebook/device/numericalKeys'
        file.create_dataset(path, data=np.array([[b'Holding'], [b'pA'], [b'raw']], dtype='S7'), compression='gzip', shuffle=True)
        file['general/labnotebook/device/numericalValues'] = np.ones((3, 1))
        original = h5py.Dataset.__getitem__
        def guard(node, key):
            assert node.name == path and len(key) == 2
            return original(node, key)
        monkeypatch.setattr(h5py.Dataset, '__getitem__', guard)
        out = module.read(file, path, h5py, [0])
        assert out['filter_IDs'] == [2, 1]
        assert out['rows'][1][0]['decoded_text'] == 'pA'
        with pytest.raises(ValueError, match='unregistered'):
            module.read(file, '/general/labnotebook/device/numericalValues', h5py, [0])


def test_nonwhitelisted_filter_and_soft_link_rejected_before_read(tmp_path, monkeypatch):
    h5py = pytest.importorskip('h5py')
    with h5py.File(tmp_path/'fixture.h5', 'w') as file:
        path = '/general/labnotebook/device/textualKeys'
        file.create_dataset(path, shape=(3, 1), dtype='S7', compression='lzf')
        file['general/labnotebook/device/numericalKeys'] = h5py.SoftLink('/never-read')
        monkeypatch.setattr(h5py.Dataset, '__getitem__', lambda *args: pytest.fail('forbidden read'))
        with pytest.raises(ValueError, match='plugin'):
            module.read(file, path, h5py, [0])
        with pytest.raises(ValueError, match='non-hard'):
            module.read(file, '/general/labnotebook/device/numericalKeys', h5py, [0])
