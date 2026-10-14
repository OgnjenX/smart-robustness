"""Independent metadata guard and mismatch tests on synthetic files only."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from test_patchseq_exporter_provenance import module as collector

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('independent_exporter', scripts/'verify_patchseq_exporter_provenance.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def test_independent_values_and_layout_match_with_no_forbidden_reads(tmp_path, monkeypatch):
    h5py = pytest.importorskip('h5py')
    source = tmp_path/'fixture.h5'
    with h5py.File(source, 'w') as file:
        file['general/generated_by'] = np.array([[b'MIES', b'example']], dtype='S8')
        file['general/source_script'] = 'never read code'
        file['general/source_script'].attrs['file_name'] = 'source-name'
        file['general/labnotebook/device/values'] = np.zeros((10, 3))
        file['general/labnotebook/device/soft'] = h5py.SoftLink('/never-open')
    original = h5py.Dataset.__getitem__
    def guard(node, index):
        assert node.name == '/general/generated_by' and len(index) == 2
        return original(node, index)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', guard)
    with h5py.File(source, 'r') as file:
        expected = collector.inspect(file, h5py)
        actual = module.reconstruct(file, h5py)
    assert expected == actual
    actual['generated_by']['rows'][0][1]['decoded_text'] = 'wrong'
    assert actual != expected


def test_strict_success_checker_refuses_missing_or_nonhard_source(tmp_path):
    h5py = pytest.importorskip('h5py')
    with h5py.File(tmp_path/'fixture.h5', 'w') as file:
        file['general/generated_by'] = h5py.ExternalLink('/never-open.h5', '/')
        with pytest.raises(ValueError, match='non-hard'):
            module.reconstruct(file, h5py)


@pytest.mark.parametrize('value', [b'x'*4097, 5, np.arange(3)])
def test_independent_text_bounds(value):
    with pytest.raises(ValueError):
        module.decode(value)
