"""Only bounded provenance text values; notebook arrays and script code forbidden."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('exporter_provenance', scripts/'read_patchseq_append_exporter_provenance.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def test_only_provenance_text_cells_are_read(tmp_path, monkeypatch):
    h5py = pytest.importorskip('h5py')
    source = tmp_path/'fixture.h5'
    with h5py.File(source, 'w') as file:
        file['general/generated_by'] = np.array([[b'MIES', b'example-version'], [b'Igor', b'8']], dtype='S20')
        file['general/source_script'] = 'never execute or read this code'
        file['general/source_script'].attrs['file_name'] = 'example.ipf'
        file['general/labnotebook/device/numericalValues'] = np.ones((10, 3))
        file['general/labnotebook/device/numericalKeys'] = np.array([b'QC'], dtype='S2')
    original = h5py.Dataset.__getitem__
    def guarded(node, index):
        assert node.name == '/general/generated_by'
        assert len(index) == 2 and all(type(i) is int for i in index)
        return original(node, index)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', guarded)
    with h5py.File(source, 'r') as file:
        result = module.inspect(file, h5py)
    assert result['generated_by']['rows'][0][0]['decoded_text'] == 'MIES'
    assert result['source_script_file_name']['decoded_text'] == 'example.ipf'
    assert len(result['labnotebook_structure']['records']) == 4


@pytest.mark.parametrize('kind', ['soft', 'external', 'filtered', 'oversized-shape', 'numeric'])
def test_unsupported_table_rejected_without_value_reads(tmp_path, monkeypatch, kind):
    h5py = pytest.importorskip('h5py')
    source = tmp_path/'fixture.h5'
    with h5py.File(source, 'w') as file:
        if kind == 'soft':
            file['general/generated_by'] = h5py.SoftLink('/never-read')
        elif kind == 'external':
            file['general/generated_by'] = h5py.ExternalLink('/never-open.h5', '/')
        elif kind == 'numeric':
            file['general/generated_by'] = np.ones((2, 2))
        else:
            file.create_dataset('general/generated_by', shape=(33 if kind == 'oversized-shape' else 2, 2),
                                dtype='S2', compression='gzip' if kind == 'filtered' else None)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', lambda *args: pytest.fail('value read forbidden'))
    with h5py.File(source, 'r') as file:
        assert 'undefined' in module.inspect(file, h5py)['generated_by']


def test_bounded_text_preserves_nonUTF8_and_refuses_other_types():
    item, size = module.text(b'\xff')
    assert item['undefined'] == 'non-UTF8' and item['raw_hex'] == 'ff' and size == 1
    for value in [5, np.arange(3), b'x'*4097]:
        with pytest.raises(ValueError):
            module.text(value)


def test_notebook_soft_links_retained_not_followed(tmp_path, monkeypatch):
    h5py = pytest.importorskip('h5py')
    source = tmp_path/'fixture.h5'
    with h5py.File(source, 'w') as file:
        file['general/labnotebook/soft'] = h5py.SoftLink('/never-read')
    monkeypatch.setattr(h5py.Dataset, '__getitem__', lambda *args: pytest.fail('no dataset values'))
    with h5py.File(source, 'r') as file:
        result = module.inspect(file, h5py)
    assert result['labnotebook_structure']['records'][1]['undefined'] == 'non-hard-link-not-followed'
