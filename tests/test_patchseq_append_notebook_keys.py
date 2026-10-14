"""Header reads cannot access trial values, waveforms or choose duplicate keys."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    from read_patchseq_append_notebook_keys import key_paths, read
    from verify_patchseq_append_notebook_keys import verify
finally:
    sys.path.pop(0)


@pytest.mark.parametrize('compression', [None, 'gzip'])
def test_only_header_cells_are_read_and_independently_verified(tmp_path, monkeypatch, compression):
    h5py = pytest.importorskip('h5py')
    path = '/general/labnotebook/device/numericalKeys'
    with h5py.File(tmp_path/'fixture.h5', 'w') as file:
        file.create_dataset(path, data=np.array([[b'I-Clamp Holding Level'], [b'pA'], [b'raw']], dtype='S32'),
                            compression=compression)
        file['general/labnotebook/device/numericalValues'] = np.ones((10, 1, 9))
        file['general/source_script'] = 'must-not-read'
        original = h5py.Dataset.__getitem__
        calls = []
        def guard(node, index):
            calls.append((node.name, index))
            assert node.name == path and len(index) == 2
            assert all(type(k) is int for k in index)
            return original(node, index)
        monkeypatch.setattr(h5py.Dataset, '__getitem__', guard)
        budget = [0]
        header = read(file, path, h5py, budget)
        assert verify(file, header, h5py) == budget[0]
        assert len(calls) == 6
        altered = copy.deepcopy(header)
        altered['rows'][1][0]['decoded_text'] = 'amperes'
        with pytest.raises(ValueError):
            verify(file, altered, h5py)


@pytest.mark.parametrize('case', ['missing', 'duplicate', 'undefined', 'extra'])
def test_header_scope_cannot_select_a_favorable_subset(case):
    layout = {'records': [{'path': '/n/numericalKeys', 'object_type': 'dataset'},
                          {'path': '/n/textualKeys', 'object_type': 'dataset'}]}
    if case == 'missing':
        layout['records'].pop()
    elif case == 'duplicate':
        layout['records'].append(copy.deepcopy(layout['records'][0]))
    elif case == 'undefined':
        layout['undefined'] = 'unsupported'
    else:
        layout['records'].append({'path': '/other/numericalKeys', 'object_type': 'dataset'})
    with pytest.raises(ValueError):
        key_paths(layout)


def test_unselected_notebook_values_are_not_paths():
    layout = {'records': [{'path': '/n/numericalKeys', 'object_type': 'dataset'},
                          {'path': '/n/textualKeys', 'object_type': 'dataset'},
                          {'path': '/n/numericalValues', 'object_type': 'dataset'}]}
    assert key_paths(layout) == ['/n/numericalKeys', '/n/textualKeys']


def test_plugin_and_external_links_fail_before_header_value_reads(tmp_path, monkeypatch):
    h5py = pytest.importorskip('h5py')
    with h5py.File(tmp_path/'fixture.h5', 'w') as file:
        path = '/general/labnotebook/device/numericalKeys'
        file.create_dataset(path, shape=(3, 1), dtype='S8', compression='lzf')
        monkeypatch.setattr(h5py.Dataset, '__getitem__', lambda *a: pytest.fail('forbidden value read'))
        with pytest.raises(ValueError):
            read(file, path, h5py, [0])
        file['general/labnotebook/device/textualKeys'] = h5py.ExternalLink('/never-open', '/')
        with pytest.raises(ValueError):
            read(file, '/general/labnotebook/device/textualKeys', h5py, [0])
