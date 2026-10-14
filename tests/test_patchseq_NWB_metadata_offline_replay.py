"""Offline HDF5 replay has no gap filling, network fallback or trace access."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import pytest
import yaml

path = Path(__file__).resolve().parents[1]/'scripts/verify_patchseq_NWB_metadata_pilot.py'
spec = importlib.util.spec_from_file_location('offline_pilot', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_saved_ranges_exact_cross_span_read_and_no_gap_fill():
    with module.SavedRanges(12, [(0, b'1234'), (4, b'5678')]) as file:
        file.seek(2)
        data = bytearray(5)
        assert file.readinto(data) == 5 and data == b'34567'
        assert file.read_spans == [[2, 7]]
        file.seek(8)
        with pytest.raises(ValueError, match='uncovered'):
            file.readinto(bytearray(1))
        with pytest.raises(io.UnsupportedOperation):
            file.write(b'x')


@pytest.mark.parametrize('ranges', [[(0, b'1234'), (3, b'x')], [(10, b'abc')], [(0, b'')]])
def test_invalid_cached_spans_rejected(ranges):
    with pytest.raises(ValueError):
        module.SavedRanges(12, ranges)


def test_literal_scalar_types_preserved_and_bad_shape_rejected():
    assert module.reconstruct_scalar(b'123', 4) == {'raw_type': 'bytes', 'raw_hex': '313233', 'decoded_text': '123'}
    assert module.reconstruct_scalar('123', 4) == {'raw_type': 'str', 'value': '123'}
    assert module.reconstruct_scalar(123, 4) == {'raw_type': 'int', 'value': 123}
    with pytest.raises(ValueError):
        module.reconstruct_scalar(b'12345', 4)
    with pytest.raises(ValueError):
        module.reconstruct_scalar(float('nan'), 4)


def test_offline_whitelist_only_with_no_full_file_network_driver(tmp_path):
    h5py = pytest.importorskip('h5py')
    source = tmp_path/'fixture.nwb'
    with h5py.File(source, 'w') as file:
        file.attrs['nwb_version'] = '2.2.5'
        file['general/subject/subject_id'] = '123'
        file['identifier'] = [b'opaque']
        file['acquisition/unapproved/data'] = [1, 2, 3]
    reg = yaml.safe_load(module.REG.read_text())
    data = source.read_bytes()
    with module.SavedRanges(len(data), [(0, data)]) as saved:
        result = module.replay(saved, reg['logical_metadata_reads'], h5py)
    assert result['logical_dataset_reads'] == ['/identifier', '/general/subject/subject_id']
    assert result['datasets']['/session_id'] == {'present': False}


@pytest.mark.parametrize('kind', ['soft_parent', 'external_parent', 'array'])
def test_forbidden_metadata_dereference_stops_before_read(tmp_path, kind):
    h5py = pytest.importorskip('h5py')
    source = tmp_path/'fixture.nwb'
    with h5py.File(source, 'w') as file:
        if kind == 'soft_parent':
            file['general'] = h5py.SoftLink('/acquisition')
        elif kind == 'external_parent':
            file['general'] = h5py.ExternalLink('/never-open.h5', '/')
        else:
            file['identifier'] = [1, 2]
    reg = yaml.safe_load(module.REG.read_text())
    data = source.read_bytes()
    with module.SavedRanges(len(data), [(0, data)]) as saved, pytest.raises((TypeError, ValueError)):
        module.replay(saved, reg['logical_metadata_reads'], h5py)


def test_changed_boolean_type_not_equal():
    with pytest.raises(ValueError):
        module.equal({'present': True}, {'present': 1})
