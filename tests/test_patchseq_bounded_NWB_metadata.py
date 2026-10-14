"""Bounded backing I/O cannot read unapproved logical arrays or follow links."""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

scripts = Path(__file__).resolve().parents[1]/'scripts'
spec = importlib.util.spec_from_file_location('patchseq_bounded_NWB', scripts/'patchseq_bounded_NWB.py')
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)
pilot_spec = importlib.util.spec_from_file_location('metadata_pilot', scripts/'run_patchseq_NWB_metadata_pilot.py')
pilot = importlib.util.module_from_spec(pilot_spec)
pilot_spec.loader.exec_module(pilot)


@pytest.fixture
def reg():
    return yaml.safe_load(pilot.REG.read_text())


def test_range_cache_cross_block_seek_and_readinto():
    raw, calls = b'0123456789abcdef', []
    limits = {'cache_block_bytes': 4, 'maximum_unique_downloaded_bytes': 12, 'maximum_range_requests': 3}
    def fetch(start, end):
        calls.append((start, end))
        return raw[start:end+1]
    with helper.BoundedReader(len(raw), limits, fetch) as reader:
        reader.seek(2)
        assert reader.read(5) == b'23456'
        reader.seek(0)
        buffer = bytearray(4)
        assert reader.readinto(buffer) == 4 and buffer == b'0123'
        assert calls == [(0, 3), (4, 7)]
        assert reader.downloaded == 8
        reader.seek(-4, 2)
        assert reader.read(4) == b'cdef'
        reader.seek(8)
        with pytest.raises(ValueError, match='budget'):
            reader.read(1)


def test_unbounded_or_writable_access_rejected():
    reader = helper.BoundedReader(100, {'cache_block_bytes': 4, 'maximum_unique_downloaded_bytes': 8, 'maximum_range_requests': 2}, lambda a, b: b'')
    with pytest.raises(ValueError):
        reader.read()
    with pytest.raises(io.UnsupportedOperation):
        reader.write(b'x')
    with pytest.raises(io.UnsupportedOperation):
        reader.truncate()
    with pytest.raises(ValueError):
        reader.seek(-1)


def test_cache_reads_still_check_deadline():
    def expired():
        raise TimeoutError('deadline')
    reader = helper.BoundedReader(4, {'cache_block_bytes': 4, 'maximum_unique_downloaded_bytes': 4, 'maximum_range_requests': 1}, lambda a, b: b'1234', expired)
    with pytest.raises(TimeoutError):
        reader.read(1)


@pytest.mark.parametrize('url', ['http://dandiarchive.s3.amazonaws.com/blobs/x', 'https://evil.example/blobs/x',
    'https://user:secret@dandiarchive.s3.amazonaws.com/blobs/x', 'https://dandiarchive.s3.amazonaws.com:8080/blobs/x',
    'https://dandiarchive.s3.amazonaws.com/blobs/../other', 'https://dandiarchive.s3.amazonaws.com/blobs/%2e%2e/other'])
def test_unsafe_locators_rejected(reg, url):
    assert not helper.permitted_url(url, reg['URL_resolution'])


def test_ambiguous_locators_not_first_selected(reg):
    raw = {'contentUrl': ['https://dandiarchive.s3.amazonaws.com/blobs/a', 'https://dandiarchive.s3.amazonaws.com/blobs/b']}
    assert len(helper.url_candidates(raw, reg['URL_resolution'])) == 2
    with pytest.raises(ValueError):
        helper.select_url(raw, reg['URL_resolution'])


def test_whitelisted_metadata_only(reg, tmp_path):
    h5py = pytest.importorskip('h5py')
    path = tmp_path/'fixture.nwb'
    with h5py.File(path, 'w') as file:
        file.attrs['nwb_version'] = '2.0.0'
        file['session_id'] = '456'
        file['general/subject/subject_id'] = '123'
        file['acquisition/never-read/data'] = [1, 2, 3]
    with h5py.File(path, 'r') as file:
        report = helper.inspect_metadata(file, reg['logical_metadata_reads'], h5py)
    assert report['logical_dataset_reads'] == ['/session_id', '/general/subject/subject_id']
    assert report['datasets']['/general/subject/specimen_id']['present'] is False


@pytest.mark.parametrize('kind', ['soft', 'external', 'array', 'compressed'])
def test_unapproved_metadata_storage_rejected(reg, tmp_path, kind):
    h5py = pytest.importorskip('h5py')
    path = tmp_path/'fixture.nwb'
    with h5py.File(path, 'w') as file:
        if kind == 'soft':
            file['session_id'] = h5py.SoftLink('/acquisition/never-read')
        elif kind == 'external':
            file['general'] = h5py.ExternalLink('/never-open.h5', '/')
        elif kind == 'array':
            file['session_id'] = [1, 2]
        else:
            file.create_dataset('session_id', data=[123], compression='gzip')
    with h5py.File(path, 'r') as file, pytest.raises((ValueError, TypeError)):
        helper.inspect_metadata(file, reg['logical_metadata_reads'], h5py)


@pytest.mark.parametrize('changed', ['etag', 'range', 'status', None])
def test_transport_immutable_range_and_no_automatic_redirects(tmp_path, monkeypatch, reg, changed):
    client = pilot.Transport(tmp_path, 'https://dandiarchive.s3.amazonaws.com/blobs/a', {'size': 100}, reg)
    client.etag = '"fixed"'
    calls = []
    def fake_run(argv, **kwargs):
        calls.append(argv)
        tag = '"changed"' if changed == 'etag' else '"fixed"'
        span = 'bytes 1-4/100' if changed == 'range' else 'bytes 0-3/100'
        status = 200 if changed == 'status' else 206
        Path(argv[argv.index('--dump-header')+1]).write_text(f'HTTP/2 {status}\nETag: {tag}\nContent-Range: {span}\n\n')
        Path(argv[argv.index('--output')+1]).write_bytes(b'1234')
        return SimpleNamespace(returncode=0, stdout=str(status), stderr='')
    monkeypatch.setattr(pilot.subprocess, 'run', fake_run)
    if changed:
        with pytest.raises(ValueError):
            client.fetch(0, 3)
    else:
        assert client.fetch(0, 3) == b'1234'
    assert len(calls) == 1
    assert calls[0][1] == '--disable' and '--location' not in calls[0]
    assert 'If-Match: "fixed"' in calls[0]
    assert len(client.records) == 1 and 'body_sha256' in client.records[0]
