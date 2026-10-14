"""Independent metadata traversal guards all waveform/reference-vector access."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('independent_trace_schema', scripts/'verify_patchseq_trace_schema.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def test_stack_reader_only_reads_whitelisted_scalar_metadata(tmp_path, monkeypatch):
    h5py = pytest.importorskip('h5py')
    source = tmp_path/'fixture.nwb'
    with h5py.File(source, 'w') as file:
        group = file.create_group('acquisition/a')
        group.attrs['neurodata_type'] = 'CurrentClampSeries'
        group.attrs['outside_whitelist'] = 'never-read'
        group['data'] = [1, 2, 3]
        group['timestamps'] = [0, 1, 2]
        group['starting_time'] = 0.0
        group['data'].attrs['resolution'] = float('nan')
        file['acquisition/a-/data'] = [1, 2]
        file['stimulus/presentation/b/starting_time'] = 0.0
    original = h5py.Dataset.__getitem__
    values = []
    def guarded(node, key):
        assert node.name.endswith('/starting_time')
        values.append(node.name)
        return original(node, key)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', guarded)
    attr_read = h5py.AttributeManager.__getitem__
    def attrs(node, key):
        assert key != 'outside_whitelist'
        return attr_read(node, key)
    monkeypatch.setattr(h5py.AttributeManager, '__getitem__', attrs)
    records, reads = [], {'attributes': [], 'datasets': []}
    with h5py.File(source, 'r') as file:
        module.reconstruct(file, yaml.safe_load(module.REG.read_text())['scope'], h5py, records, reads)
    assert reads['datasets'] == values == ['/acquisition/a/starting_time', '/stimulus/presentation/b/starting_time']
    assert [r['path'] for r in records].index('/acquisition/a/data') < [r['path'] for r in records].index('/acquisition/a-')
    assert not any('never-read' in str(r) for r in records)


def test_independent_reader_never_follows_nonhard_links_or_vector_scalars(tmp_path, monkeypatch):
    h5py = pytest.importorskip('h5py')
    source = tmp_path/'fixture.nwb'
    with h5py.File(source, 'w') as file:
        file['acquisition/external'] = h5py.ExternalLink('/never-open.h5', '/')
        file['acquisition/soft'] = h5py.SoftLink('/general')
        file['acquisition/a/starting_time'] = [1, 2, 3]
    monkeypatch.setattr(h5py.Dataset, '__getitem__', lambda *args: pytest.fail('vector read forbidden'))
    records, reads = [], {'attributes': [], 'datasets': []}
    with h5py.File(source, 'r') as file:
        module.reconstruct(file, yaml.safe_load(module.REG.read_text())['scope'], h5py, records, reads)
    assert not reads['datasets']
    assert sum(r.get('unsupported') == 'non-hard-link-never-followed' for r in records) == 2


@pytest.mark.parametrize(('raw', 'undefined'), [(float('nan'), 'nan'), (float('inf'), 'positive-infinity'),
                                              (float('-inf'), 'negative-infinity')])
def test_independent_nonfinite_metadata_is_not_a_default(raw, undefined):
    assert module.decode(raw, 4096) == {'raw_type': 'float', 'undefined': undefined}


def test_independent_raw_byte_text_and_bounds():
    assert module.decode(b'volts', 4096) == {'raw_type': 'bytes', 'raw_hex': '766f6c7473', 'decoded_text': 'volts'}
    assert module.decode(b'long', 3)['undefined'] == 'decoded-size-limit-exceeded'
    assert module.decode(b'\xff', 4096)['undefined'] == 'non-UTF8'
