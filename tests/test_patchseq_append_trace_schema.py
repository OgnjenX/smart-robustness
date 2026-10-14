"""Metadata inventory cannot read waveform, timestamp or reference vectors."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('append_trace_schema', scripts/'inventory_patchseq_append_trace_schema.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def test_only_registered_metadata_scalars_are_read(tmp_path, monkeypatch):
    h5py = pytest.importorskip('h5py')
    source = tmp_path/'fixture.nwb'
    with h5py.File(source, 'w') as file:
        series = file.create_group('acquisition/sweep_1')
        series.attrs['neurodata_type'] = 'CurrentClampSeries'
        series.attrs['namespace'] = 'core'
        series.attrs['sweep_number'] = 1
        series.attrs['clinical_unregistered'] = 'must-not-read'
        series['data'] = list(range(1000))
        series['data'].attrs['unit'] = 'volts'
        series['data'].attrs['conversion'] = 0.001
        series['data'].attrs['resolution'] = float('nan')
        series['timestamps'] = list(range(1000))
        series['starting_time'] = 12.5
        series['starting_time'].attrs['rate'] = 20000.0
        file['general/subject/description'] = 'outside-scope'
    original, accesses = h5py.Dataset.__getitem__, []
    def guarded(node, key):
        accesses.append(node.name)
        assert node.name == '/acquisition/sweep_1/starting_time'
        return original(node, key)
    attributes = h5py.AttributeManager.__getitem__
    def guarded_attribute(node, key):
        assert key != 'clinical_unregistered'
        return attributes(node, key)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', guarded)
    monkeypatch.setattr(h5py.AttributeManager, '__getitem__', guarded_attribute)
    policy = yaml.safe_load(module.REG.read_text())['scope']
    records, reads = [], {'attributes': [], 'datasets': []}
    with h5py.File(source, 'r') as file:
        module.inventory(file, policy, h5py, records, reads)
    assert accesses == reads['datasets'] == ['/acquisition/sweep_1/starting_time']
    data = next(r for r in records if r['path'].endswith('/data'))
    assert data['shape'] == [1000] and 'scalar_metadata' not in data
    assert data['attributes']['resolution']['scalar'] == {'raw_type': 'float', 'undefined': 'nan'}
    assert not any(r['path'].startswith('/general') for r in records)
    assert 'must-not-read' not in str(records)


def test_soft_external_and_unapproved_scalar_shapes_not_followed(tmp_path, monkeypatch):
    h5py = pytest.importorskip('h5py')
    source = tmp_path/'fixture.nwb'
    with h5py.File(source, 'w') as file:
        file['acquisition/soft'] = h5py.SoftLink('/general')
        file['acquisition/external'] = h5py.ExternalLink('/never-open.h5', '/')
        file['acquisition/series/starting_time'] = [1, 2, 3]
    monkeypatch.setattr(h5py.Dataset, '__getitem__', lambda *args: pytest.fail('unapproved dataset value read'))
    records, reads = [], {'attributes': [], 'datasets': []}
    with h5py.File(source, 'r') as file:
        module.inventory(file, yaml.safe_load(module.REG.read_text())['scope'], h5py, records, reads)
    assert not reads['datasets']
    assert sum(r.get('unsupported') == 'non-hard-link-never-followed' for r in records) == 2
    assert next(r for r in records if r['path'].endswith('/starting_time'))['scalar_metadata']['undefined']


@pytest.mark.parametrize(('value', 'marker'), [(float('inf'), 'positive-infinity'), (float('-inf'), 'negative-infinity'), (float('nan'), 'nan')])
def test_nonfinite_metadata_is_explicitly_undefined(value, marker):
    assert module.scalar_descriptor(value, 4096) == {'raw_type': 'float', 'undefined': marker}


def test_object_and_depth_limits_do_not_expand_after_overflow(tmp_path):
    h5py = pytest.importorskip('h5py')
    source = tmp_path/'fixture.nwb'
    with h5py.File(source, 'w') as file:
        file['acquisition/one/two/three/four/data'] = [1, 2]
    policy = yaml.safe_load(module.REG.read_text())['scope']
    records, reads = [], {'attributes': [], 'datasets': []}
    with h5py.File(source, 'r') as file:
        module.inventory(file, policy, h5py, records, reads)
    assert any(r.get('unsupported') == 'group-depth-bound' for r in records)
    policy['maximum_structural_objects_per_file'] = 1
    with h5py.File(source, 'r') as file, pytest.raises(ValueError, match='object bound'):
        module.inventory(file, policy, h5py, [], {'attributes': [], 'datasets': []})
