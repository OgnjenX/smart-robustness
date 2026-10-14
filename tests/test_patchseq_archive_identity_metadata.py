"""Metadata-only acquisition retains unknowns and cannot silently replay."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

path = Path(__file__).resolve().parents[1] / 'scripts/collect_patchseq_archive_identity_metadata.py'
spec = importlib.util.spec_from_file_location('archive_identity', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def reg():
    return yaml.safe_load(module.REG.read_text())


@pytest.mark.parametrize('raw', [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'[]'])
def test_invalid_json_rejected(raw, reg):
    with pytest.raises((ValueError, TypeError)):
        module.load_json(raw, reg['limits'])


def test_byte_and_shape_bounds(reg):
    limits = {**reg['limits'], 'maximum_response_bytes': 2}
    with pytest.raises(ValueError):
        module.load_json(b'{} ', limits)
    limits = {**reg['limits'], 'maximum_json_depth': 1}
    with pytest.raises(ValueError):
        module.load_json(b'{"a":{"b":1}}', limits)


def test_missing_null_multiple_and_unapproved_fields(reg):
    source = {'contentUrl': ['https://example.org/never-fetch'], 'path': None,
              'wasAttributedTo': [{'identifier': '123', 'age': 'unapproved'}, {'identifier': '456'}],
              'digest': {'dandi:dandi-etag': 'abc'}, 'metadata': {'wasGeneratedBy': [{'name': 'session'}]}}
    projected = module.project(source, reg)
    assert projected['top_level']['fields']['path'] == {'present': True, 'type': 'NoneType', 'value': None}
    assert projected['top_level']['fields']['asset_id'] == {'present': False}
    assert len(projected['top_level']['participants']['entries']) == 2
    assert 'age' not in str(projected)
    assert 'contentUrl' not in str(projected)
    assert projected['top_level']['fields']['digest']['algorithm_values']['dandi:dandi-etag']['value'] == 'abc'


def test_projection_bounds(reg):
    with pytest.raises(ValueError):
        module.project({'path': 'x'*4097}, reg)
    with pytest.raises(ValueError):
        module.project({'wasAttributedTo': [{}]*101}, reg)


def test_scalar_actor_not_coerced_to_list(reg):
    projected = module.project({'wasAttributedTo': '123'}, reg)
    assert projected['top_level']['participants'] == {'present': True, 'type': 'str', 'value': '123'}


@pytest.mark.parametrize('bad', ['../other', 'not-a-uuid', 'F7AF39A7-3837-4635-BA76-9C96643405D0'])
def test_url_injection_and_noncanonical_ids_rejected(bad, reg):
    with pytest.raises(ValueError):
        module.endpoint({'asset_id': bad}, reg)


@pytest.mark.parametrize('http_status', ['200', '302', '503'])
def test_single_request_receipt_and_checkpoint(monkeypatch, tmp_path, reg, http_status):
    asset = {'asset_id': 'f7af39a7-3837-4635-ba76-9c96643405d0', 'path': 'p', 'size': 123}
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        Path(argv[argv.index('--output')+1]).write_bytes(b'{"path":"p"}')
        Path(argv[argv.index('--dump-header')+1]).write_text(f'HTTP/2 {http_status}\ncontent-type: application/json\n\n')
        return SimpleNamespace(returncode=0, stdout=http_status, stderr='')

    monkeypatch.setattr(module.subprocess, 'run', fake_run)
    directory = tmp_path/'record'
    receipt = module.acquire(directory, asset, reg, 'collector')
    assert len(calls) == 1
    assert '--location' not in calls[0] and '--retry' not in calls[0]
    assert '/versions/0.210913.1639/assets/' in calls[0][-1]
    assert receipt['status'].startswith('archive-metadata' if http_status == '200' else 'acquisition-failure')
    assert module.inspect_checkpoint(directory, asset, reg, 'collector') == receipt
    assert len(calls) == 1
    (directory/'response.json').write_bytes(b'changed')
    with pytest.raises(ValueError):
        module.inspect_checkpoint(directory, asset, reg, 'collector')


def test_checkpoint_source_change_rejected(monkeypatch, tmp_path, reg):
    asset = {'asset_id': 'f7af39a7-3837-4635-ba76-9c96643405d0', 'path': 'p', 'size': 123}
    def unavailable(*args, **kwargs):
        raise OSError('offline')
    monkeypatch.setattr(module.subprocess, 'run', unavailable)
    module.acquire(tmp_path/'record', asset, reg, 'collector')
    with pytest.raises(ValueError):
        module.inspect_checkpoint(tmp_path/'record', {**asset, 'size': 124}, reg, 'collector')
