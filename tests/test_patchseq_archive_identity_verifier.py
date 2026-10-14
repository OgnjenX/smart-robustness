"""Offline verifier independently retains uncertainty and rejects changed receipts."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest
import yaml

path = Path(__file__).resolve().parents[1] / 'scripts/verify_patchseq_archive_identity_metadata.py'
spec = importlib.util.spec_from_file_location('verify_archive', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def reg():
    return yaml.safe_load(module.REG.read_text())


def test_projection_golden_unknowns_and_approved_identity_only(reg):
    source = {'identifier': 'asset', 'wasAttributedTo': [{'identifier': '123', 'sex': 'unapproved'}],
              'digest': {'dandi:sha2-256': 'abc'}, 'metadata': None, 'contentUrl': 'never-fetch'}
    result = module.reconstruct(source, reg)
    assert result['metadata_container'] == {'present': True, 'type': 'NoneType'}
    assert result['top_level']['fields']['identifier'] == {'present': True, 'type': 'str', 'value': 'asset'}
    assert result['top_level']['participants']['entries'] == [{'identifier': {'present': True, 'type': 'str', 'value': '123'}, 'schemaKey': {'present': False}}]
    assert result['top_level']['fields']['digest']['algorithm_values'] == {'dandi:sha2-256': {'type': 'str', 'value': 'abc'}}
    assert 'unapproved' not in str(result) and 'contentUrl' not in str(result)


@pytest.mark.parametrize('raw', [b'{"x":1,"x":2}', b'{"x":Infinity}', b'{"x":1e999}', b'[]'])
def test_invalid_json_rejected(raw, reg):
    with pytest.raises((ValueError, TypeError)):
        module.decode(raw, reg['limits'])


def test_bound_rejections(reg):
    limits = {**reg['limits'], 'maximum_json_nodes': 1}
    with pytest.raises(ValueError):
        module.decode(b'{"x":1}', limits)
    with pytest.raises(ValueError):
        module.reconstruct({'wasAttributedTo': [None]*101}, reg)
    with pytest.raises(ValueError):
        module.reconstruct({'path': 'x'*4097}, reg)


@pytest.mark.parametrize('header', [b'HTTP/2 302\ncontent-type: application/json\n\n',
    b'HTTP/2 200\ncontent-type: text/html\n\n', b'HTTP/2 200\ncontent-type: application/json\ncontent-encoding: gzip\n\n'])
def test_redirect_or_incompatible_response_rejected(header):
    with pytest.raises(ValueError):
        module.http_headers(header)


def receipt_fixture(directory, reg, failure=False):
    asset = {'asset_id': 'f7af39a7-3837-4635-ba76-9c96643405d0', 'path': 'p', 'size': 123}
    attempt = {'asset': asset, 'endpoint': reg['endpoint_template'].format(asset_id=asset['asset_id']),
               'registration_sha256': module.REG_HASH, 'collector_sha256': module.COLLECTOR_HASH,
               'started_at_utc': 'sealed-time'}
    body = b'{"identifier":"asset"}'
    header = b'HTTP/2 200\nContent-Type: application/json; charset=utf-8\nETag: "abc"\n\n'
    (directory/'attempt.yaml').write_text(yaml.safe_dump(attempt))
    (directory/'response.json').write_bytes(body)
    (directory/'headers.txt').write_bytes(header)
    receipt = {**attempt, 'status': 'acquisition-failure-retained' if failure else 'archive-metadata-projected-independent-verification-pending',
               'client_exit_code': 0, 'http_status': '200', 'physiological_arrays_read': False,
               'response_sha256': module.sha(directory/'response.json'), 'headers_sha256': module.sha(directory/'headers.txt')}
    if failure:
        receipt['error'] = 'unexpected collector failure'
    else:
        receipt['headers'] = module.http_headers(header)
        receipt['projection'] = module.reconstruct({'identifier': 'asset'}, reg)
    (directory/'receipt.yaml').write_text(yaml.safe_dump(receipt))
    return asset, receipt


def test_complete_receipt(reg, tmp_path):
    asset, _ = receipt_fixture(tmp_path, reg)
    assert module.verify_receipt(tmp_path, asset, reg)['projection_independently_matched']


@pytest.mark.parametrize('mutation', ['raw', 'projection', 'attempt', 'scope', 'type'])
def test_tampering_rejected(reg, tmp_path, mutation):
    asset, receipt = receipt_fixture(tmp_path, reg)
    if mutation == 'raw':
        (tmp_path/'response.json').write_bytes(b'changed')
    elif mutation == 'projection':
        receipt['projection']['top_level']['fields']['identifier']['value'] = 'changed'
    elif mutation == 'attempt':
        attempt = copy.deepcopy(receipt)
        attempt['asset']['size'] = 124
        (tmp_path/'attempt.yaml').write_text(yaml.safe_dump(attempt))
    elif mutation == 'scope':
        receipt['physiological_arrays_read'] = True
    else:
        receipt['projection']['top_level']['fields']['identifier']['value'] = 123
    (tmp_path/'receipt.yaml').write_text(yaml.safe_dump(receipt))
    with pytest.raises(ValueError):
        module.verify_receipt(tmp_path, asset, reg)


def test_unjustified_failure_not_accepted(reg, tmp_path):
    asset, _ = receipt_fixture(tmp_path, reg, failure=True)
    with pytest.raises(ValueError, match='valid complete response'):
        module.verify_receipt(tmp_path, asset, reg)


def test_valid_failed_http_attempt_not_rerun(reg, tmp_path):
    asset, receipt = receipt_fixture(tmp_path, reg, failure=True)
    receipt['http_status'] = '503'
    (tmp_path/'receipt.yaml').write_text(yaml.safe_dump(receipt))
    assert not module.verify_receipt(tmp_path, asset, reg)['projection_independently_matched']


def test_type_preserving_match():
    with pytest.raises(ValueError):
        module.same({'flag': True}, {'flag': 1})
