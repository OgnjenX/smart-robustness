"""Pinned source object and literal header validation never interprets data rows."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/acquire_patchseq_methods_metadata_schema.py'
spec = importlib.util.spec_from_file_location('methods_header', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path, raw=b'ID, confidence\nTHIS IS NOT A VALID DATA ROW\n', status='200'):
    body, headers = tmp_path/'metadata.csv', tmp_path/'headers.txt'
    body.write_bytes(raw)
    headers.write_text(f'HTTP/2 {status}\ncontent-type: text/plain\n\n')
    reg = {'expected_bytes': len(raw), 'expected_git_blob_sha1': hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest(),
           'maximum_header_bytes': 65536, 'maximum_columns': 512}
    return body, headers, reg


def test_header_only_and_literal_spaces(tmp_path):
    body, headers, reg = fixture(tmp_path)
    schema = module.inspect(body, headers, reg)
    assert schema['columns'] == ['ID', ' confidence']
    assert not schema['data_rows_interpreted']


@pytest.mark.parametrize('raw', [b'a,a\n', b'a,\n', b'a,b', b'"unclosed,b\n', b'\xff,b\n'])
def test_invalid_header_rejected(tmp_path, raw):
    body, headers, reg = fixture(tmp_path, raw)
    with pytest.raises((ValueError, UnicodeError, module.csv.Error)):
        module.inspect(body, headers, reg)


@pytest.mark.parametrize('mutation', ['bytes', 'git', 'columns', 'header_bytes', 'status'])
def test_bound_or_identity_failure(tmp_path, mutation):
    body, headers, reg = fixture(tmp_path)
    if mutation == 'bytes':
        reg['expected_bytes'] += 1
    elif mutation == 'git':
        reg['expected_git_blob_sha1'] = '0'*40
    elif mutation == 'columns':
        reg['maximum_columns'] = 1
    elif mutation == 'header_bytes':
        reg['maximum_header_bytes'] = 2
    else:
        headers.write_text('HTTP/2 302\n\n')
    with pytest.raises(ValueError):
        module.inspect(body, headers, reg)
