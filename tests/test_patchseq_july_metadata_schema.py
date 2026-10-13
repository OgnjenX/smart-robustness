"""Conditional July object identity is checked without interpreting rows."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/acquire_patchseq_july_metadata.py'
spec = importlib.util.spec_from_file_location('july_schema', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path, data=b'cell_specimen_id,structure\nmalformed\n'):
    body = tmp_path / 'metadata.csv'
    body.write_bytes(data)
    headers = tmp_path / 'headers.txt'
    headers.write_text(f'HTTP/2 200\netag: "object"\nx-amz-version-id: version\ncontent-length: {len(data)}\n\n')
    reg = {'expected_bytes': len(data), 'expected_etag': '"object"', 'expected_version_id': 'version'}
    return body, headers, reg


def test_only_header_interpreted(tmp_path):
    result = module.validate(*fixture(tmp_path))
    assert result['columns'] == ['cell_specimen_id', 'structure']
    assert 'rows' not in result


@pytest.mark.parametrize('fault', ['size', 'etag', 'version', 'length', 'status', 'encoding', 'duplicate-etag'])
def test_changed_http_identity_rejected(tmp_path, fault):
    body, headers, reg = fixture(tmp_path)
    content = headers.read_text()
    if fault == 'size':
        reg['expected_bytes'] += 1
    elif fault == 'etag':
        content = content.replace('"object"', '"changed"')
    elif fault == 'version':
        content = content.replace('version\n', 'changed\n')
    elif fault == 'length':
        content = content.replace(f'content-length: {reg["expected_bytes"]}', 'content-length: 0')
    elif fault == 'status':
        content = content.replace('200', '412')
    elif fault == 'encoding':
        content = content.rstrip() + '\ncontent-encoding: gzip\n\n'
    else:
        content = content.rstrip() + '\netag: "object"\n\n'
    headers.write_text(content)
    with pytest.raises(ValueError):
        module.validate(body, headers, reg)


@pytest.mark.parametrize('data', [b'a,a\n', b'a,\n', b'a,b', b'a'*65537 + b'\n'])
def test_invalid_csv_header_rejected(tmp_path, data):
    with pytest.raises(ValueError):
        module.validate(*fixture(tmp_path, data))
