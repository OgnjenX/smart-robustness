"""Immutable file-manifest schema checks do not interpret source rows."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/acquire_patchseq_file_manifest.py'
spec = importlib.util.spec_from_file_location('manifest_schema', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path, data):
    path = tmp_path / 'data.csv'
    path.write_bytes(data)
    blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\x00' + data).hexdigest()
    return path, len(data), blob


def test_only_header_interpreted(tmp_path):
    args = fixture(tmp_path, b'cell_specimen_id,archive_uri,technique\nmalformed,row\n')
    result = module.inspect_object(*args)
    assert result['columns'] == ['cell_specimen_id', 'archive_uri', 'technique']
    assert 'rows' not in result


@pytest.mark.parametrize('data', [b'a,a\n', b'a,\n', b'a,b', b'a' * 65537 + b'\n'])
def test_invalid_header_rejected(tmp_path, data):
    with pytest.raises(ValueError):
        module.inspect_object(*fixture(tmp_path, data))


@pytest.mark.parametrize('fault', ['size', 'blob'])
def test_changed_object_rejected(tmp_path, fault):
    path, size, blob = fixture(tmp_path, b'a,b\n1,2\n')
    with pytest.raises(ValueError):
        module.inspect_object(path, size + 1 if fault == 'size' else size,
                              '0' * 40 if fault == 'blob' else blob)
