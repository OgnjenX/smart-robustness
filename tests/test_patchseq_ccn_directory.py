"""Reference archive acquisition never interprets or executes member contents."""

from __future__ import annotations

import hashlib
import importlib.util
import zipfile
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/acquire_patchseq_ccn_directory.py'
spec = importlib.util.spec_from_file_location('ccn_directory', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path, name='taxonomy.csv', symlink=False):
    path = tmp_path / 'reference.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        if symlink:
            member = zipfile.ZipInfo(name)
            member.create_system = 3
            member.external_attr = (0o120777 << 16)
            archive.writestr(member, 'target')
        else:
            archive.writestr(name, 'not-an-interpreted-CSV-or-code')
    data = path.read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\x00' + data).hexdigest()
    return path, len(data), blob


def test_only_directory_inspected(tmp_path):
    result = module.inspect(*fixture(tmp_path))
    assert result['members'][0]['name'] == 'taxonomy.csv'
    assert 'not-an-interpreted' not in str(result)
    assert not result['member_contents_interpreted']
    assert not result['archive_extracted']


@pytest.mark.parametrize('name', ['../escape.csv', '/absolute.csv', 'folder\\escape.csv'])
def test_unsafe_directory_names_rejected(tmp_path, name):
    with pytest.raises(ValueError, match='safe archive'):
        module.inspect(*fixture(tmp_path, name))


def test_symlink_rejected(tmp_path):
    with pytest.raises(ValueError, match='safe archive'):
        module.inspect(*fixture(tmp_path, symlink=True))


@pytest.mark.parametrize('fault', ['size', 'blob'])
def test_source_identity_changes_rejected(tmp_path, fault):
    path, size, blob = fixture(tmp_path)
    with pytest.raises(ValueError):
        module.inspect(path, size + 1 if fault == 'size' else size, '0'*40 if fault == 'blob' else blob)
