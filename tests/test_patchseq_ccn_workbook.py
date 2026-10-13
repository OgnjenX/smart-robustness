"""Nested reference payload extraction is exact, bounded and nonexecuting."""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/inspect_patchseq_ccn_workbook.py'
spec = importlib.util.spec_from_file_location('ccn_workbook', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path, name='reference.xlsx', symlink=False):
    path = tmp_path / 'source.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        member = zipfile.ZipInfo(name)
        if symlink:
            member.create_system = 3
            member.external_attr = 0o120777 << 16
        archive.writestr(member, b'unchanged-source-bytes')
    return path


def test_exact_member_bytes_preserved(tmp_path):
    path = fixture(tmp_path)
    before = module.digest(path)
    assert module.source_member(path, 'reference.xlsx', 22) == b'unchanged-source-bytes'
    assert module.digest(path) == before


@pytest.mark.parametrize('fault', ['absent', 'size', 'symlink', 'directory'])
def test_unsupported_member_rejected(tmp_path, fault):
    path = fixture(tmp_path, name='reference.xlsx/' if fault == 'directory' else 'reference.xlsx',
                   symlink=fault == 'symlink')
    with pytest.raises(ValueError):
        module.source_member(path, 'absent.xlsx' if fault == 'absent' else 'reference.xlsx',
                             23 if fault == 'size' else 22)
