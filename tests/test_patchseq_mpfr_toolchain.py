"""Synthetic source-archive safeguards only; no dependency downloads/builds."""

import io
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from build_patchseq_mpfr_toolchain import tree_sha, unpack_checked


def archive(path, name, payload=b'fixed source', link=None):
    with tarfile.open(path, 'w:xz') as handle:
        info = tarfile.TarInfo(name)
        if link is None:
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
        else:
            info.type = tarfile.SYMTYPE
            info.linkname = link
            handle.addfile(info)


def test_exact_package_extract_and_digest_detects_change(tmp_path):
    path, output = tmp_path/'source.tar.xz', tmp_path/'extracted'
    archive(path, 'package-1/source.c')
    unpack_checked(path, output, 'package-1')
    source = output/'package-1'
    first = tree_sha(source)
    (source/'source.c').write_bytes(b'changed')
    assert tree_sha(source) != first
    with pytest.raises(ValueError, match='overwrite'):
        unpack_checked(path, output, 'package-1')


@pytest.mark.parametrize('name', ['/absolute', '../escape', 'package-1/../../escape',
                                  'other-package/source.c'])
def test_path_escape_or_wrong_package_rejected(tmp_path, name):
    path = tmp_path/'source.tar.xz'
    archive(path, name)
    with pytest.raises(ValueError, match='outside fixed'):
        unpack_checked(path, tmp_path/'out', 'package-1')


def test_external_symlink_rejected_by_data_filter(tmp_path):
    path = tmp_path/'source.tar.xz'
    archive(path, 'package-1/link', link='/etc/passwd')
    with pytest.raises(tarfile.FilterError):
        unpack_checked(path, tmp_path/'out', 'package-1')
