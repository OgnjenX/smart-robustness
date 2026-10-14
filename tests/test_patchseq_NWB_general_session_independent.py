"""Independent scalar reader is narrow, type preserving and link safe."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('independent_session', scripts/'verify_patchseq_NWB_general_session.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def test_independent_reader_only_accesses_registered_scalar(tmp_path, monkeypatch):
    h5py = pytest.importorskip('h5py')
    source = tmp_path/'fixture.nwb'
    with h5py.File(source, 'w') as file:
        file['general/session_id'] = '123'
        file['acquisition/trace'] = [1, 2, 3]
        file['general/subject/description'] = 'unregistered'
    original = h5py.Dataset.__getitem__
    reads = []
    def guarded(node, key):
        reads.append(node.name)
        assert node.name == '/general/session_id'
        return original(node, key)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', guarded)
    with h5py.File(source, 'r') as file:
        record = module.independent_scalar(file, h5py)
    assert reads == ['/general/session_id']
    assert record['scalar'] == {'raw_type': 'bytes', 'raw_hex': '313233', 'decoded_text': '123'}


@pytest.mark.parametrize('case', ['soft', 'external', 'array', 'numeric'])
def test_independent_reader_rejects_unsupported_identity(tmp_path, case):
    h5py = pytest.importorskip('h5py')
    source = tmp_path/'fixture.nwb'
    with h5py.File(source, 'w') as file:
        if case == 'soft':
            file['general'] = h5py.SoftLink('/acquisition')
        elif case == 'external':
            file['general'] = h5py.ExternalLink('/never-open.h5', '/')
        else:
            file['general/session_id'] = [1, 2] if case == 'array' else 123
    with h5py.File(source, 'r') as file, pytest.raises((ValueError, TypeError)):
        module.independent_scalar(file, h5py)
