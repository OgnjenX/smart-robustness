"""Acquisition byte bounds and hashes must hold without full-object allocation."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("medium_acquisition", SCRIPTS / "acquire_synphys_medium_schema.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("length", [0, 9, 10, 11, 100])
def test_bounded_retained_bytes_and_hash(length):
    raw = b"x" * length
    output = io.BytesIO()
    count, checksum = module.stream_object(io.BytesIO(raw), output, 10, chunk_bytes=3)
    assert count == min(length, 11)
    assert output.getvalue() == raw[:11]
    assert checksum == hashlib.sha256(raw[:11]).hexdigest()


def test_stream_never_requests_unbounded_read():
    class Guarded(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 4
            return super().read(size)
    result = module.stream_object(Guarded(b"abcdefghij"), io.BytesIO(), 10, chunk_bytes=4)
    assert result[0] == 10
