"""Partial receipts remain evidence; complete receipt is not model validation."""

from __future__ import annotations

import hashlib
import io
from http.client import IncompleteRead

import pytest

from smart_robustness.validation.sst_vip_physiology_download import receive_file


def test_exact_receipt(tmp_path):
    p = tmp_path / "cell.nwb"
    result = receive_file(io.BytesIO(b"abcd"), p, expected_bytes=4, maximum_bytes=10, chunk_bytes=2)
    assert result["status"] == "complete-receipt-awaiting-format-validation"
    assert p.read_bytes() == b"abcd" and result["raw_bytes"] == 4
    assert result["raw_sha256"] == hashlib.sha256(b"abcd").hexdigest()


@pytest.mark.parametrize("data", [b"ab", b"abcde", b"abcdefghijklmnop"])
def test_short_or_oversized_receipt_preserved(tmp_path, data):
    p = tmp_path / "cell.nwb"
    result = receive_file(io.BytesIO(data), p, expected_bytes=4, maximum_bytes=10, chunk_bytes=2)
    assert result["status"] == "failed"
    assert p.read_bytes() == data[:5]
    assert result["raw_bytes"] <= 5


def test_existing_file_not_overwritten(tmp_path):
    p = tmp_path / "cell.nwb"
    with p.open("xb") as f:
        f.write(b"original")
    with pytest.raises(FileExistsError):
        receive_file(io.BytesIO(b"abcd"), p, expected_bytes=4, maximum_bytes=10)
    assert p.read_bytes() == b"original"


def test_incomplete_read_preserves_bounded_partial(tmp_path):
    class Broken:
        def read(self, n):
            raise IncompleteRead(b"ab", 4)

    p = tmp_path / "cell.nwb"
    result = receive_file(Broken(), p, expected_bytes=4, maximum_bytes=10)
    assert result["status"] == "failed" and result["error_type"] == "IncompleteRead"
    assert p.read_bytes() == b"ab"


@pytest.mark.parametrize("expected,maximum,chunk", [(0, 10, 2), (11, 10, 2), (4, 10, 0)])
def test_invalid_limits_rejected_before_write(tmp_path, expected, maximum, chunk):
    p = tmp_path / "cell.nwb"
    with pytest.raises(ValueError):
        receive_file(
            io.BytesIO(b"abcd"),
            p,
            expected_bytes=expected,
            maximum_bytes=maximum,
            chunk_bytes=chunk,
        )
    assert not p.exists()
