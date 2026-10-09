"""Bounded numeric decoding rejects unsafe or ambiguous pulse metadata."""

from __future__ import annotations

import importlib.util
import io
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("pulse_storage", SCRIPTS / "audit_synphys_pulse_storage.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def packed(value):
    buf = io.BytesIO()
    np.save(buf, value)
    return buf.getvalue()


@pytest.mark.parametrize("shape", [(3,), (1, 3)])
def test_source_numeric_wrappers_retained(shape):
    result = module.decode_ids(packed(np.array([1, 2, 3], dtype=np.int64).reshape(shape)))
    assert result["ids"] == [1, 2, 3]
    assert result["source_shape"] == list(shape)


@pytest.mark.parametrize("value", [np.array([1, 1]), np.array([0, 1]), np.array([-1, 2]),
                                  np.array([1.0]), np.array([[1], [2]]), np.array([1], dtype=object), np.array([], dtype=int)])
def test_unregistered_or_ambiguous_arrays_reject(value):
    with pytest.raises(ValueError):
        module.decode_ids(packed(value))


@pytest.mark.parametrize("raw", [None, b"", b"not-npy", b"x" * 1048577])
def test_invalid_blobs_reject(raw):
    with pytest.raises((ValueError, EOFError)):
        module.decode_ids(raw)


def test_trailing_data_reject():
    with pytest.raises(ValueError):
        module.decode_ids(packed(np.array([1, 2])) + b"extra")


def test_explicit_id_count_with_registered_guard():
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE pulse_response (id INTEGER, data BLOB)")
        db.set_authorizer(module.authorizer)
        assert sum(1 for _ in db.execute("SELECT id FROM pulse_response")) == 0
        with pytest.raises(sqlite3.DatabaseError):
            db.execute("SELECT data FROM pulse_response")
