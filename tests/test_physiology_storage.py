"""Synthetic byte-preservation, corruption, publication and disk-guard controls."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from smart_robustness.validation.physiology_storage import ArrayStore, array_identity, require_space


def test_exact_duplicates_share_blob_not_attempt_receipts(tmp_path):
    array = np.array([0.0, -0.0, np.nan, np.inf])
    with ArrayStore(tmp_path, reserve_bytes=0) as store:
        first, second = store.put(array), store.put(array.copy())
        assert first == second
        assert store.get(first).tobytes() == array.tobytes()
        path = store.checkpoint("case-0000", {"native0": first, "native1": second})
        assert set(json.loads(path.read_text())) == {"native0", "native1"}
        assert len(list(tmp_path.glob("*.npz"))) == 1


def test_nan_payload_signed_zero_shape_dtype_distinguished():
    a = np.array([0x7FF8000000000001], dtype=np.uint64).view(np.float64)
    b = np.array([0x7FF8000000000002], dtype=np.uint64).view(np.float64)
    assert array_identity(a)[2] != array_identity(b)[2]
    assert array_identity(np.array([0.0]))[2] != array_identity(np.array([-0.0]))[2]
    assert array_identity(np.ones(2))[2] != array_identity(np.ones((1, 2)))[2]
    assert array_identity(np.ones(2, dtype="f4"))[2] != array_identity(np.ones(2, dtype="f8"))[2]


@pytest.mark.parametrize(
    "array",
    [
        np.array(1.0),
        np.array([]),
        np.zeros((0, 2)),
        np.arange(10)[::2],
        np.array([1.0, 2.0], dtype=">f8"),
    ],
)
def test_scalar_empty_strided_and_endian_roundtrip(tmp_path, array):
    with ArrayStore(tmp_path, reserve_bytes=0) as store:
        restored = store.get(store.put(array))
        assert restored.shape == array.shape and restored.dtype == array.dtype
        assert restored.tobytes() == array.tobytes()


def test_corruption_not_overwritten(tmp_path):
    with ArrayStore(tmp_path, reserve_bytes=0) as store:
        receipt = store.put([1.0, 2.0])
        path = store.path(receipt["content_sha256"])
        path.write_bytes(b"corrupt synthetic fixture")
        with pytest.raises(ValueError, match="file changed"):
            store.get(receipt)
        with pytest.raises((ValueError, OSError)):
            store.put([1.0, 2.0])
        assert path.read_bytes() == b"corrupt synthetic fixture"


def test_disk_guard_stops_before_new_blob(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "smart_robustness.validation.physiology_storage.shutil.disk_usage",
        lambda _: SimpleNamespace(free=100),
    )
    with (
        ArrayStore(tmp_path, reserve_bytes=100) as store,
        pytest.raises(OSError, match="insufficient space"),
    ):
        store.put(np.arange(10))
    assert not list(tmp_path.glob("*.npz")) and not list(tmp_path.glob("pending-*"))
    with pytest.raises(ValueError):
        require_space(tmp_path, -1, 0)


def test_lock_checkpoint_and_key_controls(tmp_path):
    store = ArrayStore(tmp_path, reserve_bytes=0)
    with pytest.raises(ValueError, match="context"):
        store.put([1.0])
    with store:
        with pytest.raises(OSError), ArrayStore(tmp_path, reserve_bytes=0):
            pass
        for key in ("../bad", "0" * 63):
            with pytest.raises(ValueError, match="key"):
                store.path(key)
        path = store.checkpoint("case-0000", {"context": "sealed"})
        assert store.checkpoint("case-0000", {"context": "sealed"}) == path
        with pytest.raises(ValueError, match="differs"):
            store.checkpoint("case-0000", {"context": "changed"})
        with pytest.raises(ValueError):
            store.put(np.array([{}], dtype=object))
