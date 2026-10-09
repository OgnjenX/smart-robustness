"""Independent blob reconstruction from synthetic archives."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC = importlib.util.spec_from_file_location(
        "physiology_array_verifier", SCRIPTS / "verify_sst_vip_physiology_arrays.py"
    )
    MODULE = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.remove(str(SCRIPTS))


def archive(tmp_path, array):
    meta = {"dtype": array.dtype.str, "shape": list(array.shape)}
    key = hashlib.sha256(
        json.dumps(meta, sort_keys=True).encode() + b"\0" + array.tobytes()
    ).hexdigest()
    path = tmp_path / (key + ".npz")
    np.savez_compressed(path, values=array)
    return {
        "content_sha256": key,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "metadata": meta,
        "stored_bytes": path.stat().st_size,
        "uncompressed_bytes": array.nbytes,
    }


@pytest.mark.parametrize(
    "array",
    [np.array(1.0), np.array([]), np.array([0.0, -0.0, np.nan]), np.array([1.0, 2.0], dtype=">f8")],
)
def test_independent_bitwise_roundtrip(tmp_path, array):
    restored = MODULE.read_array(tmp_path, archive(tmp_path, array))
    assert restored.tobytes() == array.tobytes() and restored.shape == array.shape


def test_corrupt_metadata_and_content_rejected(tmp_path):
    receipt = archive(tmp_path, np.array([1.0, 2.0]))
    receipt["metadata"]["shape"] = [1, 2]
    with pytest.raises(ValueError, match="shape differs"):
        MODULE.read_array(tmp_path, receipt)
    receipt["metadata"]["shape"] = [2]
    path = tmp_path / (receipt["content_sha256"] + ".npz")
    np.savez_compressed(path, values=np.array([9.0, 2.0]))
    receipt["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="content digest"):
        MODULE.read_array(tmp_path, receipt)
