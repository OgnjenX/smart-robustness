"""Independent receipt audit rejects corruption; format checks remain bounded."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def verifier(monkeypatch):
    name = "independent_physiology_verifier_test"
    spec = importlib.util.spec_from_file_location(
        name, Path("scripts/verify_sst_vip_physiology_receipts.py")
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def write_bytes(path, raw):
    with path.open("xb") as handle:
        handle.write(raw)


def test_hdf_signature_does_not_claim_nwb_validity(verifier, tmp_path):
    p = tmp_path / "1.nwb"
    raw = b"\x89HDF\r\n\x1a\n"
    write_bytes(p, raw)
    record = {
        "raw_path": str(p),
        "raw_bytes": len(raw),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "status": "complete-receipt-awaiting-format-validation",
        "preflight_record": {
            "content_length": len(raw),
            "source": {"resource": {"well_known_file_type": {"name": "NWBDownload"}}},
        },
    }
    result = verifier.verify_receipt(record, p)
    assert result["hdf5_signature_verified"] and not result["nwb_schema_validated"]
    record["raw_sha256"] = "wrong"
    with pytest.raises(ValueError, match="hash"):
        verifier.verify_receipt(record, p)


def test_rooted_swc(verifier, tmp_path):
    p = tmp_path / "1.swc"
    write_bytes(p, b"1 1 0 0 0 1 -1\n2 3 1 0 0 0.5 1\n")
    result = verifier.swc_inventory(p)
    assert result["nodes"] == 2 and result["roots"] == [1]
    assert not result["physiological_morphology_validated"]


def test_allen_zero_based_node_ids(verifier, tmp_path):
    p = tmp_path / "1.swc"
    write_bytes(p, b"0 1 0 0 0 1 -1\n1 3 1 0 0 0.5 0\n")
    result = verifier.swc_inventory(p)
    assert result["nodes"] == 2 and result["roots"] == [0]


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"1 1 0 0 0 1 -1\n1 1 0 0 0 1 -1\n",
        b"1 1 nan 0 0 1 -1\n",
        b"1 1 0 0 0 -1 -1\n",
        b"1 1 0 0 0 1 -1\n2 3 0 0 0 1 99\n",
        b"1 1 0 0 0 1 -1\n2 3 0 0 0 1 3\n3 3 0 0 0 1 2\n",
    ],
)
def test_invalid_swc_rejected(verifier, tmp_path, raw):
    p = tmp_path / "1.swc"
    write_bytes(p, raw)
    with pytest.raises(ValueError):
        verifier.swc_inventory(p)
