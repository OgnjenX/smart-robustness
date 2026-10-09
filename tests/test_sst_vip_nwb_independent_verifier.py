"""Independent verifier identifies source attachments without relying on filenames."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

h5py = pytest.importorskip("h5py")


@pytest.fixture
def verifier(monkeypatch):
    name = "independent_nwb_verifier_test"
    spec = importlib.util.spec_from_file_location(
        name, Path("scripts/verify_sst_vip_nwb_schema.py")
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_embedded_identifier(verifier):
    assert verifier.embedded_ephys_id("Allen Institute for Brain Science, Ephys Result 123") == 123
    with pytest.raises(ValueError, match="identifier"):
        verifier.embedded_ephys_id("123_ephys.nwb")


def test_independent_link_resolution(verifier, tmp_path):
    p = tmp_path / "links.h5"
    with h5py.File(p, "w") as root:
        root.create_dataset("value", data=7)
        root["alias"] = h5py.SoftLink("/value")
        root["external"] = h5py.ExternalLink("never-open.h5", "/data")
        root["cycle"] = h5py.SoftLink("/cycle")
        assert verifier.metadata_value(root, "alias") == 7
        with pytest.raises(ValueError, match="external"):
            verifier.resolve(root, "external")
        with pytest.raises(ValueError, match="limit"):
            verifier.resolve(root, "cycle")


def test_independent_series_never_reads_trace(verifier, tmp_path, monkeypatch):
    p = tmp_path / "trace.h5"
    with h5py.File(p, "w") as root:
        root.create_dataset("series/idx_start", data=0)
        root.create_dataset("series/count", data=10)
        data = root.create_dataset("series/timeseries/data", shape=(10,), dtype="f8")
        data.attrs.update(unit="Amps", conversion=1.0)
        starting = root.create_dataset("series/timeseries/starting_time", data=0.0)
        starting.attrs["rate"] = 1000.0
    original = h5py.Dataset.__getitem__

    def guarded(self, key):
        if self.name.endswith("/data"):
            pytest.fail("independent reconstruction must not read trace arrays")
        return original(self, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    with h5py.File(p, "r") as root:
        result = verifier.reconstruct_series(root, "series")
    assert result["data_shape"] == [10] and result["sampling_rate"] == 1000
    assert not result["trace_array_read"]
