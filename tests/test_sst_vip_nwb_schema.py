"""Read-only metadata inspector never reads response/stimulus trace arrays."""

from __future__ import annotations

import numpy as np
import pytest

from smart_robustness.validation.sst_vip_nwb_schema import inspect_file, safe_get, small_value

h5py = pytest.importorskip("h5py")
RULES = {
    "maximum_soft_link_resolution_steps": 8,
    "maximum_small_metadata_elements": 20,
    "maximum_epoch_names_per_file": 20,
}


def fixture_file(path, rate=1000, count=100):
    with h5py.File(path, "w") as root:
        root.create_dataset("identifier", data="synthetic")
        root.create_dataset("general/generated_by", data=np.array([b"version", b"1.2.3"]))
        for kind, unit in (("stimulus", "Amps"), ("response", "Volts")):
            base = root.create_group("epochs/Sweep_1/" + kind)
            base.create_dataset("idx_start", data=0)
            base.create_dataset("count", data=count)
            series = base.create_group("timeseries")
            data = series.create_dataset("data", shape=(100,), dtype="f8")
            data.attrs.update(unit=unit, conversion=1.0)
            starting = series.create_dataset("starting_time", data=0.0)
            starting.attrs["rate"] = rate


def test_no_trace_reads_and_hash_immutable(tmp_path, monkeypatch):
    p = tmp_path / "cell.nwb"
    fixture_file(p)
    raw = p.read_bytes()
    original = h5py.Dataset.__getitem__

    def guarded(self, key):
        if self.name.endswith("/data"):
            pytest.fail("trace array must never be read")
        return original(self, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    result = inspect_file(p, [{"sweep_number": 1}], RULES)
    assert result["sweep_numbers"] == [1] and not result["missing_metadata_sweeps"]
    assert result["sweeps"][0]["stimulus"]["sampling_rate"] == 1000
    assert not result["sweeps"][0]["errors"]
    assert not result["pipeline_version"]["interpretations_agree"]
    assert p.read_bytes() == raw


def test_missing_and_extra_sweeps_preserved(tmp_path):
    p = tmp_path / "cell.nwb"
    fixture_file(p)
    result = inspect_file(p, [{"sweep_number": 2}], RULES)
    assert result["missing_metadata_sweeps"] == [2] and result["extra_file_sweeps"] == [1]


@pytest.mark.parametrize("rate,count", [(float("nan"), 100), (0, 100), (1000, 101)])
def test_invalid_rate_or_bounds_retained_as_sweep_errors(tmp_path, rate, count):
    p = tmp_path / "cell.nwb"
    fixture_file(p, rate=rate, count=count)
    result = inspect_file(p, [{"sweep_number": 1}], RULES)
    assert len(result["sweeps"][0]["errors"]) == 2


def test_external_links_and_soft_link_cycles_rejected(tmp_path):
    p = tmp_path / "unsafe.h5"
    with h5py.File(p, "w") as root:
        root["external"] = h5py.ExternalLink("must-not-open.h5", "/data")
        root["loop"] = h5py.SoftLink("/loop")
        with pytest.raises(TypeError, match="external"):
            safe_get(root, "external")
        with pytest.raises(ValueError, match="limit"):
            safe_get(root, "loop", maximum_steps=3)


def test_internal_soft_link_and_metadata_limit(tmp_path):
    p = tmp_path / "safe.h5"
    with h5py.File(p, "w") as root:
        root.create_dataset("value", data=42)
        root["alias"] = h5py.SoftLink("/value")
        root.create_dataset("large", shape=(21,))
        assert small_value(root, "alias", RULES) == 42
        with pytest.raises(ValueError, match="limit"):
            small_value(root, "large", RULES)
