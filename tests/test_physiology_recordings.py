"""Synthetic HDF5 loader safety and repeat scoring controls, never real recordings."""

from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

from smart_robustness.validation.physiology_recordings import (
    digest,
    epoch_events,
    experimental_events,
    group_key,
    load_recording,
    score_group,
)
from smart_robustness.validation.sst_vip_nwb_schema import inspect_file

h5py = pytest.importorskip("h5py")
RULES = {
    "maximum_soft_link_resolution_steps": 32,
    "maximum_small_metadata_elements": 4096,
    "maximum_epoch_names_per_file": 100,
}


def recording():
    stimulus = np.zeros(1000)
    response = np.full(1000, -0.06)
    response[[200, 400, 600, 800]] = 0.03
    return {
        "stimulus": stimulus,
        "response": response,
        "rate": 10000.0,
        "start": 0,
        "count": 1000,
        "stimulus_sha256": hashlib.sha256(stimulus.tobytes()).hexdigest(),
    }


def fixture(path, conversion=1.0):
    r = recording()
    with h5py.File(path, "w") as root:
        root.create_dataset("general/generated_by", data=np.array([b"version", b"1.0"]))
        for kind, unit in (("stimulus", "Amps"), ("response", "Volts")):
            base = root.create_group("epochs/Sweep_1/" + kind)
            base.create_dataset("idx_start", data=0)
            base.create_dataset("count", data=1000)
            series = base.create_group("timeseries")
            data = series.create_dataset("data", data=r[kind])
            data.attrs.update(unit=unit, conversion=conversion)
            starting = series.create_dataset("starting_time", data=0.0)
            starting.attrs["rate"] = 10000.0
        root["epochs/Experiment_1"] = root["epochs/Sweep_1"]
    audit = inspect_file(path, [{"sweep_number": 1}], RULES)
    return audit["sweeps"][0], audit["pipeline_version"]


def test_read_only_loader_and_authorization(tmp_path):
    path = tmp_path / "synthetic.nwb"
    inv, versions = fixture(path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="authorization"):
        load_recording(path, digest(path), inv, versions)
    r = load_recording(path, digest(path), inv, versions, authorized=True)
    np.testing.assert_array_equal(r["response"], recording()["response"])
    assert path.read_bytes() == original


def test_hash_and_metadata_tamper_rejected(tmp_path):
    path = tmp_path / "synthetic.nwb"
    inv, versions = fixture(path)
    with pytest.raises(ValueError, match="hash changed"):
        load_recording(path, "0" * 64, inv, versions, authorized=True)
    inv["stimulus"]["sampling_rate"] = 20000.0
    with pytest.raises(ValueError, match="metadata differs"):
        load_recording(path, digest(path), inv, versions, authorized=True)


def test_old_conversion_and_disagreeing_version_rejected(tmp_path):
    path = tmp_path / "synthetic.nwb"
    inv, versions = fixture(path, conversion=0.001)
    with pytest.raises(ValueError, match="physical scaling"):
        load_recording(path, digest(path), inv, versions, authorized=True)
    versions["decoded_pipeline_version"] = [1, 1]
    with pytest.raises(ValueError, match="branches disagree"):
        load_recording(path, digest(path), inv, versions, authorized=True)


def test_exact_repeat_group_and_parity_gate():
    r = recording()
    events = experimental_events(r)
    assert len(events) == 4
    result = score_group([r, r], [events, events], [True, True])
    assert result["passed"] and result["mean_pairwise_F1"] == 1.0
    assert not score_group([r, r], [events, events], [True, False])["passed"]


def test_singleton_and_silent_repeats_inconclusive():
    r = recording()
    assert not score_group([r], [experimental_events(r)], [True])["passed"]
    r["response"][:] = -0.06
    result = score_group([r, r], [[], []], [True, True])
    assert result["mean_pairwise_F1"] is None and not result["passed"]


def test_group_epoch_and_event_boundaries():
    r = recording()
    other = copy.deepcopy(r)
    other["start"] = 1
    assert group_key(r) != group_key(other)
    with pytest.raises(ValueError, match="different stimuli"):
        score_group([r, other], [[], []], [True, True])
    r.update(start=100, count=100)
    np.testing.assert_array_equal(epoch_events([0.009, 0.01, 0.0199, 0.02], r), [0.01, 0.0199])


def test_nonfinite_trace_rejected_and_file_unchanged(tmp_path):
    path = tmp_path / "synthetic.nwb"
    inv, versions = fixture(path)
    with h5py.File(path, "r+") as root:
        root["epochs/Sweep_1/response/timeseries/data"][0] = np.nan
    original = path.read_bytes()
    with pytest.raises(ValueError, match="finite"):
        load_recording(path, digest(path), inv, versions, authorized=True)
    assert path.read_bytes() == original
