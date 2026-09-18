"""Archive and rest-metric checks using synthetic values, never a live assay."""

import numpy as np
import pytest

from smart_robustness.validation.active_apical_recording import (
    exact_trace_repeat,
    load_trace,
    rest_assessment,
    save_trace,
)


def test_archive_roundtrip_and_no_overwrite(tmp_path):
    path = tmp_path / "trace.npz"
    arrays = {"voltage": np.array([-65.0, -64.0]), "dt": np.array(0.01)}
    digest = save_trace(path, arrays)
    restored = load_trace(path, digest)
    assert exact_trace_repeat(arrays, restored)
    with pytest.raises(FileExistsError):
        save_trace(path, {"other": np.array([1.0])})
    assert exact_trace_repeat(arrays, load_trace(path, digest))
    assert not list(tmp_path.glob(".apical-trace-*"))
    with pytest.raises(ValueError, match="fingerprint"):
        load_trace(path, "incorrect")


def test_arrays_must_match_beyond_summary():
    a = {"v": np.array([1.0, 2.0, 3.0])}
    b = {"v": np.array([3.0, 2.0, 1.0])}
    assert a["v"].mean() == b["v"].mean()
    assert not exact_trace_repeat(a, b)
    assert not exact_trace_repeat(a, {"v": a["v"].astype(np.float32)})
    assert not exact_trace_repeat(a, {"different": a["v"]})


def test_pickle_arrays_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="object"):
        save_trace(tmp_path / "bad.npz", {"bad": np.array([{}], dtype=object)})


def test_rest_gate_requires_stability_no_spikes_and_identical_cells():
    a = {
        "time_ms": 900.0 + np.arange(10000) * 0.01,
        "v_distal_dendrite_mV": np.full((24, 10000), -65.0),
        "spike_time_ms": np.array([]),
    }
    assert rest_assessment(a, 0.01)["pass"]
    a["v_distal_dendrite_mV"][0, 0] += 0.2
    assert not rest_assessment(a, 0.01)["pass"]
    a["v_distal_dendrite_mV"][:] = -65.0
    a["spike_time_ms"] = np.array([950.0])
    assert not rest_assessment(a, 0.01)["pass"]
    a["spike_time_ms"] = np.array([])
    a["v_distal_dendrite_mV"][0, 0] = np.nan
    assert not rest_assessment(a, 0.01)["pass"]
