"""Synthetic traces only; these are not scientific outcomes."""

from copy import deepcopy

import numpy as np
import pytest

from smart_robustness.validation.active_apical_metrics import (
    COMPARTMENTS,
    mechanism_points,
    numerical_gate,
    summarize_response,
)


@pytest.fixture
def arrays():
    t = 1000.0 + np.arange(40000) * 0.01
    result = {
        "time_ms": t,
        "spike_time_ms": np.array([1040.0, 1060.0, 1400.0]),
        "spike_cell_index": np.array([7, 7, 7]),
    }
    for c in COMPARTMENTS:
        v = np.full((24, len(t)), -65.0)
        v[5] += 1.0
        v[6] += 2.0
        v[7] += 4.0
        result[f"v_{c}_mV"] = v
    return result


def test_areas_excess_and_latency_have_explicit_units_and_boundaries(arrays):
    r = summarize_response(arrays, 0.01)
    assert r[7]["compartments"]["soma"]["positive_area_mV_ms"] == 1600.0
    assert r[7]["paired_area_excess_mV_ms"]["distal_dendrite"] == 400.0
    assert r[7]["soma_spike_count"] == 2
    assert r[7]["first_event_latency_ms"] == pytest.approx(-10.1)
    assert r[0]["first_event_latency_ms"] is None
    assert r[7]["distal_above_minus40_ms"] == 0.0


def test_nonfinite_and_truncated_recordings_are_rejected(arrays):
    arrays["v_soma_mV"][0, 0] = np.nan
    with pytest.raises(ValueError, match="voltage"):
        summarize_response(arrays, 0.01)
    arrays["time_ms"] = arrays["time_ms"][:-1]
    with pytest.raises(ValueError, match="recording"):
        summarize_response(arrays, 0.01)


def test_numerical_gate_reports_all_failures(arrays):
    fine = summarize_response(arrays, 0.01)
    coarse = deepcopy(fine)
    assert numerical_gate(coarse, fine)["pass"]
    coarse[7]["compartments"]["soma"]["peak_voltage_mV"] += 1.01
    coarse[7]["compartments"]["distal_dendrite"]["positive_area_mV_ms"] *= 1.06
    coarse[7]["soma_spike_count"] += 1
    gate = numerical_gate(coarse, fine)
    assert not gate["pass"]
    assert set(gate["failures"][0]["reasons"]) == {
        "soma_peak",
        "distal_dendrite_area",
        "soma_event_count",
    }
    with pytest.raises(ValueError):
        numerical_gate(coarse[:-1], fine)


def test_mechanism_requires_same_positive_drive_point_and_positive_excess(arrays):
    active = summarize_response(arrays, 0.01)
    fixed = deepcopy(active)
    assert mechanism_points(active, fixed) == []
    fixed[7]["compartments"]["distal_dendrite"]["positive_area_mV_ms"] = 1500.0
    assert mechanism_points(active, fixed) == [0.25]
    active[7]["paired_area_excess_mV_ms"]["distal_dendrite"] = 0.0
    assert mechanism_points(active, fixed) == []


def test_nonfinite_metrics_cannot_pass_gates(arrays):
    original = summarize_response(arrays, 0.01)
    bad = deepcopy(original)
    bad[7]["compartments"]["distal_dendrite"]["positive_area_mV_ms"] = float("nan")
    assert not numerical_gate(bad, original)["pass"]
    assert mechanism_points(bad, original) == []
