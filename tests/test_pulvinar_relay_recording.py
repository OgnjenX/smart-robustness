"""Synthetic recording tests; no registered relay outcomes."""

from copy import deepcopy

import numpy as np
import pytest

from smart_robustness.models.pulvinar_conductance import CONTROL_NAMES
from smart_robustness.validation.pulvinar_relay_recording import (
    expected_recorded_gate,
    numerical_gate,
    summarize,
)


def _arrays():
    dt = 0.1
    rest_time = np.arange(900, 1000, dt)
    response_time = np.arange(1000, 1001, dt)
    time = np.concatenate((rest_time, response_time))
    segment = np.concatenate((np.zeros(rest_time.size, dtype=int), np.ones(response_time.size, dtype=int)))
    gate = np.zeros((len(CONTROL_NAMES), time.size))
    gate[:, rest_time.size:] = np.arange(1, len(CONTROL_NAMES) + 1)[:, None]
    soma = np.full_like(gate, -60.0)
    proximal = np.full_like(gate, -60.0)
    for cell in range(len(CONTROL_NAMES)):
        soma[cell, rest_time.size:] += cell + 1
        proximal[cell, rest_time.size:] += 2 * (cell + 1)
    return {
        "time_ms": time,
        "record_segment_id": segment,
        "spike_time_ms": np.array([1000.2, 1000.3]),
        "spike_cell_index": np.array([0, 4]),
        "dt_ms": np.array(dt),
        "frequency_hz": np.array(20.0),
        "expected_gate": gate.copy(),
        "port_003_gate": gate,
        "v_soma_mV": soma,
        "v_proximal_dendrite_mV": proximal,
        "v_distal_dendrite_mV": np.full_like(gate, -60.0),
        "i_port_003_pA": -gate,
    }


def test_expected_gate_maps_only_retained_response_samples():
    time = np.array([900.0, 999.9, 1000.0, 1000.1, 3000.0])
    segment = np.array([0, 0, 1, 1, 2])
    response = np.arange(20001 * len(CONTROL_NAMES), dtype=float).reshape(20001, -1)
    actual = expected_recorded_gate(time, segment, response, 0.1)
    np.testing.assert_array_equal(actual[:, :2], 0)
    np.testing.assert_array_equal(actual[:, 2], response[0])
    np.testing.assert_array_equal(actual[:, 3], response[1])
    np.testing.assert_array_equal(actual[:, 4], response[20000])


def test_summary_preserves_control_identity_and_spikes():
    result = summarize(_arrays())
    assert result["rest_cells_exactly_identical"]
    assert result["rest_spike_count"] == 0
    assert [row["control"] for row in result["controls"]] == list(CONTROL_NAMES)
    assert result["controls"][0]["response_spike_count"] == 1
    assert result["controls"][1]["response_spike_count"] == 0
    assert result["controls"][4]["first_response_spike_latency_ms"] == pytest.approx(0.3)
    assert result["controls"][2]["positive_soma_area_mV_ms"] > 0
    assert result["controls"][2]["inward_current_charge_pA_ms"] > 0


def test_numerical_gate_reports_each_registered_metric():
    reference = summarize(_arrays())
    assert numerical_gate(reference, deepcopy(reference))["pass"]
    changed = deepcopy(reference)
    changed["controls"][0]["response_soma_peak_mV"] += 0.51
    changed["controls"][0]["response_proximal_peak_mV"] += 0.51
    changed["controls"][0]["positive_soma_area_mV_ms"] += 1.1
    changed["controls"][0]["inward_current_charge_pA_ms"] += 1.1
    changed["controls"][0]["response_spike_count"] += 1
    changed["controls"][0]["first_response_spike_latency_ms"] += 0.051
    gate = numerical_gate(changed, reference)
    assert not gate["pass"]
    assert set(gate["failures"][0]["reasons"]) == {
        "response_soma_peak_mV", "response_proximal_peak_mV",
        "positive_soma_area_mV_ms", "inward_current_charge_pA_ms",
        "response_spike_count", "first_response_spike_latency_ms",
    }


def test_summary_rejects_gate_mismatch():
    arrays = _arrays()
    arrays["port_003_gate"][0, -1] += 1e-15
    try:
        summarize(arrays)
    except ValueError as error:
        assert "gate" in str(error)
    else:
        raise AssertionError("gate mismatch accepted")
