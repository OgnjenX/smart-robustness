"""Synthetic controls only; no experimental responses are loaded."""

import numpy as np
import pytest

from smart_robustness.validation.physiology_metrics import (
    native_trace_values,
    recruitment_gate,
    spike_agreement,
    upward_crossings,
)


def test_native_conversion_version_boundary():
    for version in ((0, 0), (1, 0)):
        np.testing.assert_array_equal(native_trace_values([1.0, 2.0], 0.001, version), [1.0, 2.0])
    np.testing.assert_array_equal(native_trace_values([1.0, 2.0], 0.001, (1, 1)), [0.001, 0.002])


def test_interpolated_crossings_and_refractory():
    np.testing.assert_allclose(upward_crossings([-0.01, 0.01, -0.01, 0.01], 1000), [0.0005, 0.0025])
    np.testing.assert_allclose(
        upward_crossings([-0.01, 0.01, -0.01, 0.01], 1000, refractory=0.003), [0.0005]
    )


def test_one_to_one_events_and_silence():
    result = spike_agreement([0.1, 0.102], [0.101])
    assert result["matches"] == 1
    assert result["f1"] == pytest.approx(2 / 3)
    assert spike_agreement([], [])["f1"] is None
    assert spike_agreement([0.1], [0.2])["f1"] == 0


def test_rate_and_reliability_gates():
    assert recruitment_gate([0.1, 0.2], [0.1, 0.2], 1.0, 0.8)["passed"]
    assert not recruitment_gate([0.1, 0.2], [0.1, 0.2], 1.0, 0.4)["passed"]
    assert not recruitment_gate([], [], 1.0, 1.0)["passed"]
    assert not recruitment_gate([0.1], [0.1], 1.0, np.nan)["passed"]


@pytest.mark.parametrize("values", [[np.nan], [[1.0]], [np.inf]])
def test_invalid_data_rejected(values):
    with pytest.raises(ValueError):
        upward_crossings(values, 1000)


def test_unsorted_spikes_rejected():
    with pytest.raises(ValueError):
        spike_agreement([0.2, 0.1], [0.1])
