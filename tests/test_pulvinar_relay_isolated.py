"""Construction and source-boundary tests; no registered scientific outcomes."""

import numpy as np
import pytest

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.models.pulvinar_conductance import CONTROL_NAMES
from smart_robustness.validation.pulvinar_relay_isolated import (
    SERIALIZED_PEAK_WEIGHT,
    build_isolated_pulvinar_relay_assay,
    merged_recording_intervals,
    response_input_arrays,
)


@pytest.fixture(scope="module")
def baseline():
    return load_frozen_classic_baseline("configs/baselines/classic_smart_calibrated_v1.yaml")


def test_input_grid_has_all_controls_and_archived_peak_weight():
    inputs = response_input_arrays(20, 0.01)
    assert inputs["gate"].shape == (round(550.1 / 0.01) + 1, len(CONTROL_NAMES))
    assert inputs["gate"].max() <= 1.01 * SERIALIZED_PEAK_WEIGHT
    np.testing.assert_array_equal(inputs["emissions_ms"], np.arange(10) * 50)
    assert np.all(np.isfinite(inputs["gate"])) and np.all(inputs["gate"] >= 0)


def test_recording_windows_merge_only_when_they_touch_or_overlap():
    assert merged_recording_intervals(0.5) == tuple(
        (2000.0 * i, 2000.0 * i + 100.1) for i in range(10)
    )
    assert merged_recording_intervals(10) == ((0.0, 1000.1),)
    assert merged_recording_intervals(20) == ((0.0, 550.1),)


def test_assay_uses_five_frozen_v2_relays_and_only_projection065_port(baseline):
    import brian2 as brian

    brian.start_scope()
    assay = build_isolated_pulvinar_relay_assay(
        baseline=baseline, frequency_hz=20, dt_ms=0.01, brian=brian,
    )
    assay.network.run(0 * brian.ms)
    assert int(assay.population.group.N) == len(CONTROL_NAMES)
    assert assay.port_name == "port_003"
    assert set(assay.state_monitor.record_variables) == {
        "v_soma", "v_proximal_dendrite", "v_distal_dendrite",
        "port_003_gate", "i_port_003",
    }
    for port in assay.population.compiled.synaptic_ports:
        assert np.all(np.asarray(getattr(assay.population.group, f"{port.name}_gate")) == 0)


@pytest.mark.parametrize("frequency", [1, 40])
def test_unregistered_frequency_rejected(frequency):
    with pytest.raises(ValueError, match="frequency"):
        response_input_arrays(frequency, 0.01)


def test_unregistered_step_rejected():
    with pytest.raises(ValueError, match="step"):
        response_input_arrays(20, 0.02)
