"""Structural and input checks; do not execute the unsealed response assay."""

import numpy as np
import pytest

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.validation.active_apical_isolated import (
    ARRIVALS_MS,
    assay_cases,
    build_isolated_apical_assay,
    frozen_l5_parameters,
    input_gate_arrays,
)


def test_complete_factorial_input_grid():
    cases = assay_cases()
    assert len(cases) == len(set(cases)) == 24
    assert sum(c.drive_factor == 0 for c in cases) == 4


def test_input_clock_and_delay_are_identical_on_refinement():
    coarse = input_gate_arrays(0.01)
    fine = input_gate_arrays(0.005)
    np.testing.assert_allclose(ARRIVALS_MS, np.array([50.0, 83.33, 116.67]) + 1000.1)
    for key in coarse:
        np.testing.assert_array_equal(coarse[key], fine[key][::2])
    assert not coarse["fast"][coarse["time_ms"] <= 1000.0].any()
    assert not coarse["slow"][coarse["time_ms"] <= 1000.0].any()


def test_unregistered_timestep_rejected():
    with pytest.raises(ValueError):
        input_gate_arrays(0.1)


@pytest.mark.parametrize("arm", ["classic_ampa", "mixed_rest_block", "mixed_active_block"])
def test_constructed_assay_preserves_classic_cell_and_ports_without_running(arm):
    import brian2 as brian

    brian.start_scope()
    brian.prefs.codegen.target = "numpy"
    baseline = load_frozen_classic_baseline("configs/baselines/classic_smart_calibrated_v1.yaml")
    original = frozen_l5_parameters(baseline)
    assay = build_isolated_apical_assay(
        baseline=baseline,
        arm=arm,
        resting_distal_mV=-65.0,
        dt_ms=0.01,
        brian=brian,
    )
    assert assay.population.cell_spec == original["cell_spec"]
    ports = assay.population.compiled.synaptic_ports
    assert ports[: len(original["synaptic_ports"])] == original["synaptic_ports"]
    assert len(assay.population.group) == 24
    assert not assay.state_monitor.active
    assert float(assay.network.t / brian.ms) == 0.0
    assert not any(isinstance(obj, brian.Synapses) for obj in assay.network.objects)
    assert "i_assay_proximal" in assay.state_monitor.record_variables
    assay.network.run(0 * brian.ms)
    assert float(assay.network.t / brian.ms) == 0.0
