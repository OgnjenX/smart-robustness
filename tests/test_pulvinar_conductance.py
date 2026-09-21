"""Isolated synthetic traces only; no SMART network outcomes."""

import math

import numpy as np
import pytest

from smart_robustness.models.pulvinar_conductance import (
    CONTROL_NAMES,
    alpha_kernel,
    conductance_controls,
    resource_history,
)
from smart_robustness.models.pulvinar_matching import first_event_match
from smart_robustness.models.pulvinar_stp import STPParameters, release_history

TYPE2 = STPParameters(0.8, 2.0, 3.33)


def test_alpha_is_causal_unit_peak_and_finite():
    wave = alpha_kernel([-1, 0, 2, 2000], 2)
    np.testing.assert_allclose(wave[:3], [0, 0, 1])
    assert np.all(np.isfinite(wave))
    assert wave[-1] >= 0


def test_resource_snapshots_and_right_continuous_jumps():
    times = np.array([0, 0.5, 1, 1.5, 2.0])
    current, snapshots = resource_history(
        times, [0, 1], depletion_fraction=0.5, recovery_ms=1,
    )
    expected_pre_second = 1 - 0.5 * math.exp(-1)
    np.testing.assert_allclose(snapshots, [1, expected_pre_second])
    np.testing.assert_allclose(current[[0, 2]], [0.5, expected_pre_second * 0.5])
    assert current[1] == pytest.approx(1 - 0.5 * math.exp(-0.5))


def test_single_event_factorization_and_matching_are_exact_on_fine_grid():
    dt = 0.001
    time = np.arange(0, 100 + dt, dt)
    traces = conductance_controls(
        time, [0], tau_ms=2, delay_ms=0.1,
        depletion_fraction=0.5, recovery_ms=100,
        type2_parameters=TYPE2,
    )
    assert tuple(traces) == CONTROL_NAMES
    # One event makes union and additive summation identical.
    np.testing.assert_array_equal(
        traces["classic_union_continuous_resource"],
        traces["additive_continuous_resource"],
    )
    # Snapshot is the undepleted source resource and differs from continuous delivery.
    expected_snapshot = alpha_kernel(time - 0.1, 2)
    np.testing.assert_array_equal(traces["additive_emission_snapshot"], expected_snapshot)
    match = first_event_match(
        alpha_tau_ms=2, recovery_ms=100, depletion_fraction=0.5,
        delay_ms=0.1, first_release=0.8,
    )
    assert traces["type2_peak_matched"].max() == pytest.approx(match.classic_peak, abs=2e-7)
    classic_area = np.trapz(traces["classic_union_continuous_resource"], time)
    matched_area = np.trapz(traces["type2_area_matched"], time)
    assert matched_area == pytest.approx(classic_area, rel=1e-9)


def test_multievent_controls_change_only_registered_factors():
    time = np.arange(0, 505, 0.01)
    emissions = np.arange(10) * 50.0
    traces = conductance_controls(
        time, emissions, tau_ms=2, delay_ms=0.1,
        depletion_fraction=0.5, recovery_ms=100,
        type2_parameters=TYPE2,
    )
    assert all(value.shape == time.shape for value in traces.values())
    assert all(np.all(np.isfinite(value)) and np.all(value >= 0) for value in traces.values())
    # With overlap, additive summation cannot be smaller than the two-wave union.
    assert np.all(traces["additive_continuous_resource"] + 1e-15 >=
                  traces["classic_union_continuous_resource"])
    # Matching controls differ only by one constant gain.
    peak = traces["type2_peak_matched"]
    area = traces["type2_area_matched"]
    nonzero = peak > 0
    assert np.ptp(area[nonzero] / peak[nonzero]) < 1e-12


def test_dynamic_trace_uses_every_release_without_normalizing_first_event():
    time = np.arange(0, 600, 0.01)
    emissions = np.arange(3) * 100.0
    traces = conductance_controls(
        time, emissions, tau_ms=2, delay_ms=0.1,
        depletion_fraction=0.5, recovery_ms=100,
        type2_parameters=TYPE2,
    )
    releases = release_history(emissions / 1000, TYPE2)["released"]
    matching = first_event_match(
        alpha_tau_ms=2, recovery_ms=100, depletion_fraction=0.5,
        delay_ms=0.1, first_release=TYPE2.utilization,
    )
    expected = sum(
        release * alpha_kernel(time - emission - 0.1, 2)
        for release, emission in zip(releases, emissions, strict=True)
    ) * matching.peak_matched_gain
    np.testing.assert_allclose(traces["type2_peak_matched"], expected, rtol=1e-14, atol=3e-15)


@pytest.mark.parametrize("dt_ms", [0.01, 0.005])
def test_resource_reference_matches_brian_event_scheduling(dt_ms):
    import brian2 as b

    emissions = np.array([0.0, 1.0, 3.0])
    clock = b.Clock(dt=dt_ms * b.ms)
    source = b.SpikeGeneratorGroup(
        1, np.zeros(emissions.size, dtype=int), emissions * b.ms,
        clock=clock, codeobj_class=b.NumpyCodeObject,
    )
    target = b.NeuronGroup(
        1, "dz/dt=(1-z)/(100*ms) : 1", method="exact",
        clock=clock, codeobj_class=b.NumpyCodeObject,
    )
    target.z = 1
    synapse = b.Synapses(
        source, target, on_pre="z_post *= 0.5", clock=clock,
        codeobj_class=b.NumpyCodeObject,
    )
    synapse.connect()
    monitor = b.StateMonitor(
        target, "z", record=True, when="end", clock=clock,
        codeobj_class=b.NumpyCodeObject,
    )
    b.Network(source, target, synapse, monitor).run(4 * b.ms, namespace={})
    times = np.asarray(monitor.t / b.ms)
    expected, _ = resource_history(
        times, emissions, depletion_fraction=0.5, recovery_ms=100,
    )
    np.testing.assert_allclose(np.asarray(monitor.z)[0], expected, rtol=0, atol=2e-14)


@pytest.mark.parametrize("times,emissions", [
    ([[0]], [0]), ([0, 1], [[0]]), ([0, 0], [0]), ([0, 1], [1, 0]),
    ([-1, 0], [0]), ([0, np.nan], [0]), ([0, 1], [np.inf]),
])
def test_invalid_time_axes(times, emissions):
    with pytest.raises(ValueError):
        conductance_controls(
            times, emissions, tau_ms=2, delay_ms=0.1,
            depletion_fraction=0.5, recovery_ms=100,
            type2_parameters=TYPE2,
        )
