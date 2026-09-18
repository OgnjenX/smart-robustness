"""Analytic component checks only: no unsealed biological response outcomes."""

import math

import numpy as np
import pytest

from smart_robustness.models.active_apical_integration import (
    ARMS,
    arrival_gate,
    isolated_receptor_ports,
    nmda_block,
    receptor_kernel,
)


def test_block_matches_declared_equation_and_increases_with_voltage():
    v = np.array([-100.0, -80.0, -60.0, -40.0, -20.0, 0.0, 20.0])
    np.testing.assert_allclose(nmda_block(v), 1 / (1 + 0.33 * np.exp(-v / 16.7)))
    assert np.all(np.diff(nmda_block(v)) > 0)
    assert np.isfinite(nmda_block([-1e6, 1e6])).all()


@pytest.mark.parametrize("rise,fall", [(2.0, 2.0), (0.7, 80.0)])
def test_kernels_are_causal_and_unit_peak(rise, fall):
    peak = rise if rise == fall else rise * fall * math.log(fall / rise) / (fall - rise)
    values = receptor_kernel([-1.0, 0.0, peak, 10000.0], rise_ms=rise, fall_ms=fall)
    np.testing.assert_allclose(values[:3], [0.0, 0.0, 1.0], atol=1e-14)
    assert 0 <= values[-1] < 1e-40


def test_union_uses_two_recent_events_not_all_history():
    t = np.array([0.0, 3.0, 6.0, 11.0])
    arrivals = [1.0, 4.0, 9.0]
    result = arrival_gate(t, arrivals, rise_ms=2.0, fall_ms=2.0)
    expected = []
    for time in t:
        recent = [a for a in arrivals if a <= time][-2:]
        waves = [(time - a) / 2 * math.exp(1 - (time - a) / 2) for a in recent]
        expected.append(1 - math.prod(1 - x for x in waves))
    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(arrival_gate(t, [], rise_ms=2.0, fall_ms=2.0), 0.0)


@pytest.mark.parametrize("arm", ARMS)
def test_probe_matches_total_conductance_not_density(arm):
    ports = isolated_receptor_ports(
        arm,
        resting_distal_mV=-65.0,
        distal_area_cm2=2e-5,
        proximal_area_cm2=5e-5,
    )
    assert all(p.reversal_mV == 0 for p in ports)
    assert ports[-1].conductance_density_mS_cm2 * 5e-5 == pytest.approx(0.09 * 2e-5)
    assert ports[0].conductance_density_mS_cm2 == pytest.approx(0.09 if arm == ARMS[0] else 0.072)


def test_active_and_fixed_block_have_equal_current_at_rest_only():
    kw = {"resting_distal_mV": -65.0, "distal_area_cm2": 2e-5, "proximal_area_cm2": 5e-5}
    active = isolated_receptor_ports("mixed_active_block", **kw)[1]
    fixed = isolated_receptor_ports("mixed_rest_block", **kw)[1]
    assert active.voltage_block and not fixed.voltage_block
    assert active.rise_ms == fixed.rise_ms == 0.7
    assert active.fall_ms == fixed.fall_ms == 80.0
    assert active.conductance_density_mS_cm2 * nmda_block(-65.0) == pytest.approx(
        fixed.conductance_density_mS_cm2
    )
    assert active.conductance_density_mS_cm2 * nmda_block(-40.0) > fixed.conductance_density_mS_cm2


@pytest.mark.parametrize("arrivals", [[1.0, 1.0], [2.0, 1.0], [float("nan")]])
def test_bad_arrival_history_rejected(arrivals):
    with pytest.raises(ValueError):
        arrival_gate([0.0, 1.0], arrivals, rise_ms=2.0, fall_ms=2.0)


@pytest.mark.parametrize("rise,fall", [(0.0, 2.0), (3.0, 2.0), (float("nan"), 2.0)])
def test_invalid_kinetics_rejected_even_without_events(rise, fall):
    with pytest.raises(ValueError):
        arrival_gate([0.0], [], rise_ms=rise, fall_ms=fall)


def test_invalid_arm_and_area_rejected():
    with pytest.raises(ValueError):
        isolated_receptor_ports(
            "other", resting_distal_mV=-65.0, distal_area_cm2=1.0, proximal_area_cm2=1.0
        )
    with pytest.raises(ValueError):
        isolated_receptor_ports(
            "classic_ampa", resting_distal_mV=-65.0, distal_area_cm2=0.0, proximal_area_cm2=1.0
        )
