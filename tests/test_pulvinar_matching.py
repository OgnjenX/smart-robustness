"""Synthetic continuous-time first-event checks; no scientific parameter fit."""

import math

import pytest
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from smart_robustness.models.pulvinar_matching import first_event_match


def test_no_depletion_is_static_kernel():
    m = first_event_match(alpha_tau_ms=3, recovery_ms=20, depletion_fraction=0,
                          delay_ms=1, first_release=0.25)
    assert m.classic_peak == pytest.approx(1)
    assert m.classic_peak_after_arrival_ms == 3
    assert m.classic_area_ms == pytest.approx(3 * math.e)
    assert m.peak_matched_gain == m.area_matched_gain == 4


@pytest.mark.parametrize("depletion,delay", [(0.2, 0), (0.7, 1), (1, 0), (1, 5)])
def test_analytic_area_and_peak_against_independent_numerics(depletion, delay):
    tau, recovery, release = 3.0, 17.0, 0.25
    m = first_event_match(alpha_tau_ms=tau, recovery_ms=recovery,
                          depletion_fraction=depletion, delay_ms=delay,
                          first_release=release)

    def waveform(t):
        return math.e * (t / tau) * math.exp(-t / tau) * (
            1 - depletion * math.exp(-(t + delay) / recovery)
        )

    integral, _ = quad(waveform, 0, math.inf)
    peak = minimize_scalar(lambda t: -waveform(t), bounds=(0, 20 * tau), method="bounded")
    assert m.classic_area_ms == pytest.approx(integral, rel=1e-10)
    assert m.classic_peak == pytest.approx(-peak.fun, rel=1e-10)
    assert m.peak_matched_gain * release == pytest.approx(m.classic_peak)
    assert m.area_matched_gain * release * m.static_area_ms == pytest.approx(integral)
    assert m.peak_matched_gain != pytest.approx(m.area_matched_gain, rel=1e-4)


@pytest.mark.parametrize("key,value", [
    ("alpha_tau_ms", 0), ("recovery_ms", -1), ("depletion_fraction", 1.1),
    ("depletion_fraction", -0.1), ("delay_ms", -1), ("first_release", 0),
    ("first_release", 1.1), ("alpha_tau_ms", math.nan), ("delay_ms", math.inf),
])
def test_invalid_inputs(key, value):
    parameters = {"alpha_tau_ms": 3, "recovery_ms": 17, "depletion_fraction": 0.7,
                  "delay_ms": 1, "first_release": 0.25}
    parameters[key] = value
    with pytest.raises(ValueError):
        first_event_match(**parameters)
