"""Synthetic algebra tests, not biological or network validation."""

import math

import numpy as np
import pytest

from smart_robustness.models.pulvinar_stp import STPParameters, release_history


def test_hand_derived_two_events_and_update_order():
    p = STPParameters(0.5, math.log(2), math.log(2))
    r = release_history([0, 1], p)
    np.testing.assert_allclose(r["u_pre"], [0, 0.25])
    np.testing.assert_allclose(r["x_pre"], [1, 0.75])
    np.testing.assert_allclose(r["u_post"], [0.5, 0.625])
    np.testing.assert_allclose(r["released"], [0.5, 0.46875])
    np.testing.assert_allclose(r["x_post"], [0.5, 0.28125])


def test_conservation_bounds_and_exact_repetition():
    p = STPParameters(0.3, 2, 3)
    arrivals = np.arange(1000) * 0.01
    a, b = release_history(arrivals, p), release_history(arrivals, p)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
        if key != "time_s":
            assert np.all((a[key] >= 0) & (a[key] <= 1))
    np.testing.assert_allclose(a["x_pre"], a["x_post"] + a["released"])


def test_long_silence_restores_first_release_and_time_origin_does_not_matter():
    p = STPParameters(0.4, 1, 1)
    r = release_history([10, 11, 1011], p)
    assert r["released"][-1] == pytest.approx(0.4)
    zero = release_history([0, 1], p)
    for key in zero:
        if key != "time_s":
            np.testing.assert_array_equal(zero[key], r[key][:2])


def test_empty_train_and_full_utilization():
    p = STPParameters(1, 1, 1)
    assert all(v.size == 0 for v in release_history([], p).values())
    r = release_history([0, 1], p)
    np.testing.assert_array_equal(r["x_post"], [0, 0])
    np.testing.assert_allclose(r["released"], [1, 1 - math.exp(-1)])


def test_independent_brian_event_driven_equations_match_oracle():
    import brian2 as b

    # Dedicated clock and Network avoid changes to the classic default clock.
    clock = b.Clock(dt=0.1 * b.ms)
    arrivals = np.array([0, 0.01, 0.03, 0.1])
    p = STPParameters(0.3, 2, 3)
    source = b.SpikeGeneratorGroup(
        1, np.zeros(4, dtype=int), arrivals * b.second, clock=clock,
        codeobj_class=b.NumpyCodeObject,
    )
    target = b.NeuronGroup(1, "", clock=clock, codeobj_class=b.NumpyCodeObject)
    synapse = b.Synapses(
        source, target,
        model="""
        du/dt = -2*Hz*u : 1 (event-driven)
        dx/dt = 3*Hz*(1-x) : 1 (event-driven)
        released : 1
        """,
        on_pre="""
        u += 0.3*(1-u)
        released = u*x
        x -= released
        """,
        clock=clock, codeobj_class=b.NumpyCodeObject,
    )
    synapse.connect()
    synapse.u = 0
    synapse.x = 1
    monitor = b.StateMonitor(
        synapse, ["u", "x", "released"], record=True, when="end", clock=clock,
        codeobj_class=b.NumpyCodeObject,
    )
    b.Network(source, target, synapse, monitor).run(101 * b.ms, namespace={})
    indices = np.rint(arrivals / 0.0001).astype(int)
    expected = release_history(arrivals, p)
    for variable, key in (("u", "u_post"), ("x", "x_post"), ("released", "released")):
        actual = np.asarray(getattr(monitor, variable))[0, indices]
        np.testing.assert_allclose(actual, expected[key], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("arrivals", [[0, 0], [1, 0], [-1], [np.nan], [np.inf], [[0]]])
def test_reject_invalid_arrivals(arrivals):
    with pytest.raises(ValueError):
        release_history(arrivals, STPParameters(0.5, 1, 1))


@pytest.mark.parametrize("values", [
    (0, 1, 1), (1.1, 1, 1), (0.5, 0, 1), (0.5, 1, -1),
    (np.nan, 1, 1), (0.5, np.inf, 1), (0.5, 1, np.nan),
])
def test_reject_invalid_parameters(values):
    with pytest.raises(ValueError):
        STPParameters(*values)
