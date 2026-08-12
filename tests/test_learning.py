from __future__ import annotations

import numpy as np
import pytest

from smart_robustness.learning import (
    LEARNING_RULES,
    equilibrium_depression_scale,
    full_postsynaptic_learning_signal,
    gated_weight_derivative,
    learning_gate,
    postsynaptic_spike_gate,
)
from smart_robustness.validation.figure6 import run_figure6_timing_curves


def test_equation6_depression_scale_uses_minimum_baseline_and_maximum() -> None:
    assert equilibrium_depression_scale(
        minimum_weight=0.0, baseline_weight=0.3, maximum_weight=6.0
    ) == pytest.approx(-0.05)


def test_equation6_post_spike_gate_has_published_piecewise_boundaries() -> None:
    d = -0.05
    elapsed = np.array([-1.0, 0.0, 0.05, 0.1, 12.6, 25.1, 30.0])
    result = postsynaptic_spike_gate(elapsed, depression_scale=d)
    assert result == pytest.approx([0.0, d + 1, d + 0.5, d, d / 2, 0.0, 0.0])


def test_equation6_gate_uses_serialized_depotentiation_length() -> None:
    assert postsynaptic_spike_gate(
        20.1, depression_scale=-0.2, depotentiation_ms=20.0
    ) == pytest.approx(0.0)


def test_full_equation6_signal_includes_above_threshold_spike_branch() -> None:
    assert full_postsynaptic_learning_signal(
        np.array([-0.1, 0.0, 0.99, 1.0, 1.1]),
        depression_scale=-0.05,
        spike_above_threshold_ms=1.0,
    ) == pytest.approx([0.0, 0.95, 0.95, 0.95, -0.05])


@pytest.mark.parametrize(
    "rule",
    ["Presynaptically gated", "Postsynaptically gated", "Dual AND gated"],
)
def test_kinness_gated_learning_potentiates_pre_before_post(rule: str) -> None:
    derivative = gated_weight_derivative(
        weight=0.3,
        minimum_weight=0.0,
        baseline_weight=0.3,
        maximum_weight=6.0,
        pre_signal=0.8,
        post_signal=0.95,
        learning_rate=0.1,
        learning_rule=rule,
    )
    assert derivative > 0


@pytest.mark.parametrize(
    "rule",
    ["Presynaptically gated", "Postsynaptically gated", "Dual AND gated"],
)
def test_kinness_gated_learning_depresses_post_before_pre(rule: str) -> None:
    derivative = gated_weight_derivative(
        weight=0.3,
        minimum_weight=0.0,
        baseline_weight=0.3,
        maximum_weight=6.0,
        pre_signal=0.8,
        post_signal=-0.05,
        learning_rate=0.1,
        learning_rule=rule,
    )
    assert derivative < 0


def test_all_five_kinness_table3_learning_gates_are_exact() -> None:
    pre, post = 0.4, -0.2
    assert LEARNING_RULES == (
        "No gate",
        "Presynaptically gated",
        "Postsynaptically gated",
        "Dual AND gated",
        "Dual OR gated",
    )
    assert learning_gate(pre, post, "No gate") == 1
    assert learning_gate(pre, post, "Presynaptically gated") == pre
    assert learning_gate(pre, post, "Postsynaptically gated") == post**2
    assert learning_gate(pre, post, "Dual AND gated") == pre * post**2
    assert learning_gate(pre, post, "Dual OR gated") == pre + post**2


def test_figure6_timing_curves_have_published_stdp_polarity() -> None:
    result = run_figure6_timing_curves()
    times = np.asarray(result.protocol.relative_times_ms)
    for rule, curve_values in result.curves.items():
        curve = np.asarray(curve_values)
        assert curve[times < 0].min() < 0, rule
        assert curve[times > 0].max() > 0, rule
        assert abs(curve[0]) < max(abs(curve.min()), abs(curve.max())), rule
        assert abs(curve[-1]) < 0.01, rule
