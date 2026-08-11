from __future__ import annotations

import numpy as np
import pytest

from smart_robustness.learning import (
    equilibrium_depression_scale,
    postsynaptic_spike_gate,
)


def test_equation6_depression_scale_uses_minimum_baseline_and_maximum() -> None:
    assert equilibrium_depression_scale(
        minimum_weight=0.0, baseline_weight=0.3, maximum_weight=6.0
    ) == pytest.approx(0.95)


def test_equation6_post_spike_gate_has_published_piecewise_boundaries() -> None:
    d = 0.95
    elapsed = np.array([-1.0, 0.0, 0.05, 0.1, 12.6, 25.1, 30.0])
    result = postsynaptic_spike_gate(elapsed, depression_scale=d)
    assert result == pytest.approx([0.0, d + 1, d + 0.5, d, d / 2, 0.0, 0.0])


def test_equation6_gate_uses_serialized_depotentiation_length() -> None:
    assert postsynaptic_spike_gate(
        20.1, depression_scale=0.8, depotentiation_ms=20.0
    ) == pytest.approx(0.0)
