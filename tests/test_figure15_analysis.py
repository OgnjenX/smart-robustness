from __future__ import annotations

import numpy as np
import pytest

from smart_robustness.analysis.figure15 import (
    assess_figure15_synchrony,
    figure15_layer4_synchrony,
)


def test_figure15_recovers_predeclared_local_gamma_pair() -> None:
    period_ms = 1000.0 / 44.0
    first = np.arange(10.0, 990.0, period_ms)
    second = first + 2.0
    indices = tuple([39] * len(first) + [40] * len(second))
    times = tuple(first) + tuple(second)
    result = figure15_layer4_synchrony(indices, times)
    assessment = assess_figure15_synchrony(result)
    assert result.lag_ms[[0, -1]] == pytest.approx((-180, 180))
    assert result.gamma_peak_hz == pytest.approx(44.3213, abs=0.01)
    assert assessment.reproduced
    assert not result.cross_correlation.flags.writeable


def test_figure15_rejects_a_gamma_peak_outside_tolerance() -> None:
    period_ms = 1000.0 / 30.0
    spikes = np.arange(10.0, 990.0, period_ms)
    result = figure15_layer4_synchrony(
        tuple([39] * len(spikes) + [40] * len(spikes)),
        tuple(spikes) + tuple(spikes + 1),
    )
    assert not assess_figure15_synchrony(result).reproduced


def test_figure15_requires_both_cells_to_fire_repeatedly() -> None:
    result = figure15_layer4_synchrony((39, 40), (10.0, 11.0))
    assessment = assess_figure15_synchrony(result)
    assert not assessment.enough_spikes
    assert not assessment.reproduced
