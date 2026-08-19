from __future__ import annotations

import numpy as np
import pytest

from smart_robustness.analysis.cross_correlation import (
    assess_figure16_cross_correlations,
    band_limited_cross_correlation,
    figure16_cross_correlations,
)
from smart_robustness.validation.higher_order import Figure16Protocol


def _sine(frequency_hz: float, *, duration_s: float = 1.0, sample_rate_hz: int = 1000):
    time = np.arange(int(duration_s * sample_rate_hz)) / sample_rate_hz
    return np.sin(2 * np.pi * frequency_hz * time)


def test_band_limited_cross_correlation_recovers_known_lag() -> None:
    signal = _sine(8) + 0.7 * _sine(9) + 0.4 * _sine(11)
    delayed = np.roll(signal, 20)
    result = band_limited_cross_correlation(signal, delayed, 1000, (8, 12))
    # X*conj(Y) places a delayed second signal at negative lag.
    assert result.peak_lag_ms == pytest.approx(-20.0)
    assert result.peak_absolute_normalized == pytest.approx(1.0, abs=1e-12)
    assert np.max(np.abs(result.normalized)) <= 1 + 1e-12


def test_figure16_analysis_separates_lower_and_gamma_coupling() -> None:
    lower = _sine(10)
    gamma_first = 0.2 * _sine(40)
    gamma_second = 0.2 * _sine(43)
    first = lower + gamma_first
    second = np.roll(lower, 8) + gamma_second
    protocol = Figure16Protocol()
    results = figure16_cross_correlations(
        first, second, 1000, bands_hz=protocol.frequency_bands_hz
    )
    assessment = assess_figure16_cross_correlations(results)
    assert len(results) == 5
    assert assessment.lower_frequency_stronger_than_gamma
    assert assessment.strongest_band_hz == (8.0, 12.0)


def test_zero_energy_band_returns_zero_normalized_curve() -> None:
    signal = _sine(10)
    result = band_limited_cross_correlation(signal, signal, 1000, (20, 100))
    assert result.peak_absolute_normalized < 1e-12


@pytest.mark.parametrize(
    "first,second,rate,band",
    (
        (np.arange(7), np.arange(7), 1000, (8, 12)),
        (np.arange(8), np.arange(9), 1000, (8, 12)),
        (np.arange(8), np.arange(8), 0, (8, 12)),
        (np.arange(8), np.arange(8), 1000, (12, 8)),
        (np.arange(8), np.arange(8), 1000, (8, 600)),
    ),
)
def test_cross_correlation_rejects_invalid_inputs(first, second, rate, band) -> None:
    with pytest.raises(ValueError):
        band_limited_cross_correlation(first, second, rate, band)
