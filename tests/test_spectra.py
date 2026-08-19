import numpy as np

from smart_robustness.analysis.spectra import band_power, dominant_frequency


def sine(frequency_hz: float, duration_s: float = 4.0, sample_rate_hz: float = 1000.0):
    time = np.arange(0, duration_s, 1 / sample_rate_hz)
    return np.sin(2 * np.pi * frequency_hz * time)


def test_dominant_frequency_recovers_gamma() -> None:
    assert dominant_frequency(sine(40), 1000) == pytest.approx(40, abs=1)


def test_band_power_separates_beta_and_gamma() -> None:
    beta_signal = sine(20)
    gamma_signal = sine(40)
    assert band_power(beta_signal, 1000, 12, 30) > band_power(beta_signal, 1000, 30, 70)
    assert band_power(gamma_signal, 1000, 30, 70) > band_power(gamma_signal, 1000, 12, 30)


import pytest
