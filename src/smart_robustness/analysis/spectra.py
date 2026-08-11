from __future__ import annotations

import numpy as np
from scipy.signal import welch


def power_spectrum(signal: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size < 8:
        raise ValueError("signal must be one-dimensional with at least 8 samples")
    values = values - values.mean()
    nperseg = min(values.size, max(8, int(sample_rate_hz)))
    return welch(values, fs=sample_rate_hz, nperseg=nperseg)


def band_power(
    signal: np.ndarray, sample_rate_hz: float, low_hz: float, high_hz: float
) -> float:
    freqs, power = power_spectrum(signal, sample_rate_hz)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    return float(np.trapz(power[mask], freqs[mask])) if mask.any() else 0.0


def dominant_frequency(
    signal: np.ndarray, sample_rate_hz: float, low_hz: float = 1.0, high_hz: float = 100.0
) -> float:
    freqs, power = power_spectrum(signal, sample_rate_hz)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not mask.any():
        return float("nan")
    return float(freqs[mask][np.argmax(power[mask])])


def summarize_rate(rate_hz: np.ndarray, sample_rate_hz: float, bands: dict) -> dict[str, float]:
    return {
        "mean_rate_hz": float(np.mean(rate_hz)),
        "dominant_frequency_hz": dominant_frequency(rate_hz, sample_rate_hz),
        "beta_power": band_power(rate_hz, sample_rate_hz, *bands["beta_hz"]),
        "gamma_power": band_power(rate_hz, sample_rate_hz, *bands["gamma_hz"]),
    }
