"""Band-limited cross-correlation used by SMART Figure 16."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BandLimitedCrossCorrelation:
    band_hz: tuple[float, float]
    lag_ms: np.ndarray
    raw: np.ndarray
    normalized: np.ndarray
    peak_absolute_normalized: float
    peak_lag_ms: float


@dataclass(frozen=True, slots=True)
class Figure16CrossCorrelationAssessment:
    peak_by_band: tuple[tuple[tuple[float, float], float], ...]
    strongest_band_hz: tuple[float, float]
    lower_frequency_stronger_than_gamma: bool


def _validated_pair(
    first: np.ndarray, second: np.ndarray, sample_rate_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 8:
        raise ValueError("signals must be equal-length one-dimensional arrays of at least 8 samples")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be finite and positive")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("signals must contain only finite values")
    return x - x.mean(), y - y.mean()


def band_limited_cross_correlation(
    first: np.ndarray,
    second: np.ndarray,
    sample_rate_hz: float,
    band_hz: tuple[float, float],
) -> BandLimitedCrossCorrelation:
    """Apply the Fourier-domain procedure stated in the Figure 16 caption.

    Positive lag means that ``second`` leads ``first``. The raw inverse FFT is
    retained alongside a bounded energy-normalized curve because the paper
    reports graphical, rather than numeric, correlation amplitudes.
    """

    x, y = _validated_pair(first, second, sample_rate_hz)
    low_hz, high_hz = band_hz
    nyquist = sample_rate_hz / 2.0
    if low_hz < 0 or high_hz <= low_hz or high_hz > nyquist:
        raise ValueError("band must be ordered, nonnegative, and within Nyquist")
    frequencies = np.fft.rfftfreq(x.size, d=1.0 / sample_rate_hz)
    # Adjacent Figure 16 bands share printed endpoints. Half-open masks avoid
    # assigning one Fourier bin to two bands; the Nyquist endpoint is included.
    mask = (frequencies >= low_hz) & (
        (frequencies < high_hz) | ((high_hz == nyquist) & (frequencies <= high_hz))
    )
    if not np.any(mask):
        raise ValueError("band contains no Fourier bins at this duration and sample rate")
    first_spectrum = np.fft.rfft(x)
    second_spectrum = np.fft.rfft(y)
    cross_spectrum = np.zeros_like(first_spectrum)
    cross_spectrum[mask] = first_spectrum[mask] * np.conj(second_spectrum[mask])
    raw = np.fft.fftshift(np.fft.irfft(cross_spectrum, n=x.size))

    first_band = np.zeros_like(first_spectrum)
    second_band = np.zeros_like(second_spectrum)
    first_band[mask] = first_spectrum[mask]
    second_band[mask] = second_spectrum[mask]
    first_filtered = np.fft.irfft(first_band, n=x.size)
    second_filtered = np.fft.irfft(second_band, n=x.size)
    energy = float(np.sqrt(np.sum(first_filtered**2) * np.sum(second_filtered**2)))
    total_energy = float(np.sqrt(np.sum(x**2) * np.sum(y**2)))
    # FFT roundoff leaves tiny nonzero coefficients in mathematically empty
    # bands. Normalizing that leakage would manufacture a unit correlation.
    meaningful_energy = energy > np.finfo(float).eps * max(total_energy, 1.0) * x.size
    normalized = raw / energy if meaningful_energy else np.zeros_like(raw)
    lag_samples = np.arange(-x.size // 2, x.size - x.size // 2)
    lag_ms = lag_samples * 1000.0 / sample_rate_hz
    peak_index = int(np.argmax(np.abs(normalized)))
    return BandLimitedCrossCorrelation(
        band_hz=(float(low_hz), float(high_hz)),
        lag_ms=lag_ms,
        raw=raw,
        normalized=normalized,
        peak_absolute_normalized=float(abs(normalized[peak_index])),
        peak_lag_ms=float(lag_ms[peak_index]),
    )


def figure16_cross_correlations(
    first: np.ndarray,
    second: np.ndarray,
    sample_rate_hz: float,
    *,
    bands_hz: tuple[tuple[float, float], ...],
) -> tuple[BandLimitedCrossCorrelation, ...]:
    if not bands_hz:
        raise ValueError("at least one Figure 16 frequency band is required")
    return tuple(
        band_limited_cross_correlation(first, second, sample_rate_hz, band)
        for band in bands_hz
    )


def assess_figure16_cross_correlations(
    results: tuple[BandLimitedCrossCorrelation, ...],
    *,
    gamma_band_hz: tuple[float, float] = (20.0, 100.0),
) -> Figure16CrossCorrelationAssessment:
    if not results:
        raise ValueError("Figure 16 assessment requires band results")
    peaks = tuple((result.band_hz, result.peak_absolute_normalized) for result in results)
    peak_map = dict(peaks)
    if gamma_band_hz not in peak_map:
        raise ValueError("Figure 16 assessment requires the declared gamma band")
    lower = tuple(value for band, value in peaks if band[1] <= gamma_band_hz[0])
    if not lower:
        raise ValueError("Figure 16 assessment requires at least one lower-frequency band")
    strongest = max(peaks, key=lambda item: item[1])[0]
    return Figure16CrossCorrelationAssessment(
        peak_by_band=peaks,
        strongest_band_hz=strongest,
        lower_frequency_stronger_than_gamma=max(lower) > peak_map[gamma_band_hz],
    )
