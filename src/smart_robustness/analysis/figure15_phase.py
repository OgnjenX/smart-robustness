"""Population-frequency and phase diagnostics for SMART Figure 15."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate, correlation_lags, periodogram


@dataclass(frozen=True, slots=True)
class PopulationGammaSpectrum:
    event_count: int
    active_cell_count: int | None
    gamma_peak_hz: float
    frequency_resolution_hz: float


@dataclass(frozen=True, slots=True)
class PopulationLag:
    peak_lag_ms: float
    peak_cross_covariance: float


def _population_histogram(
    spike_times_ms: tuple[float, ...] | np.ndarray,
    *,
    duration_ms: float,
    bin_ms: float,
) -> np.ndarray:
    times = np.asarray(spike_times_ms, dtype=float)
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError("spike times must be a finite one-dimensional vector")
    if duration_ms <= 0 or bin_ms <= 0:
        raise ValueError("duration and bin width must be positive")
    bin_count_float = duration_ms / bin_ms
    bin_count = round(bin_count_float)
    if not np.isclose(bin_count_float, bin_count, rtol=0, atol=1e-12):
        raise ValueError("duration must be an integer multiple of bin width")
    if np.any(times < 0) or np.any(times >= duration_ms):
        raise ValueError("spike times must lie in the half-open trial interval")
    return np.histogram(times, bins=np.arange(bin_count + 1) * bin_ms)[0].astype(float)


def population_gamma_spectrum(
    spike_times_ms: tuple[float, ...] | np.ndarray,
    *,
    spike_indices: tuple[int, ...] | np.ndarray | None = None,
    duration_ms: float = 1000.0,
    bin_ms: float = 1.0,
    gamma_band_hz: tuple[float, float] = (20.0, 70.0),
) -> PopulationGammaSpectrum:
    """Return the full-epoch Hamming-periodogram peak of population events."""

    times = np.asarray(spike_times_ms, dtype=float)
    counts = _population_histogram(times, duration_ms=duration_ms, bin_ms=bin_ms)
    frequencies, power = periodogram(
        counts - np.mean(counts),
        fs=1000.0 / bin_ms,
        window="hamming",
        detrend=False,
    )
    selected = (frequencies >= gamma_band_hz[0]) & (frequencies <= gamma_band_hz[1])
    gamma_peak = (
        float(frequencies[selected][np.argmax(power[selected])])
        if np.any(selected) and np.any(power[selected] > 0)
        else float("nan")
    )
    indices = None if spike_indices is None else np.asarray(spike_indices, dtype=int)
    if indices is not None and (indices.ndim != 1 or indices.size != times.size):
        raise ValueError("spike indices and times must be equal-length vectors")
    return PopulationGammaSpectrum(
        event_count=int(times.size),
        active_cell_count=None if indices is None else int(np.unique(indices).size),
        gamma_peak_hz=gamma_peak,
        frequency_resolution_hz=float(frequencies[1] - frequencies[0]),
    )


def population_peak_lag(
    source_spike_times_ms: tuple[float, ...] | np.ndarray,
    target_spike_times_ms: tuple[float, ...] | np.ndarray,
    *,
    duration_ms: float = 1000.0,
    bin_ms: float = 1.0,
    max_abs_lag_ms: float = 10.0,
) -> PopulationLag:
    """Find target-versus-source cross-covariance peak near zero lag.

    Positive lag means that the target population follows the source.
    """

    if max_abs_lag_ms <= 0:
        raise ValueError("maximum lag must be positive")
    source = _population_histogram(source_spike_times_ms, duration_ms=duration_ms, bin_ms=bin_ms)
    target = _population_histogram(target_spike_times_ms, duration_ms=duration_ms, bin_ms=bin_ms)
    cross = correlate(target - np.mean(target), source - np.mean(source), mode="full")
    lag_bins = correlation_lags(target.size, source.size, mode="full")
    selected = np.abs(lag_bins * bin_ms) <= max_abs_lag_ms
    selected_cross = cross[selected]
    selected_lags = lag_bins[selected]
    peak = int(np.argmax(selected_cross))
    return PopulationLag(
        peak_lag_ms=float(selected_lags[peak] * bin_ms),
        peak_cross_covariance=float(selected_cross[peak]),
    )
