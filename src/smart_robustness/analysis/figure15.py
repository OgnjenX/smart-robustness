"""Single-unit layer-4 synchrony analysis for SMART Figure 15."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate, correlation_lags, periodogram


@dataclass(frozen=True, slots=True)
class Figure15Synchrony:
    first_cell_index: int
    second_cell_index: int
    first_spike_count: int
    second_spike_count: int
    lag_ms: np.ndarray
    cross_correlation: np.ndarray
    frequencies_hz: np.ndarray
    power: np.ndarray
    gamma_peak_hz: float


@dataclass(frozen=True, slots=True)
class Figure15Assessment:
    enough_spikes: bool
    gamma_peak_hz: float
    target_hz: float = 44.0
    tolerance_hz: float = 5.0

    @property
    def reproduced(self) -> bool:
        return self.enough_spikes and abs(self.gamma_peak_hz - self.target_hz) <= self.tolerance_hz


def _binary_spike_train(
    spike_times_ms: np.ndarray,
    *,
    duration_ms: float,
    bin_ms: float,
) -> np.ndarray:
    bin_count_float = duration_ms / bin_ms
    bin_count = round(bin_count_float)
    if not np.isclose(bin_count_float, bin_count, rtol=0, atol=1e-12):
        raise ValueError("Figure 15 duration must be an integer multiple of bin width")
    if np.any(spike_times_ms < 0) or np.any(spike_times_ms >= duration_ms):
        raise ValueError("Figure 15 spike times must lie in the half-open trial interval")
    bins = np.floor(spike_times_ms / bin_ms).astype(int)
    train = np.zeros(bin_count, dtype=float)
    train[np.unique(bins)] = 1.0
    return train


def figure15_layer4_synchrony(
    spike_indices: tuple[int, ...] | np.ndarray,
    spike_times_ms: tuple[float, ...] | np.ndarray,
    *,
    first_cell_index: int = 39,
    second_cell_index: int = 40,
    duration_ms: float = 1000.0,
    bin_ms: float = 1.0,
    max_lag_ms: float = 180.0,
) -> Figure15Synchrony:
    """Cross-correlate a predeclared adjacent layer-4 pair and analyze its PSD."""

    indices = np.asarray(spike_indices, dtype=int)
    times = np.asarray(spike_times_ms, dtype=float)
    if indices.ndim != 1 or times.ndim != 1 or indices.size != times.size:
        raise ValueError("Figure 15 spike indices and times must be equal-length vectors")
    if not np.all(np.isfinite(times)):
        raise ValueError("Figure 15 spike times must be finite")
    if first_cell_index == second_cell_index or not (
        0 <= first_cell_index < 81 and 0 <= second_cell_index < 81
    ):
        raise ValueError("Figure 15 requires two distinct layer-4 cell indices")
    if duration_ms <= 0 or bin_ms <= 0 or max_lag_ms <= 0:
        raise ValueError("Figure 15 duration, bin width, and lag must be positive")
    first_times = times[indices == first_cell_index]
    second_times = times[indices == second_cell_index]
    first = _binary_spike_train(first_times, duration_ms=duration_ms, bin_ms=bin_ms)
    second = _binary_spike_train(second_times, duration_ms=duration_ms, bin_ms=bin_ms)
    first -= np.mean(first)
    second -= np.mean(second)
    full = correlate(first, second, mode="full", method="fft")
    lag_bins = correlation_lags(first.size, second.size, mode="full")
    selected = np.abs(lag_bins * bin_ms) <= max_lag_ms
    cross_correlation = full[selected]
    lag_ms = lag_bins[selected].astype(float) * bin_ms
    frequencies, power = periodogram(
        cross_correlation - np.mean(cross_correlation),
        fs=1000.0 / bin_ms,
        window="hamming",
        detrend=False,
    )
    gamma = (frequencies >= 20.0) & (frequencies <= 70.0)
    gamma_peak = (
        float(frequencies[gamma][np.argmax(power[gamma])])
        if np.any(gamma) and np.any(power[gamma] > 0)
        else float("nan")
    )
    for array in (lag_ms, cross_correlation, frequencies, power):
        array.setflags(write=False)
    return Figure15Synchrony(
        first_cell_index=first_cell_index,
        second_cell_index=second_cell_index,
        first_spike_count=int(first_times.size),
        second_spike_count=int(second_times.size),
        lag_ms=lag_ms,
        cross_correlation=cross_correlation,
        frequencies_hz=frequencies,
        power=power,
        gamma_peak_hz=gamma_peak,
    )


def assess_figure15_synchrony(
    result: Figure15Synchrony,
    *,
    target_hz: float = 44.0,
    tolerance_hz: float = 5.0,
) -> Figure15Assessment:
    if tolerance_hz < 0 or not np.isfinite(target_hz):
        raise ValueError("Figure 15 target and tolerance must be valid")
    return Figure15Assessment(
        enough_spikes=result.first_spike_count >= 2 and result.second_spike_count >= 2,
        gamma_peak_hz=result.gamma_peak_hz,
        target_hz=target_hz,
        tolerance_hz=tolerance_hz,
    )
