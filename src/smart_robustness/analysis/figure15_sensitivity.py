"""Predeclared analysis-family audit for the incompletely specified Figure 15."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate, correlation_lags, periodogram

FIGURE15_GAMMA_BAND_HZ = (20.0, 70.0)


@dataclass(frozen=True, slots=True)
class Figure15MethodPeak:
    method: str
    gamma_peak_hz: float
    frequency_resolution_hz: float


def _binary_train(
    spike_times_ms: np.ndarray,
    *,
    duration_ms: float,
    bin_ms: float,
) -> np.ndarray:
    bin_count_float = duration_ms / bin_ms
    bin_count = round(bin_count_float)
    if not np.isclose(bin_count_float, bin_count, rtol=0, atol=1e-12):
        raise ValueError("duration must be an integer multiple of bin width")
    if np.any(spike_times_ms < 0) or np.any(spike_times_ms >= duration_ms):
        raise ValueError("spike times must lie in the half-open trial interval")
    train = np.zeros(bin_count, dtype=float)
    train[np.unique(np.floor(spike_times_ms / bin_ms).astype(int))] = 1.0
    return train - np.mean(train)


def _method_peak(
    method: str,
    frequencies_hz: np.ndarray,
    power: np.ndarray,
) -> Figure15MethodPeak:
    low_hz, high_hz = FIGURE15_GAMMA_BAND_HZ
    gamma = (frequencies_hz >= low_hz) & (frequencies_hz <= high_hz)
    if not np.any(gamma) or not np.any(power[gamma] > 0):
        peak = float("nan")
    else:
        peak = float(frequencies_hz[gamma][np.argmax(power[gamma])])
    resolution = (
        float(frequencies_hz[1] - frequencies_hz[0])
        if frequencies_hz.size > 1
        else float("nan")
    )
    return Figure15MethodPeak(
        method=method,
        gamma_peak_hz=peak,
        frequency_resolution_hz=resolution,
    )


def figure15_analysis_sensitivity(
    first_spike_times_ms: tuple[float, ...] | np.ndarray,
    second_spike_times_ms: tuple[float, ...] | np.ndarray,
    *,
    duration_ms: float = 1000.0,
    bin_ms: float = 1.0,
    display_max_lag_ms: float = 180.0,
    methods_hamming_window_ms: float = 200.0,
) -> tuple[Figure15MethodPeak, ...]:
    """Evaluate the complete predeclared family without selecting a member.

    The family spans the source-unresolved choices that can be stated without
    changing cells, spikes, duration, or binning: linear-correlogram support,
    tapering of that correlogram, direct full-epoch cross-spectrum, and the
    Methods 4.10 200-ms non-overlapping Hamming interpretation.
    """

    first_times = np.asarray(first_spike_times_ms, dtype=float)
    second_times = np.asarray(second_spike_times_ms, dtype=float)
    if first_times.ndim != 1 or second_times.ndim != 1:
        raise ValueError("spike times must be one-dimensional")
    if not np.all(np.isfinite(first_times)) or not np.all(np.isfinite(second_times)):
        raise ValueError("spike times must be finite")
    if duration_ms <= 0 or bin_ms <= 0 or display_max_lag_ms <= 0:
        raise ValueError("duration, bin width, and display lag must be positive")

    first = _binary_train(first_times, duration_ms=duration_ms, bin_ms=bin_ms)
    second = _binary_train(second_times, duration_ms=duration_ms, bin_ms=bin_ms)
    sample_rate_hz = 1000.0 / bin_ms
    full = correlate(first, second, mode="full", method="fft")
    full -= np.mean(full)
    lag_bins = correlation_lags(first.size, second.size, mode="full")
    display = full[np.abs(lag_bins * bin_ms) <= display_max_lag_ms]
    display -= np.mean(display)

    outcomes: list[Figure15MethodPeak] = []
    for support, values in (("full", full), ("displayed", display)):
        for window in ("hamming", "boxcar"):
            frequencies, power = periodogram(
                values,
                fs=sample_rate_hz,
                window=window,
                detrend=False,
            )
            outcomes.append(
                _method_peak(
                    f"linear_correlogram_{support}_{window}_periodogram",
                    frequencies,
                    power,
                )
            )

    frequencies = np.fft.rfftfreq(first.size, d=1.0 / sample_rate_hz)
    cross_spectrum = np.fft.rfft(first) * np.conj(np.fft.rfft(second))
    outcomes.append(
        _method_peak(
            "direct_full_epoch_cross_spectrum_power",
            frequencies,
            np.abs(cross_spectrum) ** 2,
        )
    )

    window_samples_float = methods_hamming_window_ms / bin_ms
    window_samples = round(window_samples_float)
    if not np.isclose(window_samples_float, window_samples, rtol=0, atol=1e-12):
        raise ValueError("Methods Hamming window must contain an integer number of bins")
    if window_samples < 8 or first.size % window_samples:
        raise ValueError("Methods Hamming window must tile the epoch without overlap")
    taper = np.hamming(window_samples)
    segmented_power = []
    for start in range(0, first.size, window_samples):
        stop = start + window_samples
        first_segment = first[start:stop] - np.mean(first[start:stop])
        second_segment = second[start:stop] - np.mean(second[start:stop])
        cross = np.fft.rfft(first_segment * taper) * np.conj(
            np.fft.rfft(second_segment * taper)
        )
        segmented_power.append(np.abs(cross) ** 2)
    frequencies = np.fft.rfftfreq(window_samples, d=1.0 / sample_rate_hz)
    outcomes.append(
        _method_peak(
            "methods_4_10_200ms_nonoverlap_cross_spectrum_power",
            frequencies,
            np.mean(np.asarray(segmented_power), axis=0),
        )
    )
    return tuple(outcomes)
