"""Predeclared cumulative-spike spectrum analysis for SMART Figure 14."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch

FIGURE14_LOW_BAND_HZ = (2.0, 8.0)
FIGURE14_MIDDLE_CAPTION_BAND_HZ = (8.0, 20.0)
FIGURE14_MIDDLE_METHODS_BAND_HZ = (8.0, 10.0)
FIGURE14_GAMMA_BAND_HZ = (20.0, 70.0)


@dataclass(frozen=True, slots=True)
class Figure14Spectrum:
    histogram_bin_ms: float
    hamming_window_ms: float
    frequencies_hz: np.ndarray
    power: np.ndarray
    low_power: float
    middle_caption_power: float
    middle_methods_power: float
    gamma_power: float
    dominant_frequency_hz: float


@dataclass(frozen=True, slots=True)
class Figure14Assessment:
    match_gamma_dominant: bool
    mismatch_lower_frequency_dominant: bool
    mismatch_gamma_reduced: bool

    @property
    def reproduced(self) -> bool:
        return (
            self.match_gamma_dominant
            and self.mismatch_lower_frequency_dominant
            and self.mismatch_gamma_reduced
        )


def cumulative_spike_histogram(
    spike_times_ms: np.ndarray | tuple[float, ...],
    *,
    duration_ms: float = 1000.0,
    bin_ms: float = 1.0,
) -> np.ndarray:
    """Bin spikes from every cortical cell into one cumulative histogram.

    Figure 14 does not report the histogram bin width. One millisecond is the
    predeclared reconstruction default and remains part of every result.
    """

    times = np.asarray(spike_times_ms, dtype=float)
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError("spike times must be a finite one-dimensional array")
    if not np.isfinite(duration_ms) or not np.isfinite(bin_ms):
        raise ValueError("histogram duration and bin width must be finite")
    if duration_ms <= 0 or bin_ms <= 0:
        raise ValueError("histogram duration and bin width must be positive")
    bin_count_float = duration_ms / bin_ms
    bin_count = round(bin_count_float)
    if not np.isclose(bin_count_float, bin_count, rtol=0, atol=1e-12):
        raise ValueError("histogram duration must be an integer multiple of bin width")
    if np.any(times < 0) or np.any(times >= duration_ms):
        raise ValueError("spike times must lie in the half-open trial interval")
    edges = np.linspace(0.0, duration_ms, bin_count + 1)
    histogram, _ = np.histogram(times, bins=edges)
    values = histogram.astype(float)
    values.setflags(write=False)
    return values


def _band_mass(
    frequencies_hz: np.ndarray,
    power: np.ndarray,
    band_hz: tuple[float, float],
) -> float:
    low, high = band_hz
    mask = (frequencies_hz >= low) & (frequencies_hz <= high)
    if not np.any(mask):
        return 0.0
    frequency_step = float(frequencies_hz[1] - frequencies_hz[0])
    return float(np.sum(power[mask]) * frequency_step)


def figure14_spectrum_from_histogram(
    histogram: np.ndarray,
    *,
    histogram_bin_ms: float = 1.0,
    hamming_window_ms: float = 200.0,
) -> Figure14Spectrum:
    """Apply the Methods 4.10 mean subtraction and 200-ms Hamming analysis.

    Window overlap is not reported. The classic reconstruction uses contiguous,
    non-overlapping windows and records that choice rather than selecting an
    overlap after inspecting the result.
    """

    values = np.asarray(histogram, dtype=float)
    if values.ndim != 1 or values.size < 8 or not np.all(np.isfinite(values)):
        raise ValueError("histogram must be a finite one-dimensional array")
    if histogram_bin_ms <= 0 or hamming_window_ms <= 0:
        raise ValueError("Figure 14 bin and window widths must be positive")
    sample_rate_hz = 1000.0 / histogram_bin_ms
    window_samples_float = hamming_window_ms / histogram_bin_ms
    window_samples = round(window_samples_float)
    if not np.isclose(window_samples_float, window_samples, rtol=0, atol=1e-12):
        raise ValueError("Hamming window must contain an integer number of bins")
    if window_samples < 8 or window_samples > values.size:
        raise ValueError("Hamming window must span between 8 bins and the full histogram")
    centered = values - np.mean(values)
    frequencies, power = welch(
        centered,
        fs=sample_rate_hz,
        window="hamming",
        nperseg=window_samples,
        noverlap=0,
        detrend=False,
        scaling="density",
    )
    physiological = (frequencies >= FIGURE14_LOW_BAND_HZ[0]) & (
        frequencies <= FIGURE14_GAMMA_BAND_HZ[1]
    )
    dominant = (
        float(frequencies[physiological][np.argmax(power[physiological])])
        if np.any(physiological)
        else float("nan")
    )
    frequencies.setflags(write=False)
    power.setflags(write=False)
    return Figure14Spectrum(
        histogram_bin_ms=float(histogram_bin_ms),
        hamming_window_ms=float(hamming_window_ms),
        frequencies_hz=frequencies,
        power=power,
        low_power=_band_mass(frequencies, power, FIGURE14_LOW_BAND_HZ),
        middle_caption_power=_band_mass(
            frequencies, power, FIGURE14_MIDDLE_CAPTION_BAND_HZ
        ),
        middle_methods_power=_band_mass(
            frequencies, power, FIGURE14_MIDDLE_METHODS_BAND_HZ
        ),
        gamma_power=_band_mass(frequencies, power, FIGURE14_GAMMA_BAND_HZ),
        dominant_frequency_hz=dominant,
    )


def figure14_spectrum_from_spikes(
    spike_times_ms: np.ndarray | tuple[float, ...],
    *,
    duration_ms: float = 1000.0,
    histogram_bin_ms: float = 1.0,
    hamming_window_ms: float = 200.0,
) -> Figure14Spectrum:
    histogram = cumulative_spike_histogram(
        spike_times_ms,
        duration_ms=duration_ms,
        bin_ms=histogram_bin_ms,
    )
    return figure14_spectrum_from_histogram(
        histogram,
        histogram_bin_ms=histogram_bin_ms,
        hamming_window_ms=hamming_window_ms,
    )


def assess_figure14_spectra(
    match: Figure14Spectrum,
    mismatch: Figure14Spectrum,
) -> Figure14Assessment:
    """Score the three qualitative directions stated in the Figure 14 caption."""

    if match.histogram_bin_ms != mismatch.histogram_bin_ms:
        raise ValueError("Figure 14 conditions must use the same histogram bin width")
    if match.hamming_window_ms != mismatch.hamming_window_ms:
        raise ValueError("Figure 14 conditions must use the same Hamming window")
    return Figure14Assessment(
        match_gamma_dominant=(
            FIGURE14_GAMMA_BAND_HZ[0]
            <= match.dominant_frequency_hz
            <= FIGURE14_GAMMA_BAND_HZ[1]
        ),
        mismatch_lower_frequency_dominant=(
            FIGURE14_LOW_BAND_HZ[0]
            <= mismatch.dominant_frequency_hz
            < FIGURE14_GAMMA_BAND_HZ[0]
        ),
        mismatch_gamma_reduced=mismatch.gamma_power < match.gamma_power,
    )
