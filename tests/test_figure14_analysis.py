from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from smart_robustness.analysis.figure14 import (
    assess_figure14_spectra,
    cumulative_spike_histogram,
    figure14_spectrum_from_histogram,
)


def _oscillatory_histogram(frequency_hz: float, amplitude: float = 4.0) -> np.ndarray:
    time_s = np.arange(1000) / 1000.0
    return 10.0 + amplitude * np.sin(2 * np.pi * frequency_hz * time_s)


def test_cumulative_spike_histogram_uses_half_open_trial_interval() -> None:
    histogram = cumulative_spike_histogram((0.0, 0.9, 1.0, 999.9))
    assert histogram.shape == (1000,)
    assert histogram[:2] == pytest.approx((2, 1))
    assert histogram[-1] == 1
    assert histogram.sum() == 4
    assert not histogram.flags.writeable


def test_figure14_hamming_spectrum_recovers_gamma_and_slow_peaks() -> None:
    gamma = figure14_spectrum_from_histogram(_oscillatory_histogram(40))
    slow = figure14_spectrum_from_histogram(_oscillatory_histogram(5))
    assert gamma.dominant_frequency_hz == pytest.approx(40)
    assert gamma.gamma_power > gamma.middle_caption_power
    assert slow.dominant_frequency_hz == pytest.approx(5)
    assert slow.low_power > slow.gamma_power
    assert not gamma.frequencies_hz.flags.writeable
    assert not gamma.power.flags.writeable


def test_figure14_assessment_requires_all_caption_directions() -> None:
    match = figure14_spectrum_from_histogram(_oscillatory_histogram(40, 5))
    mismatch = figure14_spectrum_from_histogram(_oscillatory_histogram(5, 4))
    assessment = assess_figure14_spectra(match, mismatch)
    assert assessment.match_gamma_dominant
    assert assessment.mismatch_lower_frequency_dominant
    assert assessment.mismatch_gamma_reduced
    assert assessment.reproduced


def test_figure14_peak_gate_is_not_biased_by_unequal_bandwidth() -> None:
    match = figure14_spectrum_from_histogram(_oscillatory_histogram(40, 5))
    mismatch = figure14_spectrum_from_histogram(_oscillatory_histogram(15, 5))
    # Integrated mass can favor a wider band without changing the location of
    # the spectral peak stated by the caption.
    mismatch = replace(
        mismatch,
        gamma_power=2 * mismatch.middle_caption_power,
    )
    assert mismatch.gamma_power > mismatch.middle_caption_power
    assert mismatch.dominant_frequency_hz == pytest.approx(15)
    assert assess_figure14_spectra(match, mismatch).mismatch_lower_frequency_dominant


def test_figure14_preserves_methods_and_caption_middle_bands() -> None:
    spectrum = figure14_spectrum_from_histogram(
        _oscillatory_histogram(10) + _oscillatory_histogram(15)
    )
    assert spectrum.middle_methods_power > 0
    assert spectrum.middle_caption_power > spectrum.middle_methods_power


@pytest.mark.parametrize(
    "spikes,kwargs",
    (
        ((-0.1,), {}),
        ((1000.0,), {}),
        ((0.0,), {"duration_ms": 1000, "bin_ms": 3}),
    ),
)
def test_cumulative_histogram_rejects_invalid_protocol(spikes, kwargs) -> None:
    with pytest.raises(ValueError):
        cumulative_spike_histogram(spikes, **kwargs)
