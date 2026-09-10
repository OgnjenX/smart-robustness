from __future__ import annotations

import numpy as np
import pytest

from smart_robustness.analysis.figure15_sensitivity import (
    direct_cross_spectrum_gamma_peak,
    figure15_analysis_sensitivity,
)


def test_figure15_sensitivity_reports_every_predeclared_method() -> None:
    period_ms = 1000.0 / 44.0
    first = np.arange(10.0, 990.0, period_ms)
    second = first + 2.0
    outcomes = figure15_analysis_sensitivity(first, second)
    by_method = {outcome.method: outcome for outcome in outcomes}

    assert set(by_method) == {
        "linear_correlogram_full_hamming_periodogram",
        "linear_correlogram_full_boxcar_periodogram",
        "linear_correlogram_displayed_hamming_periodogram",
        "linear_correlogram_displayed_boxcar_periodogram",
        "direct_full_epoch_cross_spectrum_power",
        "methods_4_10_200ms_nonoverlap_cross_spectrum_power",
    }
    assert by_method[
        "linear_correlogram_full_hamming_periodogram"
    ].gamma_peak_hz == pytest.approx(44.0220, abs=0.01)
    assert by_method["direct_full_epoch_cross_spectrum_power"].gamma_peak_hz == 44.0
    assert by_method[
        "methods_4_10_200ms_nonoverlap_cross_spectrum_power"
    ].frequency_resolution_hz == 5.0
    assert all(20.0 <= outcome.gamma_peak_hz <= 70.0 for outcome in outcomes)


def test_figure15_sensitivity_does_not_accept_non_tiling_methods_window() -> None:
    with pytest.raises(ValueError, match="tile"):
        figure15_analysis_sensitivity(
            np.array([10.0, 30.0]),
            np.array([11.0, 31.0]),
            methods_hamming_window_ms=300.0,
        )


def test_direct_cross_spectrum_has_five_hz_resolution_for_diagnostic_epoch() -> None:
    first = np.arange(10.0, 200.0, 20.0)
    second = first + 1.0

    outcome = direct_cross_spectrum_gamma_peak(
        first,
        second,
        duration_ms=200.0,
        bin_ms=1.0,
    )

    assert outcome.method == "direct_full_epoch_cross_spectrum_power"
    assert outcome.gamma_peak_hz == 50.0
    assert outcome.frequency_resolution_hz == 5.0
