import numpy as np
import pytest

from smart_robustness.analysis.figure15_phase import (
    population_gamma_spectrum,
    population_peak_lag,
)


def test_population_gamma_spectrum_recovers_50_hz_events() -> None:
    times = np.arange(10.0, 1000.0, 20.0)
    result = population_gamma_spectrum(
        times,
        spike_indices=np.zeros(times.size, dtype=int),
    )

    assert result.event_count == 50
    assert result.active_cell_count == 1
    assert result.gamma_peak_hz == pytest.approx(50.0)
    assert result.frequency_resolution_hz == pytest.approx(1.0)


def test_population_peak_lag_reports_target_delay_as_positive() -> None:
    source = np.arange(10.0, 990.0, 20.0)
    target = source + 3.0

    result = population_peak_lag(source, target, max_abs_lag_ms=8.0)

    assert result.peak_lag_ms == pytest.approx(3.0)
    assert result.peak_cross_covariance > 0


def test_population_diagnostics_reject_invalid_events() -> None:
    with pytest.raises(ValueError, match="half-open"):
        population_gamma_spectrum([1000.0])
    with pytest.raises(ValueError, match="equal-length"):
        population_gamma_spectrum([1.0], spike_indices=[])
    with pytest.raises(ValueError, match="maximum lag"):
        population_peak_lag([1.0], [2.0], max_abs_lag_ms=0.0)
