import numpy as np

from smart_robustness.validation.isolated_cells import (
    IsolatedCellTrace,
    assess_figure8,
)


def _trace(condition: str, spikes: list[float]) -> IsolatedCellTrace:
    return IsolatedCellTrace(
        condition=condition,
        time_ms=np.arange(301, dtype=float),
        soma_voltage_mV=np.full(301, -60.0),
        spike_times_ms=np.asarray(spikes, dtype=float),
    )


def test_figure8_assessment_accepts_tonic_and_transient_burst_signatures() -> None:
    tonic = _trace("depolarized", [20, 70, 120, 170, 220, 270])
    burst = _trace("hyperpolarized", [20, 24, 29, 35])
    assessment = assess_figure8(tonic, burst)
    assert assessment.reproduced
    assert assessment.tonic_spike_count == 6
    assert assessment.burst_spike_count == 4


def test_figure8_assessment_rejects_two_sustained_tonic_trains() -> None:
    tonic = _trace("depolarized", [20, 70, 120, 170, 220, 270])
    not_a_burst = _trace("hyperpolarized", [20, 70, 120, 170, 220, 270])
    assessment = assess_figure8(tonic, not_a_burst)
    assert not assessment.reproduced
    assert assessment.tonic_pass
    assert not assessment.burst_pass
    assert assessment.notes
