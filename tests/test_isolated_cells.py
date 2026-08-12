import numpy as np
import pytest

from smart_robustness.validation.isolated_cells import (
    IsolatedCellTrace,
    TrnRecruitmentProtocol,
    assess_figure8,
    run_trn_recruitment_condition,
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


def test_trn_recruitment_protocol_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="durations"):
        TrnRecruitmentProtocol(drive_ms=0)
    with pytest.raises(ValueError, match="gates"):
        TrnRecruitmentProtocol(layer6ii_ampa_gate=-1)
    with pytest.raises(ValueError, match="scales"):
        TrnRecruitmentProtocol(axial_conductance_scale=0)


def test_trn_recruitment_runner_reports_independent_control_trial() -> None:
    brian = pytest.importorskip("brian2")
    brian.prefs.codegen.target = "numpy"
    result = run_trn_recruitment_condition(
        driven=False,
        protocol=TrnRecruitmentProtocol(pre_drive_ms=0.02, drive_ms=0.03),
        brian=brian,
    )
    assert not result.driven
    assert result.finite
    assert result.post_drive_spike_count == len(result.spike_times_ms)
    assert result.soma_voltage_range_mV[0] <= result.soma_voltage_range_mV[1]
    assert result.proximal_voltage_range_mV[0] <= result.proximal_voltage_range_mV[1]
    assert len(result.convention_fingerprint) == 64
    assert result.applied_layer6ii_ampa_gate == 0
    assert result.applied_layer6ii_nmda_gate == 0


def test_trn_recruitment_runner_applies_declared_driven_gates() -> None:
    brian = pytest.importorskip("brian2")
    brian.prefs.codegen.target = "numpy"
    protocol = TrnRecruitmentProtocol(
        pre_drive_ms=0.01,
        drive_ms=0.01,
        layer6ii_ampa_gate=0.5,
        layer6ii_nmda_gate=0.25,
        drive_multiplier=3,
    )
    result = run_trn_recruitment_condition(driven=True, protocol=protocol, brian=brian)
    assert result.driven
    assert result.applied_layer6ii_ampa_gate == pytest.approx(1.5)
    assert result.applied_layer6ii_nmda_gate == pytest.approx(0.75)
