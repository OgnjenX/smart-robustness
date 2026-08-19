import numpy as np
import pytest

from smart_robustness.validation.isolated_cells import (
    Figure8Protocol,
    IsolatedCellTrace,
    Layer5PropagationProtocol,
    Layer23TransferProtocol,
    TrnRecruitmentProtocol,
    assess_figure8,
    figure8_source_parameters,
    figure8_voltage_peak_times_ms,
    run_layer5_propagation_condition,
    run_layer23_transfer_condition,
    run_trn_recruitment_condition,
)


def _trace(condition: str, spikes: list[float]) -> IsolatedCellTrace:
    voltage = np.full(301, -60.0)
    for spike in spikes:
        voltage[int(spike)] = 20.0
    return IsolatedCellTrace(
        condition=condition,
        time_ms=np.arange(301, dtype=float),
        soma_voltage_mV=voltage,
        spike_times_ms=np.asarray(spikes, dtype=float),
    )


def test_figure8_assessment_accepts_tonic_and_transient_burst_signatures() -> None:
    tonic = _trace("depolarized", [20, 70, 120, 170, 220, 270])
    burst = _trace("hyperpolarized", [20, 24, 29, 35])
    assessment = assess_figure8(tonic, burst)
    assert assessment.reproduced
    assert assessment.tonic_spike_count == 6
    assert assessment.burst_spike_count == 4
    assert assessment.tonic_release_event_count == 6
    assert assessment.burst_release_event_count == 4


def test_figure8_assessment_rejects_two_sustained_tonic_trains() -> None:
    tonic = _trace("depolarized", [20, 70, 120, 170, 220, 270])
    not_a_burst = _trace("hyperpolarized", [20, 70, 120, 170, 220, 270])
    assessment = assess_figure8(tonic, not_a_burst)
    assert not assessment.reproduced
    assert assessment.tonic_pass
    assert not assessment.burst_pass
    assert assessment.notes


def test_figure8_voltage_peak_detector_uses_trace_not_release_events() -> None:
    trace = _trace("depolarized", [20, 70, 120])
    trace = IsolatedCellTrace(
        condition=trace.condition,
        time_ms=trace.time_ms,
        soma_voltage_mV=trace.soma_voltage_mV,
        spike_times_ms=np.asarray([20.0]),
    )
    assert figure8_voltage_peak_times_ms(trace).tolist() == [20.0, 70.0, 120.0]
    assessment = assess_figure8(trace, _trace("hyperpolarized", [20, 24, 29]))
    assert assessment.tonic_spike_count == 3
    assert assessment.tonic_release_event_count == 1


def test_figure8_caption_protocol_requires_finite_hyperpolarizing_conductance() -> None:
    with pytest.raises(ValueError, match="positive clamp"):
        Figure8Protocol(clamp_interpretation="caption_finite_conductance")
    protocol = Figure8Protocol(
        clamp_interpretation="caption_finite_conductance",
        hyperpolarizing_clamp_conductance_nS=10,
    )
    assert protocol.hyperpolarizing_clamp_conductance_nS == pytest.approx(10)


def test_figure8_source_parameters_require_explicit_missing_defaults() -> None:
    with pytest.raises(ValueError, match="positive"):
        figure8_source_parameters(
            leak_density_mS_cm2=0,
            specific_capacitance_uF_cm2=1,
        )
    params = figure8_source_parameters(
        leak_density_mS_cm2=0.1,
        specific_capacitance_uF_cm2=1.5,
        geometry_convention="millimeters",
    )
    assert params["cell_spec"].name == "modeldb112923_figure8_relay"
    assert params["specific_capacitance_uF_cm2"] == pytest.approx(1.5)
    assert params["e_ca_mV"] == pytest.approx(180)
    assert params["cell_spec"].soma.diameter_mm == pytest.approx(0.02)


def test_trn_recruitment_protocol_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="durations"):
        TrnRecruitmentProtocol(drive_ms=0)
    with pytest.raises(ValueError, match="gates"):
        TrnRecruitmentProtocol(layer6ii_ampa_gate=-1)
    with pytest.raises(ValueError, match="scales"):
        TrnRecruitmentProtocol(axial_conductance_scale=0)


def test_layer5_propagation_protocol_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="durations"):
        Layer5PropagationProtocol(drive_ms=0)
    with pytest.raises(ValueError, match="gates"):
        Layer5PropagationProtocol(nonspecific_ampa_gate=-1)
    with pytest.raises(ValueError, match="scales"):
        Layer5PropagationProtocol(axial_conductance_scale=0)


def test_layer23_transfer_protocol_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="durations"):
        Layer23TransferProtocol(drive_ms=0)
    with pytest.raises(ValueError, match="gates"):
        Layer23TransferProtocol(excitation_gate=-1)
    with pytest.raises(ValueError, match="inhibition delay"):
        Layer23TransferProtocol(drive_ms=1, inhibition_delay_ms=2)


def test_layer23_transfer_runner_applies_declared_gates() -> None:
    brian = pytest.importorskip("brian2")
    brian.prefs.codegen.target = "numpy"
    protocol = Layer23TransferProtocol(
        pre_drive_ms=0.01,
        drive_ms=0.01,
        excitation_gate=0.5,
        inhibition_gate=0.25,
        include_inhibition=True,
        inhibition_delay_ms=0.0,
    )
    result = run_layer23_transfer_condition(protocol=protocol, brian=brian)
    assert result.finite
    assert result.excitation_port != result.inhibition_port
    assert result.applied_excitation_gate == pytest.approx(0.5)
    assert result.applied_inhibition_gate == pytest.approx(0.25)


def test_layer5_propagation_runner_reports_source_cell_voltage_ranges() -> None:
    brian = pytest.importorskip("brian2")
    brian.prefs.codegen.target = "numpy"
    result = run_layer5_propagation_condition(
        protocol=Layer5PropagationProtocol(
            pre_drive_ms=0.01,
            drive_ms=0.01,
            dt_ms=0.01,
            drive_multiplier=0,
        ),
        brian=brian,
    )
    assert result.finite
    assert result.convention_fingerprint
    assert result.axial_conductance_scale == pytest.approx(1)
    assert result.drive_multiplier == pytest.approx(0)


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
    assert result.applied_relay_ampa_gate == 0


def test_trn_recruitment_runner_applies_declared_driven_gates() -> None:
    brian = pytest.importorskip("brian2")
    brian.prefs.codegen.target = "numpy"
    protocol = TrnRecruitmentProtocol(
        pre_drive_ms=0.01,
        drive_ms=0.01,
        layer6ii_ampa_gate=0.5,
        layer6ii_nmda_gate=0.25,
        relay_ampa_gate=2.0,
        drive_multiplier=3,
    )
    result = run_trn_recruitment_condition(driven=True, protocol=protocol, brian=brian)
    assert result.driven
    assert result.applied_layer6ii_ampa_gate == pytest.approx(1.5)
    assert result.applied_layer6ii_nmda_gate == pytest.approx(0.75)
    assert result.applied_relay_ampa_gate == pytest.approx(6.0)


def test_figure8_source_parameters_expose_legacy_calcium_unit_candidate() -> None:
    from smart_robustness.validation.isolated_cells import figure8_source_parameters

    params = figure8_source_parameters(
        leak_density_mS_cm2=0.1,
        specific_capacitance_uF_cm2=1.0,
        calcium_density_mS_cm2=0.25,
    )
    assert {compartment.g_ca_mS_cm2 for compartment in params["cell_spec"].compartments} == {
        0.25
    }
