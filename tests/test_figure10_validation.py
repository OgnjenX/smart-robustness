from __future__ import annotations

from dataclasses import replace

import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.classic_sector import figure6_runtime_conventions
from smart_robustness.validation.figure10 import (
    Figure10ConditionResult,
    assess_figure10_reset,
    run_figure10_condition,
    summarize_layer6i_selected_traces,
)
from smart_robustness.validation.layer6i_replay import run_layer6i_replay


def _condition(
    *,
    enabled: bool,
    pre_indices=(40, 40, 40),
    post_indices=(40, 41, 42),
    chain=True,
) -> Figure10ConditionResult:
    indices = pre_indices + post_indices
    times = tuple(float(i + 1) for i in range(len(pre_indices))) + tuple(
        float(101 + i) for i in range(len(post_indices))
    )
    pathway_times = (110.0,) if chain else ()
    return Figure10ConditionResult(
        pre_match_duration_ms=100,
        mismatch_duration_ms=100,
        reset_pathway_enabled=enabled,
        layer4_spike_indices=indices,
        layer4_spike_times_ms=times,
        nonspecific_spike_times_ms=pathway_times,
        layer5_spike_indices=((40,) if chain else ()),
        layer5_spike_times_ms=pathway_times,
        layer6i_spike_times_ms=pathway_times,
    )


def test_figure10_assessment_requires_causal_suppression_and_release() -> None:
    intact = _condition(enabled=True, post_indices=(41, 42))
    control = _condition(enabled=False, post_indices=(40, 40, 41))
    assessment = assess_figure10_reset(intact, control)
    assert assessment.pre_reset_winner_index == 40
    assert assessment.pre_reset_winner_indices == (40,)
    assert assessment.pre_reset_winner_pass
    assert assessment.reset_chain_pass
    assert assessment.winner_suppression_pass
    assert assessment.alternative_release_pass
    assert assessment.reproduced_reset


def test_figure10_assessment_rejects_missing_pre_reset_winner() -> None:
    intact = _condition(enabled=True, pre_indices=(), post_indices=())
    control = _condition(enabled=False, pre_indices=(), post_indices=())
    assessment = assess_figure10_reset(intact, control)
    assert not assessment.pre_reset_winner_pass
    assert not assessment.reproduced_reset


def test_figure10_assessment_treats_the_pre_reset_bar_as_one_assembly() -> None:
    pre = (38, 39, 40, 41, 42)
    intact = _condition(enabled=True, pre_indices=pre, post_indices=(10, 11))
    control = _condition(enabled=False, pre_indices=pre, post_indices=(38, 40, 10))
    assessment = assess_figure10_reset(intact, control)
    assert assessment.pre_reset_winner_indices == pre
    assert assessment.pre_reset_winner_spikes == 5
    assert assessment.intact_winner_post_spikes == 0
    assert assessment.control_winner_post_spikes == 2
    assert assessment.intact_released_alternatives == 2
    assert assessment.control_released_alternatives == 1
    assert assessment.reproduced_reset


def test_figure10_assessment_rejects_nonidentical_pre_states() -> None:
    with pytest.raises(ValueError, match="pre-mismatch"):
        assess_figure10_reset(
            _condition(enabled=True, pre_indices=(40,)),
            _condition(enabled=False, pre_indices=(41,)),
        )


def test_figure10_runner_requires_explicit_positive_protocol_values() -> None:
    with pytest.raises(ValueError, match="durations"):
        run_figure10_condition(
            top_down_current_pA=600,
            pre_match_duration_ms=0,
            mismatch_duration_ms=100,
            reset_pathway_enabled=True,
        )
    with pytest.raises(ValueError, match="requires learned weights"):
        run_figure10_condition(
            top_down_current_pA=600,
            pre_match_duration_ms=100,
            mismatch_duration_ms=100,
            reset_pathway_enabled=True,
            comparator_top_k_targets=5,
        )
    with pytest.raises(ValueError, match="event-count-limited"):
        run_figure10_condition(
            top_down_current_pA=600,
            pre_match_duration_ms=100,
            mismatch_duration_ms=100,
            reset_pathway_enabled=True,
            top_down_current_mode="until_cued_cell_event_limit",
        )
    with pytest.raises(TypeError, match="diagnostics flag"):
        run_figure10_condition(
            top_down_current_pA=600,
            pre_match_duration_ms=100,
            mismatch_duration_ms=100,
            reset_pathway_enabled=True,
            record_layer6i_diagnostics=1,
        )
    with pytest.raises(TypeError, match="reset-chain diagnostics"):
        run_figure10_condition(
            top_down_current_pA=600,
            pre_match_duration_ms=100,
            mismatch_duration_ms=100,
            reset_pathway_enabled=True,
            record_reset_chain_diagnostics=1,
        )
    with pytest.raises(ValueError, match="require layer-6I diagnostics"):
        run_figure10_condition(
            top_down_current_pA=600,
            pre_match_duration_ms=100,
            mismatch_duration_ms=100,
            reset_pathway_enabled=True,
            record_layer6i_trace_indices=(0,),
        )
    with pytest.raises(ValueError, match="unique"):
        run_figure10_condition(
            top_down_current_pA=600,
            pre_match_duration_ms=100,
            mismatch_duration_ms=100,
            reset_pathway_enabled=True,
            record_layer6i_diagnostics=True,
            record_layer6i_trace_indices=(0, 0),
        )
    with pytest.raises(ValueError, match="between 0 and 80"):
        run_figure10_condition(
            top_down_current_pA=600,
            pre_match_duration_ms=100,
            mismatch_duration_ms=100,
            reset_pathway_enabled=True,
            record_layer6i_diagnostics=True,
            record_layer6i_trace_indices=(81,),
        )
    with pytest.raises(ValueError, match="parent directory"):
        run_figure10_condition(
            top_down_current_pA=600,
            pre_match_duration_ms=100,
            mismatch_duration_ms=100,
            reset_pathway_enabled=True,
            layer6i_replay_trace_output="missing/trace.npz",
        )
    with pytest.raises(ValueError, match="top_down_current"):
        run_figure10_condition(
            top_down_current_pA=0,
            pre_match_duration_ms=100,
            mismatch_duration_ms=100,
            reset_pathway_enabled=True,
        )


def test_figure10_condition_smoke_runs_persistent_two_phase_network() -> None:
    result = run_figure10_condition(
        top_down_current_pA=600,
        pre_match_duration_ms=0.01,
        mismatch_duration_ms=0.01,
        reset_pathway_enabled=False,
        dt_ms=0.01,
        brian=brian,
    )
    assert not result.reset_pathway_enabled
    assert result.pre_match_duration_ms == pytest.approx(0.01)
    assert result.mismatch_duration_ms == pytest.approx(0.01)
    assert result.learned_state_provenance == "paper-constrained-figure6c-reference"
    assert result.top_down_current_mode == "sustained_epoch"
    assert result.layer6i_mismatch_gate_integral_ms_by_projection == ()
    assert result.layer6i_mismatch_event_transmitter_samples == ()
    assert result.layer6i_mismatch_projection025_gate_integral_ms_by_index == ()
    assert result.layer6i_mismatch_soma_voltage_peak_mV_by_index == ()
    assert result.layer6i_selected_trace_summaries == ()
    assert result.layer6i_replay_trace_path is None
    assert result.layer6i_replay_trace_sha256 is None
    assert result.layer4i_mismatch_projection026_gate_integral_ms is None


def test_figure10_condition_captures_lossless_layer6i_replay(tmp_path) -> None:
    trace = tmp_path / "layer6i.npz"
    conventions = replace(
        figure6_runtime_conventions(),
        spike_event_coordinate="absolute_physical",
        spike_event_rule="falling_threshold_crossing",
        spike_event_threshold_mV=-20.0,
    )
    result = run_figure10_condition(
        top_down_current_pA=600,
        pre_match_duration_ms=0.01,
        mismatch_duration_ms=0.02,
        reset_pathway_enabled=True,
        dt_ms=0.01,
        conventions=conventions,
        layer6i_replay_trace_output=trace,
        layer6i_replay_cell_index=0,
        brian=brian,
    )

    assert result.layer6i_replay_trace_path == str(trace)
    assert result.layer6i_replay_trace_sha256
    replay = run_layer6i_replay(trace, conventions=conventions, brian=brian)
    assert replay.exact_spike_train
    assert replay.max_voltage_error_mV < 1e-12


def test_selected_layer6i_trace_summary_preserves_peak_timing_and_threshold_gap() -> None:
    times_ms = brian.asarray([0.0, 1.0, 2.0, 3.0])
    soma = brian.asarray([[-70.0, -60.0, -45.0, -50.0]])
    proximal = brian.asarray([[-70.0, -55.0, -40.0, -48.0]])
    detector = brian.asarray([[0.0, 10.0, 25.0, 20.0]])
    gates = {
        "p023": brian.asarray([[0.0, 0.1, 0.2, 0.1]]),
        "p024": brian.asarray([[0.0, 0.0, 0.3, 0.2]]),
        "p025": brian.asarray([[0.0, 0.4, 0.5, 0.1]]),
    }
    currents = {
        "p023": brian.asarray([[0.0, 1.0, 3.0, 2.0]]),
        "p024": brian.asarray([[0.0, 2.0, 4.0, 1.0]]),
        "p025": brian.asarray([[0.0, 5.0, 7.0, 2.0]]),
    }

    summary = summarize_layer6i_selected_traces(
        selected_indices=(0,),
        times_ms=times_ms,
        mismatch_start_ms=1.0,
        spike_detector_threshold_mV=30.0,
        soma_voltage_mV=soma,
        proximal_voltage_mV=proximal,
        spike_detector_voltage_mV=detector,
        gate_by_projection=gates,
        current_pA_by_projection=currents,
        spike_indices=brian.asarray([], dtype=int),
        spike_times_ms=brian.asarray([]),
    )[0]

    assert summary.index == 0
    assert summary.mismatch_event_times_ms == ()
    assert summary.spike_detector_peak_mV == pytest.approx(25.0)
    assert summary.spike_detector_peak_time_ms == pytest.approx(2.0)
    assert summary.threshold_minus_detector_peak_mV == pytest.approx(5.0)
    assert summary.soma_voltage_peak_mV == pytest.approx(-45.0)
    assert summary.proximal_voltage_peak_mV == pytest.approx(-40.0)
    projection025 = {item.projection_id: item for item in summary.projections}["p025"]
    assert projection025.current_integral_pA_ms == pytest.approx(10.5)
    assert projection025.current_peak_pA == pytest.approx(7.0)
    assert projection025.current_peak_time_ms == pytest.approx(2.0)
    assert projection025.gate_at_current_peak == pytest.approx(0.5)
    assert projection025.soma_voltage_at_current_peak_mV == pytest.approx(-45.0)
    assert projection025.gate_peak == pytest.approx(0.5)
    assert projection025.gate_peak_time_ms == pytest.approx(2.0)
    assert projection025.current_at_soma_peak_pA == pytest.approx(7.0)
    assert projection025.current_at_proximal_peak_pA == pytest.approx(7.0)
    assert projection025.current_at_detector_peak_pA == pytest.approx(7.0)
