from __future__ import annotations

from dataclasses import replace

import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.classic_sector import figure6_runtime_conventions
from smart_robustness.validation.figure10 import (
    Figure10ConditionResult,
    assess_figure10_reset,
    compact_layer4_target_balance_summary,
    run_figure10_condition,
    summarize_layer4_balance_bins,
    summarize_layer4_inhibitory_source_traces,
    summarize_layer4_target_balance_bins,
    summarize_layer4_target_timing,
    summarize_layer6i_selected_traces,
    summarize_projection026_target_arrivals,
    summarize_projection036_target_arrivals,
)
from smart_robustness.validation.figure10_search_cycle import (
    VERTICAL_ALTERNATIVES,
    VERTICAL_TARGETS,
    run_figure10_search_cycle_condition,
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
    with pytest.raises(ValueError, match="require reset-chain diagnostics"):
        run_figure10_condition(
            top_down_current_pA=600,
            pre_match_duration_ms=100,
            mismatch_duration_ms=100,
            reset_pathway_enabled=True,
            record_layer4_balance_diagnostics=True,
        )
    with pytest.raises(ValueError, match="requires layer-4 balance diagnostics"):
        run_figure10_condition(
            top_down_current_pA=600,
            pre_match_duration_ms=100,
            mismatch_duration_ms=100,
            reset_pathway_enabled=True,
            record_layer4_target_balance_indices=(40,),
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
    with pytest.raises(ValueError, match="projection delays"):
        run_figure10_condition(
            top_down_current_pA=600,
            pre_match_duration_ms=100,
            mismatch_duration_ms=100,
            reset_pathway_enabled=True,
            persistent_projection_delays_ms={"modeldb112923.projection.036": 0.0},
        )


def test_search_cycle_runner_requires_internal_release_marker() -> None:
    assert VERTICAL_TARGETS == (22, 31, 40, 49, 58)
    assert VERTICAL_ALTERNATIVES == (22, 31, 49, 58)
    with pytest.raises(ValueError, match="strictly inside mismatch"):
        run_figure10_search_cycle_condition(
            top_down_current_pA=800,
            pre_match_duration_ms=100,
            mismatch_duration_ms=200,
            release_after_mismatch_ms=0,
            reset_pathway_enabled=True,
            learned_weights={},
            persistent_projection_weight_scales={},
            brian=brian,
        )
    with pytest.raises(ValueError, match="first-event cue clearing"):
        run_figure10_search_cycle_condition(
            top_down_current_pA=800,
            pre_match_duration_ms=100,
            mismatch_duration_ms=200,
            release_after_mismatch_ms=61.88,
            reset_pathway_enabled=True,
            learned_weights={},
            persistent_projection_weight_scales={},
            top_down_current_mode="sustained_epoch",
            brian=brian,
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
    assert result.layer4_inhibitory_source_traces == ()
    assert result.layer4e_mismatch_projection038_gate_integral_ms is None
    assert result.layer4_balance_bins == ()
    assert result.layer4_target_balance_bins == ()
    assert result.layer4_target_timing_summaries == ()


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


def test_figure10_condition_records_layer4_balance_diagnostics() -> None:
    result = run_figure10_condition(
        top_down_current_pA=600,
        pre_match_duration_ms=0.01,
        mismatch_duration_ms=0.02,
        reset_pathway_enabled=True,
        dt_ms=0.01,
        record_reset_chain_diagnostics=True,
        record_layer4_balance_diagnostics=True,
        record_layer4_target_balance_indices=(40,),
        record_layer4_target_timing_indices=(31,),
        layer4_target_timing_window_ms=(0.0, 0.02),
        layer4_target_timing_bin_width_ms=0.01,
        persistent_projection_delays_ms={"modeldb112923.projection.036": 0.2},
        brian=brian,
    )

    assert result.layer4i_mismatch_projection026_gate_integral_ms is not None
    assert result.layer4e_mismatch_projection036_gate_integral_ms is not None
    assert result.layer4e_mismatch_projection038_gate_integral_ms is not None
    assert result.layer4e_mismatch_projection038_current_integral_pA_ms is not None
    assert len(result.layer4_balance_bins) == 1
    assert result.layer4_balance_bins[0].start_from_mismatch_ms == 0.0
    assert result.layer4_balance_bins[0].end_from_mismatch_ms == pytest.approx(0.02)
    assert len(result.layer4_target_balance_bins) == 1
    assert result.layer4_target_balance_bins[0].index == 40
    assert len(result.layer4_target_timing_summaries) == 1
    assert result.layer4_target_timing_summaries[0].index == 31
    assert len(result.layer4_target_timing_summaries[0].soma_voltage_max_mV) == 2
    assert result.projection_delay_overrides_ms == (
        ("modeldb112923.projection.036", 0.2),
    )


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


def test_layer4_balance_summary_preserves_causal_bins() -> None:
    summaries = summarize_layer4_balance_bins(
        mismatch_start_ms=100.0,
        mismatch_duration_ms=20.0,
        dt_ms=5.0,
        bin_width_ms=10.0,
        state_times_ms=brian.asarray([100.0, 105.0, 110.0, 115.0]),
        projection026_current_pA=brian.asarray([1.0, 2.0, 3.0, 4.0]),
        projection036_current_pA=brian.asarray([-1.0, -2.0, -3.0, -4.0]),
        projection038_current_pA=brian.asarray([5.0, 6.0, 7.0, 8.0]),
        layer4_inhibitory_spike_times_ms=brian.asarray([101.0, 109.0, 111.0]),
        layer4_spike_indices=brian.asarray([40, 10, 40, 11]),
        layer4_spike_times_ms=brian.asarray([102.0, 108.0, 112.0, 118.0]),
        winner_indices=(40,),
    )

    assert len(summaries) == 2
    assert summaries[0].projection026_excitation_integral_pA_ms == pytest.approx(15.0)
    assert summaries[0].projection036_inhibition_integral_pA_ms == pytest.approx(-15.0)
    assert summaries[0].projection038_excitation_integral_pA_ms == pytest.approx(55.0)
    assert summaries[0].layer4_inhibitory_events == 2
    assert summaries[0].layer4_winner_events == 1
    assert summaries[0].layer4_alternative_active_count == 1
    assert summaries[1].projection026_excitation_integral_pA_ms == pytest.approx(35.0)
    assert summaries[1].layer4_inhibitory_events == 1


def test_layer4_target_balance_summary_preserves_cells_and_bins() -> None:
    summaries = summarize_layer4_target_balance_bins(
        target_indices=(1, 3),
        mismatch_start_ms=100.0,
        mismatch_duration_ms=20.0,
        dt_ms=5.0,
        bin_width_ms=10.0,
        state_times_ms=brian.asarray([100.0, 105.0, 110.0, 115.0]),
        projection035_current_pA_by_index=brian.asarray(
            [[1.0, 1.0, 1.0, 1.0], [2.0, 4.0, 6.0, 8.0],
             [1.0, 1.0, 1.0, 1.0], [3.0, 5.0, 7.0, 9.0]]
        ),
        projection036_current_pA_by_index=brian.asarray(
            [[-1.0, -1.0, -1.0, -1.0], [-2.0, -4.0, -6.0, -8.0],
             [-1.0, -1.0, -1.0, -1.0], [-3.0, -5.0, -7.0, -9.0]]
        ),
        projection037_current_pA_by_index=brian.asarray(
            [[1.0, 1.0, 1.0, 1.0], [2.0, 4.0, 6.0, 8.0],
             [1.0, 1.0, 1.0, 1.0], [3.0, 5.0, 7.0, 9.0]]
        ),
        projection038_current_pA_by_index=brian.asarray(
            [[1.0, 1.0, 1.0, 1.0], [2.0, 4.0, 6.0, 8.0],
             [1.0, 1.0, 1.0, 1.0], [3.0, 5.0, 7.0, 9.0]]
        ),
        layer4_spike_indices=brian.asarray([1, 3, 1]),
        layer4_spike_times_ms=brian.asarray([101.0, 106.0, 111.0]),
    )

    assert [(item.start_from_mismatch_ms, item.index) for item in summaries] == [
        (0.0, 1),
        (0.0, 3),
        (10.0, 1),
        (10.0, 3),
    ]
    assert summaries[0].projection036_inhibition_integral_pA_ms == pytest.approx(-30.0)
    assert summaries[0].projection035_relay_excitation_integral_pA_ms == pytest.approx(30.0)
    assert summaries[1].projection037_recurrent_excitation_integral_pA_ms == pytest.approx(
        40.0
    )
    assert summaries[1].projection038_excitation_integral_pA_ms == pytest.approx(40.0)
    assert summaries[0].layer4_events == 1
    assert summaries[1].layer4_events == 1
    assert summaries[2].layer4_events == 1
    assert summaries[3].layer4_events == 0


def test_compact_layer4_target_balance_summary_preserves_exact_arrays() -> None:
    rows = [
        {
            "index": index,
            "start_from_mismatch_ms": start,
            "end_from_mismatch_ms": start + 10.0,
            "projection035_relay_excitation_integral_pA_ms": index + start,
            "projection036_inhibition_integral_pA_ms": -(index + start),
            "projection037_recurrent_excitation_integral_pA_ms": index + 2 * start,
            "projection038_excitation_integral_pA_ms": index + 3 * start,
            "layer4_events": int(start / 10),
        }
        for start in (0.0, 10.0)
        for index in (31, 40)
    ]

    compact = compact_layer4_target_balance_summary(
        {"layer4_target_balance_bins": rows}
    )

    assert compact["layer4_target_balance_bin_edges_ms"] == [0.0, 10.0, 20.0]
    assert compact["layer4_target_balance_series_columns"][-1] == "layer4_events"
    assert compact["layer4_target_balance_series"]["31"] == [
        [31.0, 41.0],
        [-31.0, -41.0],
        [31.0, 51.0],
        [31.0, 61.0],
        [0, 1],
    ]


def test_layer4_target_timing_preserves_current_onset_before_exact_spike() -> None:
    summaries = summarize_layer4_target_timing(
        target_indices=(1,),
        mismatch_start_ms=100.0,
        window_start_from_mismatch_ms=1.0,
        window_end_from_mismatch_ms=5.0,
        dt_ms=1.0,
        bin_width_ms=1.0,
        current_activity_threshold_pA=1e-9,
        gate_activity_threshold=2.0,
        state_times_ms=brian.asarray([100.0, 101.0, 102.0, 103.0, 104.0, 105.0]),
        soma_voltage_mV_by_index=brian.asarray(
            [[-70.0] * 6, [-70.0, -68.0, -60.0, -50.0, -45.0, -55.0]]
        ),
        projection035_current_pA_by_index=brian.asarray([[0.0] * 6, [0, 1, 2, 3, 4, 5]]),
        projection036_current_pA_by_index=brian.asarray([[0.0] * 6, [0, -1, -2, -3, -4, -5]]),
        projection037_current_pA_by_index=brian.asarray([[0.0] * 6, [0, 0, 0.5, 1, 2, 3]]),
        projection038_current_pA_by_index=brian.asarray([[0.0] * 6, [0, 4, 3, 2, 1, 0]]),
        projection035_gate_by_index=brian.asarray([[0.0] * 6, [0, 1, 2, 3, 4, 5]]),
        projection036_gate_by_index=brian.asarray([[0.0] * 6, [0, 2, 4, 6, 8, 10]]),
        projection038_gate_by_index=brian.asarray([[0.0] * 6, [0, 0, 0, 3, 6, 9]]),
        layer4_spike_indices=brian.asarray([1]),
        layer4_spike_times_ms=brian.asarray([103.5]),
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.bin_edges_from_mismatch_ms == (1.0, 2.0, 3.0, 4.0, 5.0)
    assert summary.projection035_relay_excitation_integral_pA_ms == (1.0, 2.0, 3.0, 4.0)
    assert summary.projection037_first_active_time_from_mismatch_ms == 2.0
    assert summary.layer4_spike_times_from_mismatch_ms == (3.5,)
    assert summary.projection035_relay_gate_mean == (1.0, 2.0, 3.0, 4.0)
    assert summary.projection036_inhibition_gate_mean == (2.0, 4.0, 6.0, 8.0)
    assert summary.projection038_excitation_gate_mean == (0.0, 0.0, 3.0, 6.0)
    assert summary.projection035_first_gate_threshold_time_from_mismatch_ms == 2.0
    assert summary.projection036_first_gate_threshold_time_from_mismatch_ms == 1.0
    assert summary.projection038_first_gate_threshold_time_from_mismatch_ms == 3.0
    assert summary.soma_voltage_max_mV == (-68.0, -60.0, -50.0, -45.0)


def test_projection036_target_arrivals_preserve_source_edge_and_delay() -> None:
    summaries = summarize_projection036_target_arrivals(
        target_indices=(31,),
        mismatch_start_ms=100.0,
        window_start_from_mismatch_ms=75.0,
        window_end_from_mismatch_ms=76.0,
        edge_source_indices=brian.asarray([20, 21, 22, 20]),
        edge_target_indices=brian.asarray([31, 31, 31, 40]),
        edge_weights=brian.asarray([0.2, 0.5, 0.1, 0.9]),
        edge_delays_ms=brian.asarray([0.1, 0.1, 0.2, 0.1]),
        source_spike_indices=brian.asarray([20, 21, 22, 20]),
        source_spike_times_ms=brian.asarray([174.95, 175.2, 175.7, 176.0]),
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.connected_source_indices == (20, 21, 22)
    assert summary.connected_edge_weights == (0.2, 0.5, 0.1)
    assert [arrival.source_index for arrival in summary.arrivals] == [20, 21, 22]
    assert [arrival.arrival_time_from_mismatch_ms for arrival in summary.arrivals] == (
        pytest.approx([75.05, 75.3, 75.9])
    )
    assert [arrival.edge_weight for arrival in summary.arrivals] == [0.2, 0.5, 0.1]


def test_projection026_target_arrivals_preserve_source_edge_and_delay() -> None:
    summaries = summarize_projection026_target_arrivals(
        target_indices=(38, 40, 42),
        mismatch_start_ms=100.0,
        window_start_from_mismatch_ms=23.3,
        window_end_from_mismatch_ms=23.6,
        edge_source_indices=brian.asarray([10, 11, 12, 10]),
        edge_target_indices=brian.asarray([38, 40, 42, 40]),
        edge_weights=brian.asarray([0.2, 0.5, 0.1, 0.9]),
        edge_delays_ms=brian.asarray([1.0, 1.0, 1.0, 1.0]),
        source_spike_indices=brian.asarray([10, 11, 12, 10]),
        source_spike_times_ms=brian.asarray([122.35, 122.45, 122.55, 123.0]),
    )

    assert [summary.target_index for summary in summaries] == [38, 40, 42]
    assert summaries[0].connected_source_indices == (10,)
    assert summaries[1].connected_source_indices == (10, 11)
    assert summaries[1].connected_edge_weights == (0.9, 0.5)
    assert [arrival.source_index for arrival in summaries[1].arrivals] == [10, 11]
    assert [
        arrival.arrival_time_from_mismatch_ms
        for arrival in summaries[1].arrivals
    ] == pytest.approx(
        [23.35, 23.45]
    )
    assert [arrival.source_index for arrival in summaries[2].arrivals] == [12]
    assert summaries[2].arrivals[0].arrival_time_from_mismatch_ms == pytest.approx(
        23.55
    )


def test_layer4_inhibitory_source_trace_preserves_native_inputs_and_state() -> None:
    base = brian.asarray(
        [[float(index + step) for step in range(5)] for index in (38, 40, 42)]
    )
    summaries = summarize_layer4_inhibitory_source_traces(
        source_indices=(38, 40, 42),
        mismatch_start_ms=100.0,
        window_start_from_mismatch_ms=75.0,
        window_end_from_mismatch_ms=75.02,
        state_times_ms=brian.asarray([174.99, 175.0, 175.01, 175.02, 175.03]),
        soma_voltage_mV_by_index=brian.asarray(
            [[float(index + step) for step in range(5)] for index in (38, 40, 42)]
        ),
        proximal_voltage_mV_by_index=base[:, :],
        spike_detector_voltage_mV_by_index=base[:, :],
        sodium_activation_by_index=base[:, :],
        sodium_inactivation_by_index=base[:, :],
        potassium_activation_by_index=base[:, :],
        projection026_gate_by_index=base[:, :],
        projection026_current_pA_by_index=base[:, :],
        projection027_gate_by_index=base[:, :],
        projection027_current_pA_by_index=base[:, :],
        projection028_gate_by_index=base[:, :],
        projection028_current_pA_by_index=base[:, :],
        projection029_gap_current_pA_by_index=base[:, :],
        projection030_gate_by_index=base[:, :],
        projection030_current_pA_by_index=base[:, :],
        spike_indices=brian.asarray([38, 40, 42, 40]),
        spike_times_ms=brian.asarray([175.0, 175.01, 175.02, 175.03]),
    )

    assert [item.source_index for item in summaries] == [38, 40, 42]
    assert summaries[1].times_from_mismatch_ms == pytest.approx(
        (75.0, 75.01, 75.02)
    )
    assert summaries[1].soma_voltage_mV == (41.0, 42.0, 43.0)
    assert summaries[1].projection026_gate == (41.0, 42.0, 43.0)
    assert summaries[1].projection029_gap_current_pA == (41.0, 42.0, 43.0)
    assert summaries[1].spike_times_from_mismatch_ms == pytest.approx((75.01,))


def test_figure10_condition_smoke_records_layer4_inhibitory_source_trace(
    tmp_path,
) -> None:
    trace_path = tmp_path / "layer4i_trace.npz"
    result = run_figure10_condition(
        top_down_current_pA=600,
        pre_match_duration_ms=0.01,
        mismatch_duration_ms=0.02,
        reset_pathway_enabled=True,
        dt_ms=0.01,
        record_reset_chain_diagnostics=True,
        record_layer4_inhibitory_trace_indices=(40,),
        layer4_inhibitory_trace_window_ms=(0.0, 0.01),
        layer4_inhibitory_trace_output=trace_path,
        brian=brian,
    )

    assert len(result.layer4_inhibitory_source_traces) == 1
    trace = result.layer4_inhibitory_source_traces[0]
    assert trace.source_index == 40
    assert trace.times_from_mismatch_ms == pytest.approx((0.0, 0.01))
    assert len(trace.soma_voltage_mV) == 2
    assert len(trace.projection030_current_pA) == 2
    assert result.layer4_inhibitory_source_trace_path == str(trace_path)
    assert len(result.layer4_inhibitory_source_trace_sha256) == 64
    assert trace_path.is_file()
