from __future__ import annotations

import numpy as np
import pytest

from smart_robustness.validation.figure6 import (
    MINIMUM_TOP_DOWN_COMBINED_PEAK,
    MINIMUM_TOP_DOWN_HORIZONTAL_CONTRAST,
    Figure6LearningProtocol,
    Figure6LearningResult,
    Figure6MapSummary,
    assess_figure6_cortical_recruitment,
    assess_figure6_top_down_timing,
    figure6_weight_reachability,
    run_figure6_l23_current_balance,
    run_figure6_relay_current_balance,
    run_figure6_top_down_learning_phase,
)


def test_figure6_reachability_bound_accepts_source_maximum_and_rejects_plot_scale() -> None:
    at_maximum = figure6_weight_reachability(
        initial_weight=0.05,
        maximum_weight=1.5,
        learning_rate_per_ms=0.1,
        episode_ms=100.0,
        observed_weight=1.49,
    )
    assert at_maximum.reachable
    assert at_maximum.upper_bound < 1.5
    plotted_scale = figure6_weight_reachability(
        initial_weight=0.05,
        maximum_weight=1.5,
        learning_rate_per_ms=0.1,
        episode_ms=100.0,
        observed_weight=2.5,
    )
    assert not plotted_scale.reachable


def test_figure6_learning_protocol_is_the_published_horizontal_episode() -> None:
    protocol = Figure6LearningProtocol()
    assert protocol.warmup_ms == 0.0
    assert protocol.stimulus_ms == 100.0
    assert protocol.post_stimulus_ms == 0.0
    assert protocol.source_value == 120.0
    assert protocol.category_source_value == 70.0
    assert protocol.winning_layer4_index == 40
    assert protocol.active_category_index == 40
    assert protocol.layer6ii_ahp_scale == 1.0
    assert protocol.monitored_populations == (
        "thalamic_relay",
        "layer4_excitatory_v1",
        "layer6ii_excitatory_v1",
    )


def test_l23_current_balance_rejects_an_out_of_sheet_target_before_running() -> None:
    with pytest.raises(ValueError, match="target_index must address"):
        run_figure6_l23_current_balance(target_index=81)


def test_relay_current_balance_rejects_an_out_of_sheet_target_before_running() -> None:
    with pytest.raises(ValueError, match="target_index must address"):
        run_figure6_relay_current_balance(target_index=-1)


def test_intrinsic_relay_control_rejects_projection_removals_before_building() -> None:
    with pytest.raises(ValueError, match="projection controls require"):
        run_figure6_relay_current_balance(
            connected=False,
            disabled_projection_ids=("modeldb112923.projection.000",),
        )


def test_learning_phase_rejects_duplicate_targets_before_building() -> None:
    with pytest.raises(ValueError, match="nonempty and unique"):
        run_figure6_top_down_learning_phase(target_indices=(39, 39))


def test_map_retention_advantage_detects_horizontal_orientation() -> None:
    before = np.ones(81)
    after = np.full(81, 0.5)
    after[[38, 39, 40, 41, 42]] = 0.9
    summary = Figure6MapSummary("projection", "map", tuple(before), tuple(after))
    assert summary.horizontal_retention_advantage > 0
    assert summary.horizontal_orientation_contrast > 0
    assert summary.horizontal_mean > summary.vertical_mean


def test_absolute_map_contrast_is_stable_for_tiny_gaussian_tails() -> None:
    before = np.full(81, 1e-12)
    after = np.full(81, 1e-12)
    after[[38, 39, 41, 42]] = 2e-4
    after[[22, 31, 49, 58]] = 1e-4
    summary = Figure6MapSummary("projection", "map", tuple(before), tuple(after))

    assert np.isclose(summary.horizontal_orientation_contrast, 1e-4)


def test_tiny_positive_top_down_contrast_does_not_count_as_reproduction() -> None:
    before = np.ones(81)
    after = np.ones(81)
    after[[38, 39, 41, 42]] += MINIMUM_TOP_DOWN_HORIZONTAL_CONTRAST / 2
    weak = Figure6MapSummary("topdown", "map", tuple(before), tuple(after))
    bottom = Figure6MapSummary(
        "bottom", "map", tuple(before), tuple(after + np.eye(1, 81, 38).ravel())
    )
    result = Figure6LearningResult("fingerprint", 100.0, {}, bottom, weak, weak)

    assert weak.horizontal_orientation_contrast > 0
    assert not result.top_down_oriented
    assert result.relay_detector_voltage_range_mV_by_index == ()
    assert result.relay_detector_threshold_upcrossings_by_index == ()
    assert result.relay_detector_zero_downcrossings_by_index == ()
    assert result.relay_detector_arm_transitions_by_index == ()
    assert result.relay_detector_release_transitions_by_index == ()
    assert result.relay_detector_final_armed_by_index == ()


def test_figure6c_scores_the_combined_wide_and_narrow_field() -> None:
    before = np.zeros(81)
    wide_after = np.full(81, MINIMUM_TOP_DOWN_COMBINED_PEAK / 2)
    narrow_after = np.full(81, MINIMUM_TOP_DOWN_COMBINED_PEAK / 2)
    wide_after[[38, 39, 41, 42]] += 0.006
    narrow_after[[38, 39, 41, 42]] += 0.006
    wide = Figure6MapSummary("wide", "map", tuple(before), tuple(wide_after))
    narrow = Figure6MapSummary("narrow", "map", tuple(before), tuple(narrow_after))
    result = Figure6LearningResult("fingerprint", 100.0, {}, wide, wide, narrow)

    assert wide.horizontal_orientation_contrast < MINIMUM_TOP_DOWN_HORIZONTAL_CONTRAST
    assert narrow.horizontal_orientation_contrast < MINIMUM_TOP_DOWN_HORIZONTAL_CONTRAST
    assert result.top_down_combined.horizontal_orientation_contrast == pytest.approx(0.012)
    assert result.top_down_oriented


def test_figure6c_rejects_correct_shape_at_subpublished_amplitude() -> None:
    before = np.zeros(81)
    after = np.zeros(81)
    after[[38, 39, 41, 42]] = 0.1
    shaped = Figure6MapSummary("topdown", "map", tuple(before), tuple(after))
    result = Figure6LearningResult("fingerprint", 100.0, {}, shaped, shaped, shaped)
    assert result.top_down_combined.horizontal_orientation_contrast > 0.01
    assert max(result.top_down_combined.after) < MINIMUM_TOP_DOWN_COMBINED_PEAK
    assert result.top_down_shape_oriented
    assert not result.top_down_legacy_amplitude_gate_pass
    assert not result.top_down_oriented


def test_figure6c_requires_the_published_peak_not_colorbar_minimum() -> None:
    before = np.zeros(81)
    after = np.zeros(81)
    after[[38, 39, 41, 42]] = 0.6
    shaped = Figure6MapSummary("topdown", "map", tuple(before), tuple(after))
    result = Figure6LearningResult("fingerprint", 100.0, {}, shaped, shaped, shaped)
    assert max(result.top_down_combined.after) == pytest.approx(1.2)
    assert not result.top_down_oriented


def test_top_down_timing_accounts_for_the_archived_axonal_delay() -> None:
    empty = Figure6MapSummary("projection", "map", (0.0,) * 81, (0.0,) * 81)
    result = Figure6LearningResult(
        "fingerprint",
        100.0,
        {},
        empty,
        empty,
        empty,
        population_spike_indices={
            "layer6ii_excitatory_v1": (40,),
            "thalamic_relay": (40, 40),
        },
        population_spike_times_ms={
            "layer6ii_excitatory_v1": (58.28,),
            "thalamic_relay": (59.81, 80.0),
        },
    )
    assessment = assess_figure6_top_down_timing(result)
    assert assessment.teaching_arrival_ms == pytest.approx(60.28)
    assert assessment.preceding_post_minus_arrival_ms == pytest.approx(-0.47)
    assert assessment.following_post_minus_arrival_ms == pytest.approx(19.72)
    assert assessment.causal_pair_in_learning_window


def test_top_down_timing_selects_the_closest_causal_pair_in_the_episode() -> None:
    empty = Figure6MapSummary("projection", "map", (0.0,) * 81, (0.0,) * 81)
    result = Figure6LearningResult(
        "fingerprint",
        100.0,
        {},
        empty,
        empty,
        empty,
        population_spike_indices={
            "layer6ii_excitatory_v1": (40, 40, 40),
            "thalamic_relay": (40, 40),
        },
        population_spike_times_ms={
            "layer6ii_excitatory_v1": (8.1, 46.13, 69.01),
            "thalamic_relay": (5.28, 77.57),
        },
    )
    assessment = assess_figure6_top_down_timing(result)
    assert assessment.category_spike_ms == pytest.approx(69.01)
    assert assessment.teaching_arrival_ms == pytest.approx(71.01)
    assert assessment.following_relay_spike_ms == pytest.approx(77.57)
    assert assessment.following_post_minus_arrival_ms == pytest.approx(6.56)
    assert assessment.causal_pair_in_learning_window


def test_cortical_recruitment_requires_ordered_layer4_layer23_layer5_events() -> None:
    empty = Figure6MapSummary("projection", "map", (0.0,) * 81, (0.0,) * 81)
    complete = Figure6LearningResult(
        "fingerprint",
        100.0,
        {},
        empty,
        empty,
        empty,
        population_spike_times_ms={
            "layer4_excitatory_v1": (10.0,),
            "layer23_excitatory_v1": (12.0,),
            "layer5_excitatory_v1": (14.0,),
            "layer6i_excitatory_v1": (9.0,),
            "layer6ii_excitatory_v1": (8.0,),
        },
    )
    assessment = assess_figure6_cortical_recruitment(complete)
    assert assessment.feedforward_chain_complete

    missing_layer23 = Figure6LearningResult(
        "fingerprint",
        100.0,
        {},
        empty,
        empty,
        empty,
        population_spike_times_ms={
            "layer4_excitatory_v1": (10.0,),
            "layer5_excitatory_v1": (14.0,),
            "layer6ii_excitatory_v1": (8.0,),
        },
    )
    assert not assess_figure6_cortical_recruitment(
        missing_layer23
    ).feedforward_chain_complete
