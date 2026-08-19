from __future__ import annotations

import numpy as np
import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.classic_sector import (
    build_first_order_connected_sector,
    figure6_runtime_conventions,
)
from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.figure7 import (
    FIGURE7_REQUIRED_LEARNED_PROJECTIONS,
    Figure6ReferenceExpectation,
    Figure7ConditionResult,
    apply_figure7_learned_state,
    assess_figure7_arousal,
    assess_figure7_pathway,
    assess_figure7_reproduction,
    paper_constrained_figure6_expectation,
    run_figure7_condition,
)


def _result(condition: MatchCondition, spike_count: int) -> Figure7ConditionResult:
    return Figure7ConditionResult(
        condition=condition,
        duration_ms=100.0,
        nonspecific_spike_times_ms=tuple(float(index * 10) for index in range(spike_count)),
    )


def test_figure7_arousal_accepts_published_approximate_rates() -> None:
    assessment = assess_figure7_arousal(
        _result(MatchCondition.MATCH, 4),
        _result(MatchCondition.MISMATCH, 7),
    )
    assert assessment.match_rate_hz == pytest.approx(40.0)
    assert assessment.mismatch_rate_hz == pytest.approx(70.0)
    assert assessment.reproduced_arousal


def test_rate_fit_without_mismatch_disinhibition_is_not_reproduction() -> None:
    assessment = assess_figure7_arousal(
        _result(MatchCondition.MATCH, 7),
        _result(MatchCondition.MISMATCH, 4),
        tolerance_hz=30.0,
    )
    assert assessment.match_rate_pass
    assert assessment.mismatch_rate_pass
    assert not assessment.mismatch_disinhibition_pass
    assert not assessment.reproduced_arousal


def test_figure7_pathway_requires_more_relay_cells_and_trn_spikes_during_match() -> None:
    match = Figure7ConditionResult(
        condition=MatchCondition.MATCH,
        duration_ms=100.0,
        nonspecific_spike_times_ms=(10.0, 35.0, 60.0, 85.0),
        relay_spike_indices=(38, 39, 40, 41, 42),
        relay_spike_times_ms=(10.0, 10.0, 10.0, 10.0, 10.0),
        trn_spike_indices=(38, 39, 40, 41),
        trn_spike_times_ms=(12.0, 12.0, 12.0, 12.0),
    )
    mismatch = Figure7ConditionResult(
        condition=MatchCondition.MISMATCH,
        duration_ms=100.0,
        nonspecific_spike_times_ms=(5.0, 20.0, 35.0, 50.0, 65.0, 80.0, 95.0),
        relay_spike_indices=(40,),
        relay_spike_times_ms=(10.0,),
        trn_spike_indices=(40,),
        trn_spike_times_ms=(12.0,),
    )

    pathway = assess_figure7_pathway(match, mismatch)
    combined = assess_figure7_reproduction(match, mismatch)

    assert pathway.match_active_relay_cells == 5
    assert pathway.mismatch_active_relay_cells == 1
    assert pathway.relay_subset_pass
    assert pathway.trn_order_pass
    assert pathway.reproduced_pathway
    assert combined.reproduced


def test_output_rate_fit_without_caption_pathway_is_not_full_reproduction() -> None:
    match = _result(MatchCondition.MATCH, 4)
    mismatch = _result(MatchCondition.MISMATCH, 7)

    assessment = assess_figure7_reproduction(match, mismatch)

    assert assessment.arousal.reproduced_arousal
    assert not assessment.pathway.reproduced_pathway
    assert not assessment.reproduced


def test_figure7_scorer_rejects_swapped_conditions() -> None:
    with pytest.raises(ValueError, match="match result"):
        assess_figure7_arousal(
            _result(MatchCondition.MISMATCH, 4),
            _result(MatchCondition.MISMATCH, 7),
        )


def test_figure7_scorers_reject_different_trial_durations() -> None:
    match = _result(MatchCondition.MATCH, 4)
    mismatch = Figure7ConditionResult(
        condition=MatchCondition.MISMATCH,
        duration_ms=200.0,
        nonspecific_spike_times_ms=(10.0,) * 14,
    )
    with pytest.raises(ValueError, match="same duration"):
        assess_figure7_arousal(match, mismatch)
    with pytest.raises(ValueError, match="same duration"):
        assess_figure7_pathway(match, mismatch)


class _Projection:
    def __init__(self) -> None:
        self.w = np.zeros(3)
        self.w_maximum = np.ones(3)
        self.modifiable = np.ones(3)
        self.i = np.asarray((40, 40, 40))
        self.j = np.asarray((39, 40, 41))

    def __len__(self) -> int:
        return 3


def test_figure7_requires_and_freezes_an_explicit_learned_state() -> None:
    projections = {projection_id: _Projection() for projection_id in FIGURE7_REQUIRED_LEARNED_PROJECTIONS}
    learned = {
        projection_id: np.asarray((0.2, 0.8, 0.2))
        for projection_id in FIGURE7_REQUIRED_LEARNED_PROJECTIONS
    }
    apply_figure7_learned_state(projections, learned)
    for projection in projections.values():
        assert np.asarray(projection.w) == pytest.approx((0.2, 0.8, 0.2))
        assert np.asarray(projection.modifiable) == pytest.approx(0)


def test_figure7_rejects_missing_or_out_of_bounds_learned_weights() -> None:
    projections = {projection_id: _Projection() for projection_id in FIGURE7_REQUIRED_LEARNED_PROJECTIONS}
    with pytest.raises(ValueError, match="missing learned"):
        apply_figure7_learned_state(projections, {})
    invalid = {
        projection_id: np.asarray((0.2, 1.1, 0.2))
        for projection_id in FIGURE7_REQUIRED_LEARNED_PROJECTIONS
    }
    with pytest.raises(ValueError, match="exceed declared maxima"):
        apply_figure7_learned_state(projections, invalid)


def test_paper_constrained_reference_is_horizontal_and_split_across_channels() -> None:
    projections = {projection_id: _Projection() for projection_id in FIGURE7_REQUIRED_LEARNED_PROJECTIONS}
    learned = paper_constrained_figure6_expectation(
        projections,
        Figure6ReferenceExpectation(
            peak_combined_weight=1.0,
            sigma_x_cells=2.0,
            sigma_y_cells=0.5,
        ),
    )
    combined = sum(np.asarray(values) for values in learned.values())
    assert combined == pytest.approx((np.exp(-0.5 / 4), 1.0, np.exp(-0.5 / 4)))


def test_source_derived_reference_matches_runtime_projection_arrays() -> None:
    brian.start_scope()
    brian.prefs.codegen.target = "numpy"
    sector = build_first_order_connected_sector(
        conventions=figure6_runtime_conventions(), brian=brian
    )
    runtime = paper_constrained_figure6_expectation(sector.projections)
    source = paper_constrained_figure6_expectation(
        sector.projections, derive_from_source=True
    )
    for projection_id in FIGURE7_REQUIRED_LEARNED_PROJECTIONS:
        assert np.asarray(source[projection_id]) == pytest.approx(runtime[projection_id])


def test_figure7_runner_requires_exactly_one_learned_state_source() -> None:
    with pytest.raises(ValueError, match="requires exactly one"):
        run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=100.0,
        )


def test_figure7_runner_rejects_invalid_projection_discriminators() -> None:
    with pytest.raises(ValueError, match="unknown projection scale"):
        run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=100.0,
            use_paper_constrained_reference=True,
            duration_ms=0.01,
            projection_weight_scales={"not-a-projection": 2.0},
        )
    with pytest.raises(ValueError, match="top_down_cue_lead_ms"):
        run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=100.0,
            use_paper_constrained_reference=True,
            top_down_cue_lead_ms=-1.0,
        )
    with pytest.raises(ValueError, match="equilibration_ms"):
        run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=100.0,
            use_paper_constrained_reference=True,
            equilibration_ms=-1.0,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=100.0,
            use_paper_constrained_reference=True,
            duration_ms=0.01,
            projection_weight_scales={"modeldb112923.projection.000": 0.0},
        )
    with pytest.raises(ValueError, match="requires exactly one"):
        run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=100.0,
            learned_weights={},
            use_paper_constrained_reference=True,
        )

    with pytest.raises(ValueError, match="requires exactly one"):
        run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=1000,
            use_paper_constrained_reference=True,
            pretrain_with_figure6_episode=True,
        )


def test_figure7_pathway_diagnostics_require_a_post_startup_window() -> None:
    with pytest.raises(ValueError, match="diagnostics require duration_ms > 45"):
        run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=600,
            use_paper_constrained_reference=True,
            duration_ms=45,
            record_relay_diagnostics=True,
        )


def test_exact_relay_clamp_rejects_undefined_full_network_combination() -> None:
    with pytest.raises(ValueError, match="only defined for the first-order"):
        run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=600,
            use_paper_constrained_reference=True,
            exact_relay_voltage_clamp=True,
            include_higher_order_loop=True,
        )


def test_exact_relay_clamp_rejects_cue_lead_that_would_preapply_bottom_up() -> None:
    with pytest.raises(ValueError, match="cue-only lead"):
        run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=600,
            use_paper_constrained_reference=True,
            exact_relay_voltage_clamp=True,
            top_down_cue_lead_ms=10.0,
        )


def test_figure7_first_order_condition_runs_through_cpp_standalone(tmp_path) -> None:
    result = run_figure7_condition(
        condition=MatchCondition.MATCH,
        top_down_current_pA=600,
        use_paper_constrained_reference=True,
        duration_ms=0.01,
        dt_ms=0.01,
        top_down_cue_lead_ms=0.01,
        cpp_standalone_directory=tmp_path,
    )
    assert result.condition is MatchCondition.MATCH
    assert result.network_scope == "first_order"
    assert result.top_down_cue_lead_ms == 0.01
    assert result.equilibration_ms == 0.0


def test_figure7_result_accepts_relay_pathway_diagnostics() -> None:
    result = _result(MatchCondition.MATCH, 4)
    assert result.relay_top_down_ampa_peak_by_index == ()
    assert result.layer4_spike_indices == ()
    assert result.relay_top_down_ampa_integral_ms_by_index == ()
    assert result.relay_top_down_nmda_peak_by_index == ()
    assert result.relay_distal_voltage_range_mV_by_index == ()
    assert result.relay_proximal_voltage_range_mV_by_index == ()
    assert result.relay_soma_voltage_range_mV_by_index == ()
    assert result.relay_trn_gaba_peak_by_index == ()
    assert result.relay_trn_gaba_integral_ms_by_index == ()
    assert result.top_down_cue_lead_ms == 0.0
    assert result.equilibration_ms == 0.0
    assert result.trn_layer6ii_ampa_peak_by_index == ()
    assert result.trn_layer6ii_nmda_peak_by_index == ()
    assert result.trn_relay_ampa_peak_by_index == ()
    assert result.trn_proximal_voltage_range_mV_by_index == ()
    assert result.trn_soma_voltage_range_mV_by_index == ()
    assert result.trn_post_startup_soma_voltage_range_mV_by_index == ()
    assert result.trn_driven_current_range_pA_by_index_and_source == ()
    assert result.nonspecific_trn_gaba_peak is None
    assert result.nonspecific_trn_gaba_integral_ms is None
    assert result.nonspecific_post_startup_trn_gaba_peak is None
    assert result.nonspecific_layer6ii_ampa_peak is None
    assert result.nonspecific_layer6ii_nmda_peak is None
    assert result.nonspecific_direct_input_current_range_pA is None
    assert result.nonspecific_trn_current_range_pA is None
    assert result.nonspecific_layer6ii_current_range_pA is None
    assert result.nonspecific_voltage_range_mV_by_compartment == ()
    assert result.v1_cortical_spike_times_ms == ()
