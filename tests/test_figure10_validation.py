from __future__ import annotations

import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.validation.figure10 import (
    Figure10ConditionResult,
    assess_figure10_reset,
    run_figure10_condition,
)


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
