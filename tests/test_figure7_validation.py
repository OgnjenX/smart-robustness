from __future__ import annotations

import numpy as np
import pytest

from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.figure7 import (
    FIGURE7_REQUIRED_LEARNED_PROJECTIONS,
    Figure7ConditionResult,
    apply_figure7_learned_state,
    assess_figure7_arousal,
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


def test_figure7_scorer_rejects_swapped_conditions() -> None:
    with pytest.raises(ValueError, match="match result"):
        assess_figure7_arousal(
            _result(MatchCondition.MISMATCH, 4),
            _result(MatchCondition.MISMATCH, 7),
        )


class _Projection:
    def __init__(self) -> None:
        self.w = np.zeros(3)
        self.w_maximum = np.ones(3)
        self.modifiable = np.ones(3)

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
