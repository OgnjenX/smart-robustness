from __future__ import annotations

import pytest

from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.figure7 import (
    Figure7ConditionResult,
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
