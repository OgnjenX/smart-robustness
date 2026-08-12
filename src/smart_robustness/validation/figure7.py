"""Predeclared Figure 7 match/mismatch arousal metrics.

Figure 7 reports approximate firing rates for the single nonspecific thalamic
cell.  It does not provide a complete trace or a numeric reset latency, so the
rate result and the later qualitative reset result are deliberately separate.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..protocols import MatchCondition

FIGURE7_MATCH_RATE_HZ = 40.0
FIGURE7_MISMATCH_RATE_HZ = 70.0
FIGURE7_RATE_TOLERANCE_HZ = 10.0


@dataclass(frozen=True, slots=True)
class Figure7ConditionResult:
    condition: MatchCondition
    duration_ms: float
    nonspecific_spike_times_ms: tuple[float, ...]
    layer4_spike_times_ms: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be positive")

    @property
    def nonspecific_rate_hz(self) -> float:
        return len(self.nonspecific_spike_times_ms) * 1000.0 / self.duration_ms


@dataclass(frozen=True, slots=True)
class Figure7ArousalAssessment:
    match_rate_hz: float
    mismatch_rate_hz: float
    match_target_hz: float = FIGURE7_MATCH_RATE_HZ
    mismatch_target_hz: float = FIGURE7_MISMATCH_RATE_HZ
    tolerance_hz: float = FIGURE7_RATE_TOLERANCE_HZ

    @property
    def match_rate_pass(self) -> bool:
        return abs(self.match_rate_hz - self.match_target_hz) <= self.tolerance_hz

    @property
    def mismatch_rate_pass(self) -> bool:
        return abs(self.mismatch_rate_hz - self.mismatch_target_hz) <= self.tolerance_hz

    @property
    def mismatch_disinhibition_pass(self) -> bool:
        return self.mismatch_rate_hz > self.match_rate_hz

    @property
    def reproduced_arousal(self) -> bool:
        return (
            self.match_rate_pass
            and self.mismatch_rate_pass
            and self.mismatch_disinhibition_pass
        )


def assess_figure7_arousal(
    match: Figure7ConditionResult,
    mismatch: Figure7ConditionResult,
    *,
    tolerance_hz: float = FIGURE7_RATE_TOLERANCE_HZ,
) -> Figure7ArousalAssessment:
    """Score only the published Figure 7 nonspecific-thalamus rate claim."""

    if match.condition is not MatchCondition.MATCH:
        raise ValueError("match result must use the match condition")
    if mismatch.condition is not MatchCondition.MISMATCH:
        raise ValueError("mismatch result must use the mismatch condition")
    if tolerance_hz < 0:
        raise ValueError("tolerance_hz cannot be negative")
    return Figure7ArousalAssessment(
        match_rate_hz=match.nonspecific_rate_hz,
        mismatch_rate_hz=mismatch.nonspecific_rate_hz,
        tolerance_hz=tolerance_hz,
    )
