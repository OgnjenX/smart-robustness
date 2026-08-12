"""Predeclared Figure 7 match/mismatch arousal metrics.

Figure 7 reports approximate firing rates for the single nonspecific thalamic
cell.  It does not provide a complete trace or a numeric reset latency, so the
rate result and the later qualitative reset result are deliberately separate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..protocols import MatchCondition
from .figure6 import TOP_DOWN_NARROW_PROJECTION_ID, TOP_DOWN_WIDE_PROJECTION_ID

FIGURE7_MATCH_RATE_HZ = 40.0
FIGURE7_MISMATCH_RATE_HZ = 70.0
FIGURE7_RATE_TOLERANCE_HZ = 10.0
FIGURE7_REQUIRED_LEARNED_PROJECTIONS = (
    TOP_DOWN_WIDE_PROJECTION_ID,
    TOP_DOWN_NARROW_PROJECTION_ID,
)


def apply_figure7_learned_state(
    projections: Mapping[str, object],
    learned_weights: Mapping[str, tuple[float, ...] | np.ndarray],
    *,
    freeze_learning: bool = True,
) -> None:
    """Install an explicit Figure 6-trained expectation for Figure 7.

    Requiring a weight snapshot prevents the untrained circular Gaussian from
    being mislabeled as the learned horizontal expectation shown in Figure 7.
    """

    missing = set(FIGURE7_REQUIRED_LEARNED_PROJECTIONS) - set(learned_weights)
    if missing:
        raise ValueError(f"missing learned Figure 7 projections: {sorted(missing)}")
    for projection_id in FIGURE7_REQUIRED_LEARNED_PROJECTIONS:
        projection = projections[projection_id]
        values = np.asarray(learned_weights[projection_id], dtype=float)
        if values.shape != (len(projection),):
            raise ValueError(
                f"{projection_id}: expected {len(projection)} weights, got {values.shape}"
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"{projection_id}: learned weights must be finite and nonnegative")
        maximum = np.asarray(projection.w_maximum[:], dtype=float)
        if np.any(values > maximum + 1e-12):
            raise ValueError(f"{projection_id}: learned weights exceed declared maxima")
        projection.w = values
        if freeze_learning:
            projection.modifiable = 0


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
