"""Reproducible Figure 6a SMART spike-timing learning curves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..learning import (
    LEARNING_RULES,
    equilibrium_depression_scale,
    full_postsynaptic_learning_signal,
    gated_weight_derivative,
)
from ..models.currents import biexponential_normalization

BOTTOM_UP_PROJECTION_ID = "modeldb112923.projection.035"
TOP_DOWN_WIDE_PROJECTION_ID = "modeldb112923.projection.005"
TOP_DOWN_NARROW_PROJECTION_ID = "modeldb112923.projection.007"
HORIZONTAL_INDICES = (38, 39, 40, 41, 42)
VERTICAL_INDICES = (22, 31, 40, 49, 58)
HORIZONTAL_ONLY_INDICES = (38, 39, 41, 42)
VERTICAL_ONLY_INDICES = (22, 31, 49, 58)


@dataclass(frozen=True, slots=True)
class Figure6TimingProtocol:
    relative_times_ms: tuple[float, ...] = tuple(float(value) for value in range(-30, 31))
    rise_ms: float = 1.0
    fall_ms: float = 7.0
    depotentiation_ms: float = 25.0
    initial_weight: float = 0.3
    baseline_weight: float = 0.3
    maximum_weight: float = 6.0
    learning_rate_per_ms: float = 0.1
    dt_ms: float = 0.01
    spike_above_threshold_ms: float = 1.0


@dataclass(frozen=True, slots=True)
class Figure6TimingResult:
    protocol: Figure6TimingProtocol
    curves: dict[str, tuple[float, ...]]

    def peak_time_ms(self, rule: str) -> float:
        values = np.asarray(self.curves[rule])
        return self.protocol.relative_times_ms[int(np.argmax(values))]

    def trough_time_ms(self, rule: str) -> float:
        values = np.asarray(self.curves[rule])
        return self.protocol.relative_times_ms[int(np.argmin(values))]


@dataclass(frozen=True, slots=True)
class Figure6MapSummary:
    projection_id: str
    map_role: str
    before: tuple[float, ...]
    after: tuple[float, ...]

    @property
    def delta(self) -> tuple[float, ...]:
        return tuple(after - before for before, after in zip(self.before, self.after, strict=True))

    @property
    def horizontal_mean(self) -> float:
        return float(np.mean(np.asarray(self.after)[list(HORIZONTAL_INDICES)]))

    @property
    def vertical_mean(self) -> float:
        return float(np.mean(np.asarray(self.after)[list(VERTICAL_INDICES)]))

    @property
    def horizontal_retention_advantage(self) -> float:
        before = np.asarray(self.before)
        after = np.asarray(self.after)
        retention = np.divide(after, before, out=np.ones_like(after), where=before != 0)
        # The center belongs to both bars and carries no orientation
        # information, despite being the largest coefficient of the Gaussian.
        horizontal = float(np.mean(retention[list(HORIZONTAL_ONLY_INDICES)]))
        vertical = float(np.mean(retention[list(VERTICAL_ONLY_INDICES)]))
        return horizontal - vertical


@dataclass(frozen=True, slots=True)
class Figure6LearningResult:
    convention_fingerprint: str
    duration_ms: float
    population_spikes: dict[str, int]
    bottom_up: Figure6MapSummary
    top_down_wide: Figure6MapSummary
    top_down_narrow: Figure6MapSummary

    @property
    def bottom_up_oriented(self) -> bool:
        return self.bottom_up.horizontal_retention_advantage > 0

    @property
    def top_down_oriented(self) -> bool:
        return (
            self.top_down_wide.horizontal_retention_advantage > 0
            and self.top_down_narrow.horizontal_retention_advantage > 0
        )


def _incoming_map(projection: Any, weights: np.ndarray, *, target_index: int) -> np.ndarray:
    source = np.asarray(projection.i[:], dtype=int)
    target = np.asarray(projection.j[:], dtype=int)
    selected = target == target_index
    result = np.zeros(81, dtype=float)
    result[source[selected]] = weights[selected]
    return result


def _outgoing_map(projection: Any, weights: np.ndarray, *, source_index: int) -> np.ndarray:
    source = np.asarray(projection.i[:], dtype=int)
    target = np.asarray(projection.j[:], dtype=int)
    selected = source == source_index
    result = np.zeros(81, dtype=float)
    result[target[selected]] = weights[selected]
    return result


def summarize_figure6_learning(
    *,
    convention_fingerprint: str,
    duration_ms: float,
    population_spikes: dict[str, int],
    projections: dict[str, Any],
    before_weights: dict[str, np.ndarray],
    winning_layer4_index: int = 40,
    active_category_index: int = 40,
) -> Figure6LearningResult:
    """Extract the Figure 6b incoming and Figure 6c outgoing 9x9 maps."""

    summaries: dict[str, Figure6MapSummary] = {}
    for projection_id, role in (
        (BOTTOM_UP_PROJECTION_ID, "incoming_to_winning_layer4"),
        (TOP_DOWN_WIDE_PROJECTION_ID, "outgoing_from_active_layer6ii"),
        (TOP_DOWN_NARROW_PROJECTION_ID, "outgoing_from_active_layer6ii"),
    ):
        projection = projections[projection_id]
        before = np.asarray(before_weights[projection_id], dtype=float)
        after = np.asarray(projection.w[:], dtype=float)
        if projection_id == BOTTOM_UP_PROJECTION_ID:
            before_map = _incoming_map(projection, before, target_index=winning_layer4_index)
            after_map = _incoming_map(projection, after, target_index=winning_layer4_index)
        else:
            before_map = _outgoing_map(projection, before, source_index=active_category_index)
            after_map = _outgoing_map(projection, after, source_index=active_category_index)
        summaries[projection_id] = Figure6MapSummary(
            projection_id=projection_id,
            map_role=role,
            before=tuple(float(value) for value in before_map),
            after=tuple(float(value) for value in after_map),
        )
    return Figure6LearningResult(
        convention_fingerprint=convention_fingerprint,
        duration_ms=duration_ms,
        population_spikes=population_spikes,
        bottom_up=summaries[BOTTOM_UP_PROJECTION_ID],
        top_down_wide=summaries[TOP_DOWN_WIDE_PROJECTION_ID],
        top_down_narrow=summaries[TOP_DOWN_NARROW_PROJECTION_ID],
    )


def _pre_signal(elapsed_ms: np.ndarray, *, rise_ms: float, fall_ms: float) -> np.ndarray:
    signal = np.zeros_like(elapsed_ms)
    active = elapsed_ms >= 0
    normalization = biexponential_normalization(rise_ms, fall_ms)
    signal[active] = normalization * (
        np.exp(-elapsed_ms[active] / fall_ms) - np.exp(-elapsed_ms[active] / rise_ms)
    )
    return signal


def run_figure6_timing_curves(
    protocol: Figure6TimingProtocol | None = None,
) -> Figure6TimingResult:
    """Integrate Equations 25/28 for the five Figure 6a gating families."""

    protocol = protocol or Figure6TimingProtocol()
    depression = equilibrium_depression_scale(
        minimum_weight=0.0,
        baseline_weight=protocol.baseline_weight,
        maximum_weight=protocol.maximum_weight,
    )
    curves = {rule: [] for rule in LEARNING_RULES}
    for relative_ms in protocol.relative_times_ms:
        pre_time = max(0.0, -relative_ms)
        post_time = max(0.0, relative_ms)
        stop = max(pre_time, post_time) + protocol.depotentiation_ms + 1.0
        times = np.arange(0.0, stop + protocol.dt_ms / 2, protocol.dt_ms)
        pre = _pre_signal(times - pre_time, rise_ms=protocol.rise_ms, fall_ms=protocol.fall_ms)
        post = full_postsynaptic_learning_signal(
            times - post_time,
            depression_scale=depression,
            spike_above_threshold_ms=protocol.spike_above_threshold_ms,
            depotentiation_ms=protocol.depotentiation_ms,
        )
        for rule in LEARNING_RULES:
            weight = protocol.initial_weight
            for pre_value, post_value in zip(pre, post, strict=True):
                weight += protocol.dt_ms * gated_weight_derivative(
                    weight=weight,
                    minimum_weight=0.0,
                    baseline_weight=protocol.baseline_weight,
                    maximum_weight=protocol.maximum_weight,
                    pre_signal=float(pre_value),
                    post_signal=float(post_value),
                    learning_rate=protocol.learning_rate_per_ms,
                    learning_rule=rule,
                )
                weight = float(np.clip(weight, 0.0, protocol.maximum_weight))
            curves[rule].append(weight - protocol.initial_weight)
    return Figure6TimingResult(
        protocol=protocol,
        curves={rule: tuple(values) for rule, values in curves.items()},
    )
