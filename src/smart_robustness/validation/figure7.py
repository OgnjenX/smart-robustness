"""Predeclared Figure 7 match/mismatch arousal metrics.

Figure 7 reports approximate firing rates for the single nonspecific thalamic
cell.  It does not provide a complete trace or a numeric reset latency, so the
rate result and the later qualitative reset result are deliberately separate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..protocols import (
    ClassicMatchMismatchCue,
    MatchCondition,
    apply_match_mismatch_cue,
    clear_match_mismatch_cue,
)
from .figure6 import TOP_DOWN_NARROW_PROJECTION_ID, TOP_DOWN_WIDE_PROJECTION_ID

FIGURE7_MATCH_RATE_HZ = 40.0
FIGURE7_MISMATCH_RATE_HZ = 70.0
FIGURE7_RATE_TOLERANCE_HZ = 10.0
FIGURE7_REQUIRED_LEARNED_PROJECTIONS = (
    TOP_DOWN_WIDE_PROJECTION_ID,
    TOP_DOWN_NARROW_PROJECTION_ID,
)


@dataclass(frozen=True, slots=True)
class Figure6ReferenceExpectation:
    """Paper-constrained horizontal expectation for downstream assays.

    Figure 6c provides a combined graphical field but no raw wide/narrow
    weights.  This calibrated state is therefore a downstream reference, not a
    claim that the current network learned the field.
    """

    source_index: int = 40
    peak_combined_weight: float = 2.5
    sigma_x_cells: float = 2.0
    sigma_y_cells: float = 0.8

    def __post_init__(self) -> None:
        if not 0 <= self.source_index < 81:
            raise ValueError("source_index must address the 9x9 sheet")
        if self.peak_combined_weight < 0:
            raise ValueError("peak_combined_weight cannot be negative")
        if self.sigma_x_cells <= 0 or self.sigma_y_cells <= 0:
            raise ValueError("reference sigmas must be positive")


def paper_constrained_figure6_expectation(
    projections: Mapping[str, object],
    reference: Figure6ReferenceExpectation | None = None,
) -> dict[str, tuple[float, ...]]:
    """Construct a labeled Figure 6c-like state for Figure 7/10 tests.

    The combined target is split equally between the archived wide and narrow
    adaptive AMPA channels because the publication does not identify their
    individual post-learning maps.  This underidentification is part of the
    returned state's documented provenance.
    """

    reference = reference or Figure6ReferenceExpectation()
    desired = np.zeros(81, dtype=float)
    source_y, source_x = divmod(reference.source_index, 9)
    for index in range(81):
        y, x = divmod(index, 9)
        dx = min(abs(x - source_x), 9 - abs(x - source_x))
        dy = min(abs(y - source_y), 9 - abs(y - source_y))
        desired[index] = reference.peak_combined_weight * np.exp(
            -0.5
            * (
                (dx / reference.sigma_x_cells) ** 2
                + (dy / reference.sigma_y_cells) ** 2
            )
        )
    learned: dict[str, tuple[float, ...]] = {}
    for projection_id in FIGURE7_REQUIRED_LEARNED_PROJECTIONS:
        projection = projections[projection_id]
        values = np.asarray(projection.w[:], dtype=float).copy()
        source = np.asarray(projection.i[:], dtype=int)
        target = np.asarray(projection.j[:], dtype=int)
        selected = source == reference.source_index
        maximum = np.asarray(projection.w_maximum[:], dtype=float)
        values[selected] = np.minimum(desired[target[selected]] / 2.0, maximum[selected])
        learned[projection_id] = tuple(float(value) for value in values)
    return learned


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
    relay_spike_indices: tuple[int, ...] = ()
    relay_spike_times_ms: tuple[float, ...] = ()
    trn_spike_indices: tuple[int, ...] = ()
    trn_spike_times_ms: tuple[float, ...] = ()
    category_spike_indices: tuple[int, ...] = ()
    category_spike_times_ms: tuple[float, ...] = ()
    convention_fingerprint: str | None = None
    top_down_current_pA: float | None = None
    learned_state_provenance: str = "unspecified"

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


def run_figure7_condition(
    *,
    condition: MatchCondition,
    top_down_current_pA: float,
    learned_weights: Mapping[str, tuple[float, ...] | np.ndarray] | None = None,
    use_paper_constrained_reference: bool = False,
    paper_reference: Figure6ReferenceExpectation | None = None,
    conventions=None,
    duration_ms: float = 100.0,
    dt_ms: float = 0.01,
    exact_relay_voltage_clamp: bool = False,
    brian=None,
) -> Figure7ConditionResult:
    """Run one source-labeled Figure 7 match or mismatch condition."""

    if learned_weights is not None and use_paper_constrained_reference:
        raise ValueError("pass learned_weights or request the paper-constrained reference, not both")
    if learned_weights is None and not use_paper_constrained_reference:
        raise ValueError("Figure 7 requires an explicit learned expectation state")
    if duration_ms <= 0 or dt_ms <= 0:
        raise ValueError("duration_ms and dt_ms must be positive")
    if brian is None:
        import brian2 as brian
    from ..classic_sector import (
        build_first_order_connected_sector,
        build_first_order_voltage_clamp_sector,
        figure6_runtime_conventions,
    )

    conventions = conventions or figure6_runtime_conventions()
    brian.start_scope()
    brian.defaultclock.dt = dt_ms * brian.ms
    orientation = ClassicMatchMismatchCue(
        condition=condition,
        top_down_current_pA=top_down_current_pA,
        duration_ms=duration_ms,
    ).bottom_up_stimulus
    if exact_relay_voltage_clamp:
        sector = build_first_order_voltage_clamp_sector(
            clamped_relay_indices=orientation.active_indices,
            holding_mV=orientation.expected_holding_mV,
            conventions=conventions,
            brian=brian,
        )
    else:
        sector = build_first_order_connected_sector(conventions=conventions, brian=brian)
    if use_paper_constrained_reference:
        learned_weights = paper_constrained_figure6_expectation(
            sector.projections, paper_reference
        )
        provenance = "paper-constrained-figure6c-reference"
    else:
        provenance = "simulated-learned-weight-snapshot"
    assert learned_weights is not None
    apply_figure7_learned_state(sector.projections, learned_weights)
    nonspecific = brian.SpikeMonitor(
        sector.populations["thalamic_nonspecific"].group,
        name=f"figure7_{condition.value}_nonspecific_spikes",
    )
    layer4 = brian.SpikeMonitor(
        sector.populations["layer4_excitatory_v1"].group,
        name=f"figure7_{condition.value}_layer4_spikes",
    )
    relay = brian.SpikeMonitor(
        sector.populations["thalamic_relay"].group,
        name=f"figure7_{condition.value}_relay_spikes",
    )
    trn = brian.SpikeMonitor(
        sector.populations["trn"].group,
        name=f"figure7_{condition.value}_trn_spikes",
    )
    category = brian.SpikeMonitor(
        sector.populations["layer6ii_excitatory_v1"].group,
        name=f"figure7_{condition.value}_category_spikes",
    )
    sector.network.add(nonspecific, layer4, relay, trn, category)
    cue = ClassicMatchMismatchCue(
        condition=condition,
        top_down_current_pA=top_down_current_pA,
        duration_ms=duration_ms,
    )
    apply_match_mismatch_cue(
        sector, cue, apply_relay_input=not exact_relay_voltage_clamp, brian=brian
    )
    sector.network.run(duration_ms * brian.ms)
    clear_match_mismatch_cue(sector, cue, brian=brian)
    return Figure7ConditionResult(
        condition=condition,
        duration_ms=duration_ms,
        nonspecific_spike_times_ms=tuple(float(value) for value in nonspecific.t / brian.ms),
        layer4_spike_times_ms=tuple(float(value) for value in layer4.t / brian.ms),
        relay_spike_indices=tuple(int(value) for value in relay.i),
        relay_spike_times_ms=tuple(float(value) for value in relay.t / brian.ms),
        trn_spike_indices=tuple(int(value) for value in trn.i),
        trn_spike_times_ms=tuple(float(value) for value in trn.t / brian.ms),
        category_spike_indices=tuple(int(value) for value in category.i),
        category_spike_times_ms=tuple(float(value) for value in category.t / brian.ms),
        convention_fingerprint=conventions.fingerprint,
        top_down_current_pA=top_down_current_pA,
        learned_state_provenance=provenance,
    )
