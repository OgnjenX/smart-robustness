"""Event-locked Figure 10 search-cycle reconstruction.

This module is deliberately separate from the hash-pinned Figure 10 harness.
It implements the source-audited transition from a masked failed-recognition
state to reinstated bottom-up input without changing the historical harness or
any model equation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..protocols import (
    ClassicMatchMismatchCue,
    MatchCondition,
    apply_bar_stimulus,
    apply_match_mismatch_cue,
    clear_match_mismatch_cue,
)
from .figure7 import (
    TopDownCurrentMode,
    apply_figure7_learned_state,
    top_k_comparator_relay_input_gains,
)
from .figure10 import FIGURE10_RESET_INPUT_PROJECTIONS

VERTICAL_TARGETS = (22, 31, 40, 49, 58)
VERTICAL_ALTERNATIVES = (22, 31, 49, 58)


@dataclass(frozen=True, slots=True)
class Figure10SearchCycleConditionResult:
    """Bounded evidence from one event-locked Figure 10 condition."""

    reset_pathway_enabled: bool
    pre_layer4_events: int
    pre_layer4_active_indices: tuple[int, ...]
    mismatch_layer4_events: tuple[tuple[int, float], ...]
    winner_post_events: int
    alternative_events: int
    alternative_active_indices: tuple[int, ...]
    first_alternative_event_ms: float | None
    alternative_events_before_release: int
    nonspecific_post_events: int
    layer5_post_events: int
    layer6i_post_events: int
    layer4_inhibitory_post_events: int
    projection035_post_release_integral_pA_ms: tuple[tuple[int, float], ...]


def run_figure10_search_cycle_condition(
    *,
    top_down_current_pA: float,
    pre_match_duration_ms: float,
    mismatch_duration_ms: float,
    release_after_mismatch_ms: float,
    reset_pathway_enabled: bool,
    learned_weights: Mapping[str, tuple[float, ...] | np.ndarray],
    persistent_projection_weight_scales: Mapping[str, float],
    persistent_projection_delays_ms: Mapping[str, float] | None = None,
    comparator_top_k_targets: int = 5,
    comparator_source_index: int = 40,
    top_down_current_mode: TopDownCurrentMode | str = (
        TopDownCurrentMode.UNTIL_CUED_CELL_FIRST_EVENT
    ),
    conventions=None,
    dt_ms: float = 0.01,
    brian=None,
) -> Figure10SearchCycleConditionResult:
    """Run one yoked masked-to-full-input search-cycle condition."""

    if pre_match_duration_ms <= 0 or mismatch_duration_ms <= 0 or dt_ms <= 0:
        raise ValueError("Figure 10 durations and dt_ms must be positive")
    if not 0 < release_after_mismatch_ms < mismatch_duration_ms:
        raise ValueError("release marker must lie strictly inside mismatch")
    if top_down_current_pA <= 0:
        raise ValueError("top-down current must be positive")
    current_mode = TopDownCurrentMode(top_down_current_mode)
    if current_mode is not TopDownCurrentMode.UNTIL_CUED_CELL_FIRST_EVENT:
        raise ValueError("search-cycle assay requires first-event cue clearing")
    if brian is None:
        import brian2 as brian

    from ..classic_sector import build_first_order_connected_sector

    brian.start_scope()
    brian.defaultclock.dt = dt_ms * brian.ms
    sector = build_first_order_connected_sector(conventions=conventions, brian=brian)

    unknown_scales = set(persistent_projection_weight_scales) - set(
        sector.projections
    )
    if unknown_scales:
        raise ValueError(f"unknown projection scale IDs: {sorted(unknown_scales)}")
    for projection_id, scale in persistent_projection_weight_scales.items():
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("projection weight scales must be finite and positive")
        projection = sector.projections[projection_id]
        for block in getattr(projection, "blocks", (projection,)):
            block.w = f"w*({float(scale)!r})"

    delays = persistent_projection_delays_ms or {}
    unknown_delays = set(delays) - set(sector.projections)
    if unknown_delays:
        raise ValueError(f"unknown projection delay IDs: {sorted(unknown_delays)}")
    for projection_id, delay_ms in delays.items():
        if not np.isfinite(delay_ms) or delay_ms <= 0:
            raise ValueError("projection delays must be finite and positive")
        projection = sector.projections[projection_id]
        for block in getattr(projection, "blocks", (projection,)):
            block.delay = float(delay_ms) * brian.ms

    apply_figure7_learned_state(sector.projections, learned_weights)
    masked_gains = top_k_comparator_relay_input_gains(
        learned_weights,
        target_count=comparator_top_k_targets,
        source_index=comparator_source_index,
    )
    full_gains = np.ones(81, dtype=float)

    layer4 = brian.SpikeMonitor(
        sector.populations["layer4_excitatory_v1"].group,
        name=(
            "search_cycle_intact_layer4"
            if reset_pathway_enabled
            else "search_cycle_control_layer4"
        ),
    )
    nonspecific = brian.SpikeMonitor(
        sector.populations["thalamic_nonspecific"].group,
        name=(
            "search_cycle_intact_nonspecific"
            if reset_pathway_enabled
            else "search_cycle_control_nonspecific"
        ),
    )
    layer5 = brian.SpikeMonitor(
        sector.populations["layer5_excitatory_v1"].group,
        name=(
            "search_cycle_intact_layer5"
            if reset_pathway_enabled
            else "search_cycle_control_layer5"
        ),
    )
    layer6i = brian.SpikeMonitor(
        sector.populations["layer6i_excitatory_v1"].group,
        name=(
            "search_cycle_intact_layer6i"
            if reset_pathway_enabled
            else "search_cycle_control_layer6i"
        ),
    )
    layer4_inhibitory = brian.SpikeMonitor(
        sector.populations["layer4_inhibitory_v1"].group,
        name=(
            "search_cycle_intact_layer4_inhibitory"
            if reset_pathway_enabled
            else "search_cycle_control_layer4_inhibitory"
        ),
    )
    layer4_input = brian.StateMonitor(
        sector.populations["layer4_excitatory_v1"].group,
        "i_port_000",
        record=VERTICAL_TARGETS,
        name=(
            "search_cycle_intact_layer4_input"
            if reset_pathway_enabled
            else "search_cycle_control_layer4_input"
        ),
    )
    sector.network.add(
        layer4,
        nonspecific,
        layer5,
        layer6i,
        layer4_inhibitory,
        layer4_input,
    )

    match = ClassicMatchMismatchCue(
        condition=MatchCondition.MATCH,
        top_down_current_pA=top_down_current_pA,
        duration_ms=pre_match_duration_ms,
    )
    category = sector.populations[match.top_down_population].group
    category.clear_drive_on_spike = 0
    category.clear_drive_on_spike[match.top_down_cell_index] = 1
    apply_match_mismatch_cue(
        sector, match, relay_input_gains=masked_gains, brian=brian
    )
    sector.network.run(pre_match_duration_ms * brian.ms)
    clear_match_mismatch_cue(sector, match, brian=brian)

    if not reset_pathway_enabled:
        for projection_id in FIGURE10_RESET_INPUT_PROJECTIONS:
            sector.projections[projection_id].w = 0

    mismatch = ClassicMatchMismatchCue(
        condition=MatchCondition.MISMATCH,
        top_down_current_pA=top_down_current_pA,
        duration_ms=mismatch_duration_ms,
    )
    category.clear_drive_on_spike = 0
    category.clear_drive_on_spike[mismatch.top_down_cell_index] = 1
    apply_match_mismatch_cue(
        sector, mismatch, relay_input_gains=masked_gains, brian=brian
    )
    sector.network.run(release_after_mismatch_ms * brian.ms)
    apply_bar_stimulus(
        sector,
        mismatch.bottom_up_stimulus,
        relay_input_gains=full_gains,
    )
    sector.network.run(
        (mismatch_duration_ms - release_after_mismatch_ms) * brian.ms
    )
    clear_match_mismatch_cue(sector, mismatch, brian=brian)

    layer4_indices = np.asarray(layer4.i, dtype=int)
    layer4_times_ms = np.asarray(layer4.t / brian.ms, dtype=float)
    pre_mask = layer4_times_ms < pre_match_duration_ms
    mismatch_mask = ~pre_mask
    mismatch_indices = layer4_indices[mismatch_mask]
    mismatch_times = layer4_times_ms[mismatch_mask] - pre_match_duration_ms
    pre_indices = layer4_indices[pre_mask]
    winner_indices = tuple(int(value) for value in np.unique(pre_indices))
    winner_set = set(winner_indices)
    alternative_set = set(VERTICAL_ALTERNATIVES)
    alternative_times = mismatch_times[
        np.asarray([index in alternative_set for index in mismatch_indices])
    ]

    state_times_ms = np.asarray(layer4_input.t / brian.ms, dtype=float)
    post_release = state_times_ms >= (
        pre_match_duration_ms + release_after_mismatch_ms
    )
    post_release_times = state_times_ms[post_release]
    p035_pA = np.asarray(layer4_input.i_port_000 / brian.pA)[:, post_release]
    p035_integrals = tuple(
        (int(target), float(np.trapz(trace, post_release_times)))
        for target, trace in zip(VERTICAL_TARGETS, p035_pA, strict=True)
    )

    def _post_count(monitor) -> int:
        return int(
            np.count_nonzero(
                np.asarray(monitor.t / brian.ms, dtype=float)
                >= pre_match_duration_ms
            )
        )

    mismatch_events = tuple(
        (int(index), float(time_ms))
        for index, time_ms in zip(mismatch_indices, mismatch_times, strict=True)
    )
    return Figure10SearchCycleConditionResult(
        reset_pathway_enabled=reset_pathway_enabled,
        pre_layer4_events=int(np.count_nonzero(pre_mask)),
        pre_layer4_active_indices=winner_indices,
        mismatch_layer4_events=mismatch_events,
        winner_post_events=sum(index in winner_set for index in mismatch_indices),
        alternative_events=sum(
            index in alternative_set for index in mismatch_indices
        ),
        alternative_active_indices=tuple(
            sorted(alternative_set.intersection(set(mismatch_indices.tolist())))
        ),
        first_alternative_event_ms=(
            None if alternative_times.size == 0 else float(np.min(alternative_times))
        ),
        alternative_events_before_release=int(
            np.count_nonzero(alternative_times < release_after_mismatch_ms)
        ),
        nonspecific_post_events=_post_count(nonspecific),
        layer5_post_events=_post_count(layer5),
        layer6i_post_events=_post_count(layer6i),
        layer4_inhibitory_post_events=_post_count(layer4_inhibitory),
        projection035_post_release_integral_pA_ms=p035_integrals,
    )
