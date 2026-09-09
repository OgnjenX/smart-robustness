"""Passive target-balance audit for the event-locked Figure 10 search cycle.

This module is deliberately separate from the hash-pinned Figure 10 harness.
It repeats the closed search-cycle endpoint with bounded current, voltage, and
source-arrival readouts without changing any executable model quantity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

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
from .figure10 import (
    FIGURE10_RESET_INPUT_PROJECTIONS,
    Figure10Layer4TargetTimingSummary,
    Figure10Projection026TargetArrivalSummary,
    Figure10Projection036TargetArrivalSummary,
    summarize_layer4_target_timing,
    summarize_projection026_target_arrivals,
    summarize_projection036_target_arrivals,
)

VERTICAL_TARGETS = (22, 31, 40, 49, 58)
VERTICAL_ALTERNATIVES = (22, 31, 49, 58)
AUDIT_TARGETS = (22, 31, 38, 39, 40, 41, 42, 49, 58)
ARRIVAL_TARGETS = (31, 40)


@dataclass(frozen=True, slots=True)
class Figure10SearchCycleBalanceConditionResult:
    """Bounded evidence from one passive event-locked balance audit."""

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
    target_timing_summaries: tuple[Figure10Layer4TargetTimingSummary, ...]
    projection026_arrival_summaries: tuple[
        Figure10Projection026TargetArrivalSummary, ...
    ]
    projection036_arrival_summaries: tuple[
        Figure10Projection036TargetArrivalSummary, ...
    ]


def run_figure10_search_cycle_balance_condition(
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
    audit_window_ms: tuple[float, float] = (61.0, 110.0),
    audit_bin_width_ms: float = 1.0,
    brian=None,
) -> Figure10SearchCycleBalanceConditionResult:
    """Repeat one search-cycle condition with passive balance readouts."""

    if pre_match_duration_ms <= 0 or mismatch_duration_ms <= 0 or dt_ms <= 0:
        raise ValueError("Figure 10 durations and dt_ms must be positive")
    if not 0 < release_after_mismatch_ms < mismatch_duration_ms:
        raise ValueError("release marker must lie strictly inside mismatch")
    if top_down_current_pA <= 0:
        raise ValueError("top-down current must be positive")
    if (
        len(audit_window_ms) != 2
        or audit_window_ms[0] < 0
        or audit_window_ms[1] <= audit_window_ms[0]
        or audit_window_ms[1] > mismatch_duration_ms
    ):
        raise ValueError("audit window must lie within mismatch")
    if audit_bin_width_ms <= 0:
        raise ValueError("audit bin width must be positive")
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
    layer4_state = brian.StateMonitor(
        sector.populations["layer4_excitatory_v1"].group,
        (
            "v_soma",
            "port_000_gate",
            "i_port_000",
            "port_001_gate",
            "i_port_001",
            "i_port_002",
            "port_003_gate",
            "i_port_003",
        ),
        record=AUDIT_TARGETS,
        name=(
            "search_cycle_balance_intact_layer4_state"
            if reset_pathway_enabled
            else "search_cycle_balance_control_layer4_state"
        ),
    )
    sector.network.add(
        layer4,
        nonspecific,
        layer5,
        layer6i,
        layer4_inhibitory,
        layer4_state,
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

    state_times_ms = np.asarray(layer4_state.t / brian.ms, dtype=float)
    post_release = state_times_ms >= (
        pre_match_duration_ms + release_after_mismatch_ms
    )
    post_release_times = state_times_ms[post_release]
    p035_pA = np.asarray(layer4_state.i_port_000 / brian.pA)[:, post_release]
    target_rows = {target: row for row, target in enumerate(AUDIT_TARGETS)}
    p035_integrals = tuple(
        (
            int(target),
            float(np.trapz(p035_pA[target_rows[target]], post_release_times)),
        )
        for target in VERTICAL_TARGETS
    )

    row_summaries = summarize_layer4_target_timing(
        target_indices=tuple(range(len(AUDIT_TARGETS))),
        mismatch_start_ms=pre_match_duration_ms,
        window_start_from_mismatch_ms=float(audit_window_ms[0]),
        window_end_from_mismatch_ms=float(audit_window_ms[1]),
        dt_ms=dt_ms,
        bin_width_ms=audit_bin_width_ms,
        current_activity_threshold_pA=1e-9,
        gate_activity_threshold=0.1,
        state_times_ms=state_times_ms,
        soma_voltage_mV_by_index=np.asarray(layer4_state.v_soma / brian.mV),
        projection035_current_pA_by_index=np.asarray(
            layer4_state.i_port_000 / brian.pA
        ),
        projection036_current_pA_by_index=np.asarray(
            layer4_state.i_port_001 / brian.pA
        ),
        projection037_current_pA_by_index=np.asarray(
            layer4_state.i_port_002 / brian.pA
        ),
        projection038_current_pA_by_index=np.asarray(
            layer4_state.i_port_003 / brian.pA
        ),
        projection035_gate_by_index=np.asarray(layer4_state.port_000_gate),
        projection036_gate_by_index=np.asarray(layer4_state.port_001_gate),
        projection038_gate_by_index=np.asarray(layer4_state.port_003_gate),
        layer4_spike_indices=layer4_indices,
        layer4_spike_times_ms=layer4_times_ms,
    )
    target_summaries = tuple(
        replace(summary, index=target)
        for target, summary in zip(AUDIT_TARGETS, row_summaries, strict=True)
    )

    def _projection_arrays(projection_id: str):
        projection = sector.projections[projection_id]
        if hasattr(projection, "blocks"):
            return (
                np.asarray(projection.i, dtype=int),
                np.asarray(projection.j, dtype=int),
                np.asarray(projection.read("w"), dtype=float),
                np.concatenate(
                    [
                        np.asarray(block.delay[:] / brian.ms, dtype=float)
                        for block in projection.blocks
                    ]
                ),
            )
        return (
            np.asarray(projection.i[:], dtype=int),
            np.asarray(projection.j[:], dtype=int),
            np.asarray(projection.w[:], dtype=float),
            np.asarray(projection.delay[:] / brian.ms, dtype=float),
        )

    p026_sources, p026_targets, p026_weights, p026_delays = _projection_arrays(
        "modeldb112923.projection.026"
    )
    p036_sources, p036_targets, p036_weights, p036_delays = _projection_arrays(
        "modeldb112923.projection.036"
    )
    p026_arrivals = summarize_projection026_target_arrivals(
        target_indices=ARRIVAL_TARGETS,
        mismatch_start_ms=pre_match_duration_ms,
        window_start_from_mismatch_ms=float(audit_window_ms[0]),
        window_end_from_mismatch_ms=float(audit_window_ms[1]),
        edge_source_indices=p026_sources,
        edge_target_indices=p026_targets,
        edge_weights=p026_weights,
        edge_delays_ms=p026_delays,
        source_spike_indices=np.asarray(layer6i.i),
        source_spike_times_ms=np.asarray(layer6i.t / brian.ms),
    )
    p036_arrivals = summarize_projection036_target_arrivals(
        target_indices=ARRIVAL_TARGETS,
        mismatch_start_ms=pre_match_duration_ms,
        window_start_from_mismatch_ms=float(audit_window_ms[0]),
        window_end_from_mismatch_ms=float(audit_window_ms[1]),
        edge_source_indices=p036_sources,
        edge_target_indices=p036_targets,
        edge_weights=p036_weights,
        edge_delays_ms=p036_delays,
        source_spike_indices=np.asarray(layer4_inhibitory.i),
        source_spike_times_ms=np.asarray(layer4_inhibitory.t / brian.ms),
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
    return Figure10SearchCycleBalanceConditionResult(
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
        target_timing_summaries=target_summaries,
        projection026_arrival_summaries=p026_arrivals,
        projection036_arrival_summaries=p036_arrivals,
    )
