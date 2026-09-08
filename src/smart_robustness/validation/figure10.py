"""Qualitative Figure 10 mismatch-reset validation.

Grossberg and Versace (2008) do not tabulate a reset latency or probability.
This module therefore tests the published causal ordering and winner switch,
with an explicit pathway-disconnection negative control, rather than fitting an
unreported numerical trace.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from ..protocols import (
    ClassicMatchMismatchCue,
    MatchCondition,
    apply_match_mismatch_cue,
    clear_match_mismatch_cue,
)
from .figure7 import (
    TopDownCurrentMode,
    apply_figure7_learned_state,
    paper_constrained_figure6_expectation,
    top_k_comparator_relay_input_gains,
)

FIGURE10_RESET_INPUT_PROJECTIONS = (
    "modeldb112923.projection.017",  # nonspecific thalamus -> layer-5 AMPA
    "modeldb112923.projection.018",  # nonspecific thalamus -> layer-5 NMDA
)


@dataclass(frozen=True, slots=True)
class Figure10Layer6iProjectionTraceSummary:
    """Selected-cell timing summary for one layer-6I synaptic projection."""

    projection_id: str
    gate_integral_ms: float
    current_integral_pA_ms: float
    current_peak_pA: float
    current_peak_time_ms: float
    gate_at_current_peak: float
    gate_peak: float
    gate_peak_time_ms: float
    current_at_gate_peak_pA: float
    soma_voltage_at_current_peak_mV: float
    proximal_voltage_at_current_peak_mV: float
    spike_detector_voltage_at_current_peak_mV: float
    gate_at_soma_peak: float
    current_at_soma_peak_pA: float
    gate_at_proximal_peak: float
    current_at_proximal_peak_pA: float
    gate_at_detector_peak: float
    current_at_detector_peak_pA: float


@dataclass(frozen=True, slots=True)
class Figure10Layer6iTraceSummary:
    """Threshold and input timing for one preregistered layer-6I cell."""

    index: int
    mismatch_event_times_ms: tuple[float, ...]
    spike_detector_threshold_mV: float
    spike_detector_peak_mV: float
    spike_detector_peak_time_ms: float
    threshold_minus_detector_peak_mV: float
    soma_voltage_peak_mV: float
    soma_voltage_peak_time_ms: float
    proximal_voltage_peak_mV: float
    proximal_voltage_peak_time_ms: float
    soma_voltage_at_detector_peak_mV: float
    proximal_voltage_at_detector_peak_mV: float
    projections: tuple[Figure10Layer6iProjectionTraceSummary, ...]


@dataclass(frozen=True, slots=True)
class Figure10Layer4BalanceBin:
    """Time-binned layer-6I-to-layer-4 reset-path balance."""

    start_from_mismatch_ms: float
    end_from_mismatch_ms: float
    projection026_excitation_integral_pA_ms: float
    layer4_inhibitory_events: int
    projection036_inhibition_integral_pA_ms: float
    projection038_excitation_integral_pA_ms: float
    layer4_winner_events: int
    layer4_alternative_events: int
    layer4_alternative_active_count: int


@dataclass(frozen=True, slots=True)
class Figure10Layer4TargetBalanceBin:
    """Time-binned projection balance at one layer-4 excitatory cell."""

    index: int
    start_from_mismatch_ms: float
    end_from_mismatch_ms: float
    projection035_relay_excitation_integral_pA_ms: float
    projection036_inhibition_integral_pA_ms: float
    projection037_recurrent_excitation_integral_pA_ms: float
    projection038_excitation_integral_pA_ms: float
    layer4_events: int


@dataclass(frozen=True, slots=True)
class Figure10Layer4TargetTimingSummary:
    """One-ms causal-order summary for one layer-4 excitatory cell."""

    index: int
    window_start_from_mismatch_ms: float
    window_end_from_mismatch_ms: float
    bin_edges_from_mismatch_ms: tuple[float, ...]
    projection035_relay_excitation_integral_pA_ms: tuple[float, ...]
    projection036_inhibition_integral_pA_ms: tuple[float, ...]
    projection037_recurrent_excitation_integral_pA_ms: tuple[float, ...]
    projection038_excitation_integral_pA_ms: tuple[float, ...]
    projection035_relay_gate_mean: tuple[float, ...]
    projection036_inhibition_gate_mean: tuple[float, ...]
    soma_voltage_min_mV: tuple[float, ...]
    soma_voltage_max_mV: tuple[float, ...]
    layer4_spike_times_from_mismatch_ms: tuple[float, ...]
    projection037_first_active_time_from_mismatch_ms: float | None


@dataclass(frozen=True, slots=True)
class Figure10ConditionResult:
    """Spike evidence from one persistent pre-reset then mismatch episode."""

    pre_match_duration_ms: float
    mismatch_duration_ms: float
    reset_pathway_enabled: bool
    layer4_spike_indices: tuple[int, ...]
    layer4_spike_times_ms: tuple[float, ...]
    nonspecific_spike_times_ms: tuple[float, ...] = ()
    layer5_spike_indices: tuple[int, ...] = ()
    layer5_spike_times_ms: tuple[float, ...] = ()
    layer6i_spike_indices: tuple[int, ...] = ()
    layer6i_spike_times_ms: tuple[float, ...] = ()
    layer4_inhibitory_spike_indices: tuple[int, ...] = ()
    layer4_inhibitory_spike_times_ms: tuple[float, ...] = ()
    convention_fingerprint: str | None = None
    learned_state_provenance: str | None = None
    comparator_target_count: int | None = None
    top_down_current_mode: str = TopDownCurrentMode.SUSTAINED_EPOCH.value
    layer6i_mismatch_gate_integral_ms_by_projection: tuple[tuple[str, float], ...] = ()
    layer6i_mismatch_current_integral_pA_ms_by_projection: tuple[tuple[str, float], ...] = ()
    layer6i_mismatch_current_peak_pA_by_projection: tuple[tuple[str, float], ...] = ()
    layer6i_mismatch_event_transmitter_samples: tuple[tuple[int, float, float], ...] = ()
    layer6i_mismatch_projection025_gate_integral_ms_by_index: tuple[tuple[int, float], ...] = ()
    layer6i_mismatch_projection025_current_integral_pA_ms_by_index: tuple[
        tuple[int, float], ...
    ] = ()
    layer6i_mismatch_projection025_current_peak_pA_by_index: tuple[tuple[int, float], ...] = ()
    layer6i_mismatch_soma_voltage_peak_mV_by_index: tuple[tuple[int, float], ...] = ()
    layer6i_mismatch_proximal_voltage_peak_mV_by_index: tuple[tuple[int, float], ...] = ()
    layer6i_selected_trace_summaries: tuple[Figure10Layer6iTraceSummary, ...] = ()
    layer6i_replay_trace_path: str | None = None
    layer6i_replay_trace_sha256: str | None = None
    layer4i_mismatch_projection026_gate_integral_ms: float | None = None
    layer4i_mismatch_projection026_current_integral_pA_ms: float | None = None
    layer4i_mismatch_projection026_current_peak_pA: float | None = None
    layer4e_mismatch_projection036_gate_integral_ms: float | None = None
    layer4e_mismatch_projection036_current_integral_pA_ms: float | None = None
    layer4e_mismatch_projection036_current_trough_pA: float | None = None
    layer4e_mismatch_projection038_gate_integral_ms: float | None = None
    layer4e_mismatch_projection038_current_integral_pA_ms: float | None = None
    layer4e_mismatch_projection038_current_peak_pA: float | None = None
    layer4_balance_bins: tuple[Figure10Layer4BalanceBin, ...] = ()
    layer4_target_balance_bins: tuple[Figure10Layer4TargetBalanceBin, ...] = ()
    layer4_target_timing_summaries: tuple[
        Figure10Layer4TargetTimingSummary, ...
    ] = ()

    def __post_init__(self) -> None:
        if self.pre_match_duration_ms <= 0 or self.mismatch_duration_ms <= 0:
            raise ValueError("Figure 10 phase durations must be positive")
        if len(self.layer4_spike_indices) != len(self.layer4_spike_times_ms):
            raise ValueError("layer-4 spike indices and times must have equal length")
        if self.layer5_spike_indices and len(self.layer5_spike_indices) != len(
            self.layer5_spike_times_ms
        ):
            raise ValueError("layer-5 spike indices and times must have equal length")
        if self.layer6i_spike_indices and len(self.layer6i_spike_indices) != len(
            self.layer6i_spike_times_ms
        ):
            raise ValueError("layer-6I spike indices and times must have equal length")
        if self.layer4_inhibitory_spike_indices and len(
            self.layer4_inhibitory_spike_indices
        ) != len(self.layer4_inhibitory_spike_times_ms):
            raise ValueError("layer-4 inhibitory spike indices and times must have equal length")

    def layer4_counts(self, *, after_mismatch: bool) -> np.ndarray:
        indices = np.asarray(self.layer4_spike_indices, dtype=int)
        times = np.asarray(self.layer4_spike_times_ms, dtype=float)
        selected = times >= self.pre_match_duration_ms
        if not after_mismatch:
            selected = ~selected
        return np.bincount(indices[selected], minlength=81)

    def mismatch_spike_count(self, spike_times_ms: tuple[float, ...]) -> int:
        return int(np.count_nonzero(np.asarray(spike_times_ms) >= self.pre_match_duration_ms))


@dataclass(frozen=True, slots=True)
class Figure10ResetAssessment:
    """Predeclared qualitative and causal acceptance gates for Figure 10."""

    pre_reset_winner_index: int | None
    pre_reset_winner_indices: tuple[int, ...]
    pre_reset_winner_spikes: int
    intact_winner_post_spikes: int
    control_winner_post_spikes: int
    intact_released_alternatives: int
    control_released_alternatives: int
    intact_nonspecific_spikes: int
    intact_layer5_spikes: int
    intact_layer6i_spikes: int

    @property
    def pre_reset_winner_pass(self) -> bool:
        return bool(self.pre_reset_winner_indices) and self.pre_reset_winner_spikes > 0

    @property
    def reset_chain_pass(self) -> bool:
        return (
            self.intact_nonspecific_spikes > 0
            and self.intact_layer5_spikes > 0
            and self.intact_layer6i_spikes > 0
        )

    @property
    def winner_suppression_pass(self) -> bool:
        return self.intact_winner_post_spikes < self.control_winner_post_spikes

    @property
    def alternative_release_pass(self) -> bool:
        return self.intact_released_alternatives > self.control_released_alternatives

    @property
    def reproduced_reset(self) -> bool:
        return (
            self.pre_reset_winner_pass
            and self.reset_chain_pass
            and self.winner_suppression_pass
            and self.alternative_release_pass
        )


def assess_figure10_reset(
    intact: Figure10ConditionResult,
    disconnected_control: Figure10ConditionResult,
) -> Figure10ResetAssessment:
    """Compare the intact reset pathway with its causal negative control."""

    if not intact.reset_pathway_enabled or disconnected_control.reset_pathway_enabled:
        raise ValueError("expected an intact result followed by a disconnected control")
    if (
        intact.pre_match_duration_ms != disconnected_control.pre_match_duration_ms
        or intact.mismatch_duration_ms != disconnected_control.mismatch_duration_ms
    ):
        raise ValueError("Figure 10 condition durations must match")
    intact_pre = intact.layer4_counts(after_mismatch=False)
    control_pre = disconnected_control.layer4_counts(after_mismatch=False)
    if not np.array_equal(intact_pre, control_pre):
        raise ValueError("reset-pathway control changed the pre-mismatch layer-4 state")
    winner_indices = tuple(int(value) for value in np.flatnonzero(intact_pre > 0))
    winner = winner_indices[0] if winner_indices else None
    winner_spikes = int(np.sum(intact_pre[list(winner_indices)])) if winner_indices else 0
    intact_post = intact.layer4_counts(after_mismatch=True)
    control_post = disconnected_control.layer4_counts(after_mismatch=True)
    if not winner_indices:
        intact_winner = control_winner = 0
        intact_alternatives = control_alternatives = 0
    else:
        intact_winner = int(np.sum(intact_post[list(winner_indices)]))
        control_winner = int(np.sum(control_post[list(winner_indices)]))
        alternative_mask = ~np.isin(np.arange(81), winner_indices)
        intact_alternatives = int(np.count_nonzero(intact_post[alternative_mask]))
        control_alternatives = int(np.count_nonzero(control_post[alternative_mask]))
    return Figure10ResetAssessment(
        pre_reset_winner_index=winner,
        pre_reset_winner_indices=winner_indices,
        pre_reset_winner_spikes=winner_spikes,
        intact_winner_post_spikes=intact_winner,
        control_winner_post_spikes=control_winner,
        intact_released_alternatives=intact_alternatives,
        control_released_alternatives=control_alternatives,
        intact_nonspecific_spikes=intact.mismatch_spike_count(intact.nonspecific_spike_times_ms),
        intact_layer5_spikes=intact.mismatch_spike_count(intact.layer5_spike_times_ms),
        intact_layer6i_spikes=intact.mismatch_spike_count(intact.layer6i_spike_times_ms),
    )


def summarize_layer6i_selected_traces(
    *,
    selected_indices: tuple[int, ...],
    times_ms: np.ndarray,
    mismatch_start_ms: float,
    spike_detector_threshold_mV: float,
    soma_voltage_mV: np.ndarray,
    proximal_voltage_mV: np.ndarray,
    spike_detector_voltage_mV: np.ndarray,
    gate_by_projection: Mapping[str, np.ndarray],
    current_pA_by_projection: Mapping[str, np.ndarray],
    spike_indices: np.ndarray,
    spike_times_ms: np.ndarray,
) -> tuple[Figure10Layer6iTraceSummary, ...]:
    """Reduce selected layer-6I traces without changing the simulated model."""

    times = np.asarray(times_ms, dtype=float)
    if times.ndim != 1:
        raise ValueError("layer-6I trace times must be one-dimensional")
    window = times >= mismatch_start_ms
    mismatch_times = times[window]
    if mismatch_times.size == 0:
        raise ValueError("layer-6I traces do not contain the mismatch window")
    voltage_arrays = {
        "soma": np.asarray(soma_voltage_mV, dtype=float),
        "proximal": np.asarray(proximal_voltage_mV, dtype=float),
        "detector": np.asarray(spike_detector_voltage_mV, dtype=float),
    }
    projection_ids = tuple(gate_by_projection)
    if set(projection_ids) != set(current_pA_by_projection):
        raise ValueError("layer-6I gate and current projection IDs must match")
    trace_arrays = (
        tuple(voltage_arrays.values())
        + tuple(
            np.asarray(gate_by_projection[projection_id], dtype=float)
            for projection_id in projection_ids
        )
        + tuple(
            np.asarray(current_pA_by_projection[projection_id], dtype=float)
            for projection_id in projection_ids
        )
    )
    expected_shape = voltage_arrays["soma"].shape
    if len(expected_shape) != 2 or expected_shape[1] != times.size:
        raise ValueError("layer-6I traces must have shape (cells, times)")
    if any(values.shape != expected_shape for values in trace_arrays):
        raise ValueError("all layer-6I traces must have the same shape")
    if len(set(selected_indices)) != len(selected_indices):
        raise ValueError("selected layer-6I trace indices must be unique")
    if any(index < 0 or index >= expected_shape[0] for index in selected_indices):
        raise ValueError("selected layer-6I trace index is out of range")

    event_indices = np.asarray(spike_indices, dtype=int)
    event_times = np.asarray(spike_times_ms, dtype=float)
    if event_indices.shape != event_times.shape:
        raise ValueError("layer-6I spike indices and times must have equal shape")

    summaries: list[Figure10Layer6iTraceSummary] = []
    for index in selected_indices:
        soma = voltage_arrays["soma"][index, window]
        proximal = voltage_arrays["proximal"][index, window]
        detector = voltage_arrays["detector"][index, window]
        soma_peak_position = int(np.argmax(soma))
        proximal_peak_position = int(np.argmax(proximal))
        detector_peak_position = int(np.argmax(detector))
        projections: list[Figure10Layer6iProjectionTraceSummary] = []
        for projection_id in projection_ids:
            gate = np.asarray(gate_by_projection[projection_id], dtype=float)[index, window]
            current = np.asarray(current_pA_by_projection[projection_id], dtype=float)[
                index, window
            ]
            current_peak_position = int(np.argmax(current))
            gate_peak_position = int(np.argmax(gate))
            projections.append(
                Figure10Layer6iProjectionTraceSummary(
                    projection_id=projection_id,
                    gate_integral_ms=float(np.trapz(gate, mismatch_times)),
                    current_integral_pA_ms=float(np.trapz(current, mismatch_times)),
                    current_peak_pA=float(current[current_peak_position]),
                    current_peak_time_ms=float(mismatch_times[current_peak_position]),
                    gate_at_current_peak=float(gate[current_peak_position]),
                    gate_peak=float(gate[gate_peak_position]),
                    gate_peak_time_ms=float(mismatch_times[gate_peak_position]),
                    current_at_gate_peak_pA=float(current[gate_peak_position]),
                    soma_voltage_at_current_peak_mV=float(soma[current_peak_position]),
                    proximal_voltage_at_current_peak_mV=float(proximal[current_peak_position]),
                    spike_detector_voltage_at_current_peak_mV=float(
                        detector[current_peak_position]
                    ),
                    gate_at_soma_peak=float(gate[soma_peak_position]),
                    current_at_soma_peak_pA=float(current[soma_peak_position]),
                    gate_at_proximal_peak=float(gate[proximal_peak_position]),
                    current_at_proximal_peak_pA=float(current[proximal_peak_position]),
                    gate_at_detector_peak=float(gate[detector_peak_position]),
                    current_at_detector_peak_pA=float(current[detector_peak_position]),
                )
            )
        selected_events = (event_indices == index) & (event_times >= mismatch_start_ms)
        detector_peak = float(detector[detector_peak_position])
        summaries.append(
            Figure10Layer6iTraceSummary(
                index=index,
                mismatch_event_times_ms=tuple(
                    float(value) for value in event_times[selected_events]
                ),
                spike_detector_threshold_mV=float(spike_detector_threshold_mV),
                spike_detector_peak_mV=detector_peak,
                spike_detector_peak_time_ms=float(mismatch_times[detector_peak_position]),
                threshold_minus_detector_peak_mV=float(spike_detector_threshold_mV - detector_peak),
                soma_voltage_peak_mV=float(soma[soma_peak_position]),
                soma_voltage_peak_time_ms=float(mismatch_times[soma_peak_position]),
                proximal_voltage_peak_mV=float(proximal[proximal_peak_position]),
                proximal_voltage_peak_time_ms=float(mismatch_times[proximal_peak_position]),
                soma_voltage_at_detector_peak_mV=float(soma[detector_peak_position]),
                proximal_voltage_at_detector_peak_mV=float(proximal[detector_peak_position]),
                projections=tuple(projections),
            )
        )
    return tuple(summaries)


def summarize_layer4_balance_bins(
    *,
    mismatch_start_ms: float,
    mismatch_duration_ms: float,
    dt_ms: float,
    bin_width_ms: float,
    state_times_ms: np.ndarray,
    projection026_current_pA: np.ndarray,
    projection036_current_pA: np.ndarray,
    projection038_current_pA: np.ndarray,
    layer4_inhibitory_spike_times_ms: np.ndarray,
    layer4_spike_indices: np.ndarray,
    layer4_spike_times_ms: np.ndarray,
    winner_indices: tuple[int, ...],
) -> tuple[Figure10Layer4BalanceBin, ...]:
    """Reduce the reset-path currents and events into fixed causal time bins."""

    if mismatch_duration_ms <= 0 or dt_ms <= 0 or bin_width_ms <= 0:
        raise ValueError("layer-4 balance timing values must be positive")
    times = np.asarray(state_times_ms, dtype=float)
    current_traces = tuple(
        np.asarray(trace, dtype=float)
        for trace in (
            projection026_current_pA,
            projection036_current_pA,
            projection038_current_pA,
        )
    )
    if times.ndim != 1 or any(trace.shape != times.shape for trace in current_traces):
        raise ValueError("layer-4 balance currents must match the one-dimensional time grid")
    l4_indices = np.asarray(layer4_spike_indices, dtype=int)
    l4_times = np.asarray(layer4_spike_times_ms, dtype=float)
    if l4_indices.shape != l4_times.shape:
        raise ValueError("layer-4 balance spike indices and times must match")
    winner_set = np.asarray(winner_indices, dtype=int)
    summaries: list[Figure10Layer4BalanceBin] = []
    start = 0.0
    while start < mismatch_duration_ms - dt_ms / 2:
        end = min(start + bin_width_ms, mismatch_duration_ms)
        absolute_start = mismatch_start_ms + start
        absolute_end = mismatch_start_ms + end
        state_window = (times >= absolute_start) & (times < absolute_end)
        inhibitory_events = np.asarray(layer4_inhibitory_spike_times_ms, dtype=float)
        inhibitory_count = int(
            np.count_nonzero(
                (inhibitory_events >= absolute_start)
                & (inhibitory_events < absolute_end)
            )
        )
        layer4_window = (l4_times >= absolute_start) & (l4_times < absolute_end)
        layer4_bin_indices = l4_indices[layer4_window]
        winner_mask = np.isin(layer4_bin_indices, winner_set)
        alternative_indices = layer4_bin_indices[~winner_mask]
        summaries.append(
            Figure10Layer4BalanceBin(
                start_from_mismatch_ms=start,
                end_from_mismatch_ms=end,
                projection026_excitation_integral_pA_ms=float(
                    np.sum(current_traces[0][state_window]) * dt_ms
                ),
                layer4_inhibitory_events=inhibitory_count,
                projection036_inhibition_integral_pA_ms=float(
                    np.sum(current_traces[1][state_window]) * dt_ms
                ),
                projection038_excitation_integral_pA_ms=float(
                    np.sum(current_traces[2][state_window]) * dt_ms
                ),
                layer4_winner_events=int(np.count_nonzero(winner_mask)),
                layer4_alternative_events=int(alternative_indices.size),
                layer4_alternative_active_count=int(np.unique(alternative_indices).size),
            )
        )
        start = end
    return tuple(summaries)


def summarize_layer4_target_balance_bins(
    *,
    target_indices: tuple[int, ...],
    mismatch_start_ms: float,
    mismatch_duration_ms: float,
    dt_ms: float,
    bin_width_ms: float,
    state_times_ms: np.ndarray,
    projection035_current_pA_by_index: np.ndarray,
    projection036_current_pA_by_index: np.ndarray,
    projection037_current_pA_by_index: np.ndarray,
    projection038_current_pA_by_index: np.ndarray,
    layer4_spike_indices: np.ndarray,
    layer4_spike_times_ms: np.ndarray,
) -> tuple[Figure10Layer4TargetBalanceBin, ...]:
    """Reduce focal layer-4 currents and events without changing simulation."""

    if mismatch_duration_ms <= 0 or dt_ms <= 0 or bin_width_ms <= 0:
        raise ValueError("layer-4 target balance timing values must be positive")
    if len(set(target_indices)) != len(target_indices):
        raise ValueError("layer-4 target balance indices must be unique")
    times = np.asarray(state_times_ms, dtype=float)
    currents = tuple(
        np.asarray(current, dtype=float)
        for current in (
            projection035_current_pA_by_index,
            projection036_current_pA_by_index,
            projection037_current_pA_by_index,
            projection038_current_pA_by_index,
        )
    )
    current035, current036, current037, current038 = currents
    if times.ndim != 1 or current036.ndim != 2 or any(
        current.shape != current036.shape for current in currents
    ):
        raise ValueError("layer-4 target current arrays must be matching matrices")
    if current036.shape[1] != times.size:
        raise ValueError("layer-4 target currents must match the time grid")
    if any(index < 0 or index >= current036.shape[0] for index in target_indices):
        raise ValueError("layer-4 target balance index is outside the current matrix")
    spike_indices = np.asarray(layer4_spike_indices, dtype=int)
    spike_times = np.asarray(layer4_spike_times_ms, dtype=float)
    if spike_indices.shape != spike_times.shape:
        raise ValueError("layer-4 target spike indices and times must match")

    summaries: list[Figure10Layer4TargetBalanceBin] = []
    start = 0.0
    while start < mismatch_duration_ms - dt_ms / 2:
        end = min(start + bin_width_ms, mismatch_duration_ms)
        absolute_start = mismatch_start_ms + start
        absolute_end = mismatch_start_ms + end
        state_window = (times >= absolute_start) & (times < absolute_end)
        spike_window = (spike_times >= absolute_start) & (spike_times < absolute_end)
        for index in target_indices:
            summaries.append(
                Figure10Layer4TargetBalanceBin(
                    index=index,
                    start_from_mismatch_ms=start,
                    end_from_mismatch_ms=end,
                    projection035_relay_excitation_integral_pA_ms=float(
                        np.sum(current035[index, state_window]) * dt_ms
                    ),
                    projection036_inhibition_integral_pA_ms=float(
                        np.sum(current036[index, state_window]) * dt_ms
                    ),
                    projection037_recurrent_excitation_integral_pA_ms=float(
                        np.sum(current037[index, state_window]) * dt_ms
                    ),
                    projection038_excitation_integral_pA_ms=float(
                        np.sum(current038[index, state_window]) * dt_ms
                    ),
                    layer4_events=int(
                        np.count_nonzero(spike_indices[spike_window] == index)
                    ),
                )
            )
        start = end
    return tuple(summaries)


def compact_layer4_target_balance_summary(summary: dict) -> dict:
    """Store complete per-target bins as exact arrays instead of repeated rows."""

    rows = summary.pop("layer4_target_balance_bins")
    if not rows:
        summary["layer4_target_balance_bin_edges_ms"] = []
        summary["layer4_target_balance_series"] = {}
        return summary
    fields = (
        "projection035_relay_excitation_integral_pA_ms",
        "projection036_inhibition_integral_pA_ms",
        "projection037_recurrent_excitation_integral_pA_ms",
        "projection038_excitation_integral_pA_ms",
        "layer4_events",
    )
    indices = sorted({int(row["index"]) for row in rows})
    first_index_rows = [row for row in rows if int(row["index"]) == indices[0]]
    summary["layer4_target_balance_bin_edges_ms"] = [
        float(first_index_rows[0]["start_from_mismatch_ms"]),
        *(float(row["end_from_mismatch_ms"]) for row in first_index_rows),
    ]
    summary["layer4_target_balance_series_columns"] = list(fields)
    summary["layer4_target_balance_series"] = {
        str(index): [
            [row[field] for row in rows if int(row["index"]) == index]
            for field in fields
        ]
        for index in indices
    }
    return summary


def summarize_layer4_target_timing(
    *,
    target_indices: tuple[int, ...],
    mismatch_start_ms: float,
    window_start_from_mismatch_ms: float,
    window_end_from_mismatch_ms: float,
    dt_ms: float,
    bin_width_ms: float,
    current_activity_threshold_pA: float,
    state_times_ms: np.ndarray,
    soma_voltage_mV_by_index: np.ndarray,
    projection035_current_pA_by_index: np.ndarray,
    projection036_current_pA_by_index: np.ndarray,
    projection037_current_pA_by_index: np.ndarray,
    projection038_current_pA_by_index: np.ndarray,
    projection035_gate_by_index: np.ndarray,
    projection036_gate_by_index: np.ndarray,
    layer4_spike_indices: np.ndarray,
    layer4_spike_times_ms: np.ndarray,
) -> tuple[Figure10Layer4TargetTimingSummary, ...]:
    """Reduce a focal trace while preserving sub-bin event/current ordering."""

    if dt_ms <= 0 or bin_width_ms <= 0 or current_activity_threshold_pA < 0:
        raise ValueError("layer-4 target timing values are invalid")
    if window_start_from_mismatch_ms < 0 or (
        window_end_from_mismatch_ms <= window_start_from_mismatch_ms
    ):
        raise ValueError("layer-4 target timing window is invalid")
    if len(set(target_indices)) != len(target_indices):
        raise ValueError("layer-4 target timing indices must be unique")
    times = np.asarray(state_times_ms, dtype=float)
    matrices = tuple(
        np.asarray(value, dtype=float)
        for value in (
            soma_voltage_mV_by_index,
            projection035_current_pA_by_index,
            projection036_current_pA_by_index,
            projection037_current_pA_by_index,
            projection038_current_pA_by_index,
            projection035_gate_by_index,
            projection036_gate_by_index,
        )
    )
    voltage, current035, current036, current037, current038, gate035, gate036 = matrices
    if times.ndim != 1 or voltage.ndim != 2 or any(
        value.shape != voltage.shape for value in matrices
    ):
        raise ValueError("layer-4 target timing arrays must be matching matrices")
    if voltage.shape[1] != times.size:
        raise ValueError("layer-4 target timing arrays must match the time grid")
    if any(index < 0 or index >= voltage.shape[0] for index in target_indices):
        raise ValueError("layer-4 target timing index is outside the state matrix")
    spike_indices = np.asarray(layer4_spike_indices, dtype=int)
    spike_times = np.asarray(layer4_spike_times_ms, dtype=float)
    if spike_indices.shape != spike_times.shape:
        raise ValueError("layer-4 target timing spike indices and times must match")

    absolute_start = mismatch_start_ms + window_start_from_mismatch_ms
    absolute_end = mismatch_start_ms + window_end_from_mismatch_ms
    trace_window = (times >= absolute_start) & (times < absolute_end)
    trace_times = times[trace_window]
    if trace_times.size == 0:
        raise ValueError("layer-4 target timing window contains no state samples")
    edges = [window_start_from_mismatch_ms]
    while edges[-1] < window_end_from_mismatch_ms - dt_ms / 2:
        edges.append(min(edges[-1] + bin_width_ms, window_end_from_mismatch_ms))

    summaries: list[Figure10Layer4TargetTimingSummary] = []
    for index in target_indices:
        current_series = (current035[index], current036[index], current037[index], current038[index])
        integrals = [[] for _ in current_series]
        voltage_min: list[float] = []
        voltage_max: list[float] = []
        gate035_mean: list[float] = []
        gate036_mean: list[float] = []
        for start, end in pairwise(edges):
            window = (
                (times >= mismatch_start_ms + start)
                & (times < mismatch_start_ms + end)
            )
            if not np.any(window):
                raise ValueError("layer-4 target timing bin contains no state samples")
            for values, output in zip(current_series, integrals, strict=True):
                output.append(float(np.sum(values[window]) * dt_ms))
            voltage_min.append(float(np.min(voltage[index, window])))
            voltage_max.append(float(np.max(voltage[index, window])))
            gate035_mean.append(float(np.mean(gate035[index, window])))
            gate036_mean.append(float(np.mean(gate036[index, window])))
        active = np.flatnonzero(
            np.abs(current037[index, trace_window]) > current_activity_threshold_pA
        )
        target_spikes = spike_times[
            (spike_indices == index)
            & (spike_times >= absolute_start)
            & (spike_times < absolute_end)
        ]
        summaries.append(
            Figure10Layer4TargetTimingSummary(
                index=index,
                window_start_from_mismatch_ms=window_start_from_mismatch_ms,
                window_end_from_mismatch_ms=window_end_from_mismatch_ms,
                bin_edges_from_mismatch_ms=tuple(float(value) for value in edges),
                projection035_relay_excitation_integral_pA_ms=tuple(integrals[0]),
                projection036_inhibition_integral_pA_ms=tuple(integrals[1]),
                projection037_recurrent_excitation_integral_pA_ms=tuple(integrals[2]),
                projection038_excitation_integral_pA_ms=tuple(integrals[3]),
                projection035_relay_gate_mean=tuple(gate035_mean),
                projection036_inhibition_gate_mean=tuple(gate036_mean),
                soma_voltage_min_mV=tuple(voltage_min),
                soma_voltage_max_mV=tuple(voltage_max),
                layer4_spike_times_from_mismatch_ms=tuple(
                    float(value - mismatch_start_ms) for value in target_spikes
                ),
                projection037_first_active_time_from_mismatch_ms=(
                    None
                    if active.size == 0
                    else float(trace_times[int(active[0])] - mismatch_start_ms)
                ),
            )
        )
    return tuple(summaries)


def run_figure10_condition(
    *,
    top_down_current_pA: float,
    pre_match_duration_ms: float,
    mismatch_duration_ms: float,
    reset_pathway_enabled: bool,
    learned_weights: Mapping[str, tuple[float, ...] | np.ndarray] | None = None,
    persistent_projection_weight_scales: Mapping[str, float] | None = None,
    comparator_top_k_targets: int | None = None,
    comparator_source_index: int = 40,
    top_down_current_mode: TopDownCurrentMode | str = (TopDownCurrentMode.SUSTAINED_EPOCH),
    record_layer6i_diagnostics: bool = False,
    record_layer6i_trace_indices: tuple[int, ...] = (),
    layer6i_replay_trace_output: str | Path | None = None,
    layer6i_replay_cell_index: int = 0,
    record_reset_chain_diagnostics: bool = False,
    record_layer4_balance_diagnostics: bool = False,
    record_layer4_target_balance_indices: tuple[int, ...] = (),
    record_layer4_target_timing_indices: tuple[int, ...] = (),
    layer4_target_timing_window_ms: tuple[float, float] = (65.0, 85.0),
    layer4_target_timing_bin_width_ms: float = 1.0,
    layer4_target_timing_current_threshold_pA: float = 1e-9,
    conventions=None,
    dt_ms: float = 0.01,
    cpp_standalone_directory: str | Path | None = None,
    brian=None,
) -> Figure10ConditionResult:
    """Run one persistent match-to-mismatch episode on the classic sector.

    Durations and category current are required because the paper does not
    report unique numeric values for this Figure 10 sequence.
    """

    if pre_match_duration_ms <= 0 or mismatch_duration_ms <= 0 or dt_ms <= 0:
        raise ValueError("Figure 10 durations and dt_ms must be positive")
    if top_down_current_pA <= 0:
        raise ValueError("top_down_current_pA must be positive")
    if not isinstance(record_layer6i_diagnostics, bool):
        raise TypeError("layer-6I diagnostics flag must be boolean")
    if not isinstance(record_layer6i_trace_indices, tuple) or any(
        isinstance(index, bool) or not isinstance(index, int)
        for index in record_layer6i_trace_indices
    ):
        raise TypeError("layer-6I trace indices must be a tuple of integers")
    if len(set(record_layer6i_trace_indices)) != len(record_layer6i_trace_indices):
        raise ValueError("layer-6I trace indices must be unique")
    if any(index < 0 or index >= 81 for index in record_layer6i_trace_indices):
        raise ValueError("layer-6I trace index must be between 0 and 80")
    if record_layer6i_trace_indices and not record_layer6i_diagnostics:
        raise ValueError("selected layer-6I traces require layer-6I diagnostics")
    if isinstance(layer6i_replay_cell_index, bool) or not isinstance(
        layer6i_replay_cell_index, int
    ):
        raise TypeError("layer-6I replay cell index must be an integer")
    if layer6i_replay_cell_index < 0 or layer6i_replay_cell_index >= 81:
        raise ValueError("layer-6I replay cell index must be between 0 and 80")
    if layer6i_replay_trace_output is not None:
        replay_output = Path(layer6i_replay_trace_output)
        if replay_output.exists():
            raise FileExistsError(replay_output)
        if not replay_output.parent.is_dir():
            raise ValueError("layer-6I replay trace parent directory does not exist")
    if not isinstance(record_reset_chain_diagnostics, bool):
        raise TypeError("reset-chain diagnostics flag must be boolean")
    if not isinstance(record_layer4_balance_diagnostics, bool):
        raise TypeError("layer-4 balance diagnostics flag must be boolean")
    if record_layer4_balance_diagnostics and not record_reset_chain_diagnostics:
        raise ValueError("layer-4 balance diagnostics require reset-chain diagnostics")
    if not isinstance(record_layer4_target_balance_indices, tuple) or any(
        isinstance(index, bool) or not isinstance(index, int)
        for index in record_layer4_target_balance_indices
    ):
        raise TypeError("layer-4 target balance indices must be a tuple of integers")
    if len(set(record_layer4_target_balance_indices)) != len(
        record_layer4_target_balance_indices
    ):
        raise ValueError("layer-4 target balance indices must be unique")
    if any(index < 0 or index >= 81 for index in record_layer4_target_balance_indices):
        raise ValueError("layer-4 target balance index must be between 0 and 80")
    if record_layer4_target_balance_indices and not record_layer4_balance_diagnostics:
        raise ValueError("layer-4 target balance requires layer-4 balance diagnostics")
    if not isinstance(record_layer4_target_timing_indices, tuple) or any(
        isinstance(index, bool) or not isinstance(index, int)
        for index in record_layer4_target_timing_indices
    ):
        raise TypeError("layer-4 target timing indices must be a tuple of integers")
    if len(set(record_layer4_target_timing_indices)) != len(
        record_layer4_target_timing_indices
    ):
        raise ValueError("layer-4 target timing indices must be unique")
    if any(index < 0 or index >= 81 for index in record_layer4_target_timing_indices):
        raise ValueError("layer-4 target timing index must be between 0 and 80")
    if record_layer4_target_timing_indices and not record_layer4_balance_diagnostics:
        raise ValueError("layer-4 target timing requires layer-4 balance diagnostics")
    if record_layer4_target_timing_indices and (
        not isinstance(layer4_target_timing_window_ms, tuple)
        or len(layer4_target_timing_window_ms) != 2
        or layer4_target_timing_window_ms[0] < 0
        or layer4_target_timing_window_ms[1] <= layer4_target_timing_window_ms[0]
        or layer4_target_timing_window_ms[1] > mismatch_duration_ms
    ):
        raise ValueError("layer-4 target timing window must lie within mismatch")
    if layer4_target_timing_bin_width_ms <= 0:
        raise ValueError("layer-4 target timing bin width must be positive")
    if layer4_target_timing_current_threshold_pA < 0:
        raise ValueError("layer-4 target timing current threshold cannot be negative")
    current_mode = TopDownCurrentMode(top_down_current_mode)
    if current_mode is TopDownCurrentMode.UNTIL_CUED_CELL_EVENT_LIMIT:
        raise ValueError("Figure 10 does not define an event-count-limited cue")
    if comparator_top_k_targets is not None and learned_weights is None:
        raise ValueError("the reconstructed comparator requires learned weights")
    if brian is None:
        import brian2 as brian
    if cpp_standalone_directory is not None:
        brian.device.reinit()
        brian.set_device(
            "cpp_standalone",
            directory=str(Path(cpp_standalone_directory).resolve()),
            build_on_run=False,
        )
    from ..classic_sector import build_first_order_connected_sector, figure6_runtime_conventions

    conventions = conventions or figure6_runtime_conventions()
    brian.start_scope()
    brian.defaultclock.dt = dt_ms * brian.ms
    sector = build_first_order_connected_sector(conventions=conventions, brian=brian)
    scales = persistent_projection_weight_scales or {}
    unknown_scales = set(scales) - set(sector.projections)
    if unknown_scales:
        raise ValueError(f"unknown projection scale IDs: {sorted(unknown_scales)}")
    for projection_id, scale in scales.items():
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("projection weight scales must be finite and positive")
        projection = sector.projections[projection_id]
        for block in getattr(projection, "blocks", (projection,)):
            block.w = f"w*({float(scale)!r})"

    if learned_weights is None:
        learned = paper_constrained_figure6_expectation(
            sector.projections, derive_from_source=cpp_standalone_directory is not None
        )
        learned_state_provenance = "paper-constrained-figure6c-reference"
    else:
        learned = learned_weights
        learned_state_provenance = "simulated-learned-weight-snapshot"
    apply_figure7_learned_state(
        sector.projections,
        learned,
        verify_runtime_bounds=cpp_standalone_directory is None,
    )
    relay_input_gains = (
        None
        if comparator_top_k_targets is None
        else top_k_comparator_relay_input_gains(
            learned,
            target_count=comparator_top_k_targets,
            source_index=comparator_source_index,
        )
    )
    layer4 = brian.SpikeMonitor(sector.populations["layer4_excitatory_v1"].group)
    nonspecific = brian.SpikeMonitor(sector.populations["thalamic_nonspecific"].group)
    layer5 = brian.SpikeMonitor(sector.populations["layer5_excitatory_v1"].group)
    layer6i = brian.SpikeMonitor(sector.populations["layer6i_excitatory_v1"].group)
    layer4_inhibitory = None
    if record_reset_chain_diagnostics:
        layer4_inhibitory = brian.SpikeMonitor(
            sector.populations["layer4_inhibitory_v1"].group,
            name=(
                "figure10_intact_layer4_inhibitory_spikes"
                if reset_pathway_enabled
                else "figure10_control_layer4_inhibitory_spikes"
            ),
        )
    sector.network.add(layer4, nonspecific, layer5, layer6i)
    if layer4_inhibitory is not None:
        sector.network.add(layer4_inhibitory)
    layer6i_state = None
    if record_layer6i_diagnostics:
        layer6i_state = brian.StateMonitor(
            sector.populations["layer6i_excitatory_v1"].group,
            (
                "port_000_gate",
                "port_001_gate",
                "port_002_gate",
                "i_port_000",
                "i_port_001",
                "i_port_002",
                "v_soma",
                "v_proximal_dendrite",
                "spike_detector_voltage",
            ),
            record=True,
            name=(
                "figure10_intact_layer6i_state"
                if reset_pathway_enabled
                else "figure10_control_layer6i_state"
            ),
        )
        sector.network.add(layer6i_state)
    layer6i_transmitter_state = None
    layer4i_state = None
    layer4e_inhibitory_state = None
    if record_reset_chain_diagnostics:
        layer6i_transmitter_state = brian.StateMonitor(
            sector.populations["layer6i_excitatory_v1"].group,
            "transmitter",
            record=True,
            when="thresholds",
            order=0,
            name=(
                "figure10_intact_layer6i_transmitter"
                if reset_pathway_enabled
                else "figure10_control_layer6i_transmitter"
            ),
        )
        layer4i_state = brian.StateMonitor(
            sector.populations["layer4_inhibitory_v1"].group,
            ("port_000_gate", "i_port_000"),
            record=True,
            name=(
                "figure10_intact_layer4i_projection026"
                if reset_pathway_enabled
                else "figure10_control_layer4i_projection026"
            ),
        )
        layer4e_variables = ["port_001_gate", "i_port_001"]
        if record_layer4_balance_diagnostics:
            layer4e_variables.extend(("port_003_gate", "i_port_003"))
        if record_layer4_target_balance_indices:
            layer4e_variables.extend(("i_port_000", "i_port_002"))
        if record_layer4_target_timing_indices:
            layer4e_variables.extend(
                ("port_000_gate", "i_port_000", "i_port_002", "v_soma")
            )
        layer4e_variables = list(dict.fromkeys(layer4e_variables))
        layer4e_inhibitory_state = brian.StateMonitor(
            sector.populations["layer4_excitatory_v1"].group,
            tuple(layer4e_variables),
            record=True,
            name=(
                "figure10_intact_layer4e_projection036"
                if reset_pathway_enabled
                else "figure10_control_layer4e_projection036"
            ),
        )
        sector.network.add(layer6i_transmitter_state, layer4i_state, layer4e_inhibitory_state)
    match = ClassicMatchMismatchCue(
        condition=MatchCondition.MATCH,
        top_down_current_pA=top_down_current_pA,
        duration_ms=pre_match_duration_ms,
    )
    if current_mode is TopDownCurrentMode.UNTIL_CUED_CELL_FIRST_EVENT:
        category_group = sector.populations[match.top_down_population].group
        category_group.clear_drive_on_spike = 0
        category_group.clear_drive_on_spike[match.top_down_cell_index] = 1
    apply_match_mismatch_cue(sector, match, relay_input_gains=relay_input_gains, brian=brian)
    sector.network.run(pre_match_duration_ms * brian.ms)
    clear_match_mismatch_cue(sector, match, brian=brian)
    layer6i_replay_state = None
    layer6i_replay_initial_state = None
    layer6i_replay_start_ms = None
    if layer6i_replay_trace_output is not None:
        from .layer6i_replay import (
            LAYER6I_REPLAY_MONITOR_VARIABLES,
            capture_layer6i_initial_state,
        )

        layer6i_group = sector.populations["layer6i_excitatory_v1"].group
        layer6i_replay_initial_state = capture_layer6i_initial_state(
            layer6i_group,
            cell_index=layer6i_replay_cell_index,
            brian=brian,
        )
        layer6i_replay_start_ms = float(sector.network.t / brian.ms)
        layer6i_replay_state = brian.StateMonitor(
            layer6i_group,
            LAYER6I_REPLAY_MONITOR_VARIABLES,
            record=[layer6i_replay_cell_index],
            when="thresholds",
            order=0,
            name=(
                "figure10_intact_layer6i_replay_state"
                if reset_pathway_enabled
                else "figure10_control_layer6i_replay_state"
            ),
        )
        sector.network.add(layer6i_replay_state)
    # Disconnect only at mismatch onset.  The intact and negative-control
    # conditions must establish exactly the same pre-reset winner.
    if not reset_pathway_enabled:
        for projection_id in FIGURE10_RESET_INPUT_PROJECTIONS:
            sector.projections[projection_id].w = 0

    mismatch = ClassicMatchMismatchCue(
        condition=MatchCondition.MISMATCH,
        top_down_current_pA=top_down_current_pA,
        duration_ms=mismatch_duration_ms,
    )
    if current_mode is TopDownCurrentMode.UNTIL_CUED_CELL_FIRST_EVENT:
        category_group = sector.populations[mismatch.top_down_population].group
        category_group.clear_drive_on_spike = 0
        category_group.clear_drive_on_spike[mismatch.top_down_cell_index] = 1
    apply_match_mismatch_cue(sector, mismatch, relay_input_gains=relay_input_gains, brian=brian)
    sector.network.run(mismatch_duration_ms * brian.ms)
    clear_match_mismatch_cue(sector, mismatch, brian=brian)
    if cpp_standalone_directory is not None:
        from ..standalone import build_and_run_cpp_standalone

        build_and_run_cpp_standalone(brian, cpp_standalone_directory)

    layer6i_replay_trace_sha256 = None
    if layer6i_replay_trace_output is not None:
        from .layer6i_replay import write_layer6i_replay_trace

        assert layer6i_replay_state is not None
        assert layer6i_replay_initial_state is not None
        assert layer6i_replay_start_ms is not None
        layer6i_replay_trace_sha256 = write_layer6i_replay_trace(
            layer6i_replay_state,
            layer6i,
            layer6i_replay_trace_output,
            initial_state=layer6i_replay_initial_state,
            cell_index=layer6i_replay_cell_index,
            mismatch_start_ms=layer6i_replay_start_ms,
            duration_ms=mismatch_duration_ms,
            dt_ms=dt_ms,
            condition=("intact" if reset_pathway_enabled else "disconnected_control"),
            fingerprint=conventions.fingerprint,
            brian=brian,
        )

    gate_integrals: tuple[tuple[str, float], ...] = ()
    current_integrals: tuple[tuple[str, float], ...] = ()
    current_peaks: tuple[tuple[str, float], ...] = ()
    projection025_gate_by_index: tuple[tuple[int, float], ...] = ()
    projection025_current_integral_by_index: tuple[tuple[int, float], ...] = ()
    projection025_current_peak_by_index: tuple[tuple[int, float], ...] = ()
    soma_voltage_peak_by_index: tuple[tuple[int, float], ...] = ()
    proximal_voltage_peak_by_index: tuple[tuple[int, float], ...] = ()
    selected_trace_summaries: tuple[Figure10Layer6iTraceSummary, ...] = ()
    if layer6i_state is not None:
        times_ms = np.asarray(layer6i_state.t / brian.ms)
        mismatch_window = times_ms >= pre_match_duration_ms
        mismatch_times_ms = times_ms[mismatch_window]
        projection_ports = (
            ("modeldb112923.projection.023", "port_000"),
            ("modeldb112923.projection.024", "port_001"),
            ("modeldb112923.projection.025", "port_002"),
        )
        gate_integrals = tuple(
            (
                projection_id,
                float(
                    np.trapz(
                        np.sum(
                            np.asarray(getattr(layer6i_state, f"{port}_gate"))[:, mismatch_window],
                            axis=0,
                        ),
                        mismatch_times_ms,
                    )
                ),
            )
            for projection_id, port in projection_ports
        )
        current_traces = {
            projection_id: np.sum(
                np.asarray(getattr(layer6i_state, f"i_{port}") / brian.pA)[:, mismatch_window],
                axis=0,
            )
            for projection_id, port in projection_ports
        }
        current_integrals = tuple(
            (projection_id, float(np.trapz(trace, mismatch_times_ms)))
            for projection_id, trace in current_traces.items()
        )
        current_peaks = tuple(
            (projection_id, float(np.max(trace))) for projection_id, trace in current_traces.items()
        )
        projection025_gate = np.asarray(layer6i_state.port_002_gate)[:, mismatch_window]
        projection025_current_pA = np.asarray(layer6i_state.i_port_002 / brian.pA)[
            :, mismatch_window
        ]
        projection025_gate_by_index = tuple(
            (index, float(np.trapz(trace, mismatch_times_ms)))
            for index, trace in enumerate(projection025_gate)
        )
        projection025_current_integral_by_index = tuple(
            (index, float(np.trapz(trace, mismatch_times_ms)))
            for index, trace in enumerate(projection025_current_pA)
        )
        projection025_current_peak_by_index = tuple(
            (index, float(np.max(trace))) for index, trace in enumerate(projection025_current_pA)
        )
        soma_voltage_peak_by_index = tuple(
            (index, float(np.max(trace)))
            for index, trace in enumerate(
                np.asarray(layer6i_state.v_soma / brian.mV)[:, mismatch_window]
            )
        )
        proximal_voltage_peak_by_index = tuple(
            (index, float(np.max(trace)))
            for index, trace in enumerate(
                np.asarray(layer6i_state.v_proximal_dendrite / brian.mV)[:, mismatch_window]
            )
        )
        if record_layer6i_trace_indices:
            selected_trace_summaries = summarize_layer6i_selected_traces(
                selected_indices=record_layer6i_trace_indices,
                times_ms=times_ms,
                mismatch_start_ms=pre_match_duration_ms,
                spike_detector_threshold_mV=conventions.spike_event_threshold_mV,
                soma_voltage_mV=np.asarray(layer6i_state.v_soma / brian.mV),
                proximal_voltage_mV=np.asarray(layer6i_state.v_proximal_dendrite / brian.mV),
                spike_detector_voltage_mV=np.asarray(
                    layer6i_state.spike_detector_voltage / brian.mV
                ),
                gate_by_projection={
                    projection_id: np.asarray(getattr(layer6i_state, f"{port}_gate"))
                    for projection_id, port in projection_ports
                },
                current_pA_by_projection={
                    projection_id: np.asarray(getattr(layer6i_state, f"i_{port}") / brian.pA)
                    for projection_id, port in projection_ports
                },
                spike_indices=np.asarray(layer6i.i),
                spike_times_ms=np.asarray(layer6i.t / brian.ms),
            )

    transmitter_samples: tuple[tuple[int, float, float], ...] = ()
    if layer6i_transmitter_state is not None:
        transmitter_times_ms = np.asarray(layer6i_transmitter_state.t / brian.ms)
        transmitter_values = np.asarray(layer6i_transmitter_state.transmitter)
        transmitter_samples = tuple(
            (
                int(index),
                float(time_ms),
                float(
                    transmitter_values[
                        int(index),
                        int(np.argmin(np.abs(transmitter_times_ms - float(time_ms)))),
                    ]
                ),
            )
            for index, time_ms in zip(layer6i.i, layer6i.t / brian.ms, strict=True)
            if float(time_ms) >= pre_match_duration_ms
        )

    def _single_projection_summary(state, port):
        if state is None:
            return None, None, None
        state_times_ms = np.asarray(state.t / brian.ms)
        window = state_times_ms >= pre_match_duration_ms
        mismatch_times_ms = state_times_ms[window]
        gate_trace = np.sum(np.asarray(getattr(state, f"{port}_gate"))[:, window], axis=0)
        current_trace = np.sum(
            np.asarray(getattr(state, f"i_{port}") / brian.pA)[:, window], axis=0
        )
        return (
            float(np.trapz(gate_trace, mismatch_times_ms)),
            float(np.trapz(current_trace, mismatch_times_ms)),
            current_trace,
        )

    projection026_gate, projection026_current, projection026_trace = _single_projection_summary(
        layer4i_state, "port_000"
    )
    projection036_gate, projection036_current, projection036_trace = _single_projection_summary(
        layer4e_inhibitory_state, "port_001"
    )
    projection038_gate = projection038_current = projection038_trace = None
    layer4_balance_bins: tuple[Figure10Layer4BalanceBin, ...] = ()
    layer4_target_balance_bins: tuple[Figure10Layer4TargetBalanceBin, ...] = ()
    layer4_target_timing_summaries: tuple[Figure10Layer4TargetTimingSummary, ...] = ()
    if record_layer4_balance_diagnostics:
        projection038_gate, projection038_current, projection038_trace = (
            _single_projection_summary(layer4e_inhibitory_state, "port_003")
        )
        assert layer4i_state is not None
        assert layer4e_inhibitory_state is not None
        assert layer4_inhibitory is not None
        assert projection026_trace is not None
        assert projection036_trace is not None
        assert projection038_trace is not None
        pre_layer4_indices = np.asarray(layer4.i, dtype=int)[
            np.asarray(layer4.t / brian.ms, dtype=float) < pre_match_duration_ms
        ]
        winner_indices = tuple(int(value) for value in np.unique(pre_layer4_indices))
        layer4i_times_ms = np.asarray(layer4i_state.t / brian.ms)
        layer4_balance_bins = summarize_layer4_balance_bins(
            mismatch_start_ms=pre_match_duration_ms,
            mismatch_duration_ms=mismatch_duration_ms,
            dt_ms=dt_ms,
            bin_width_ms=10.0,
            state_times_ms=layer4i_times_ms[
                layer4i_times_ms >= pre_match_duration_ms
            ],
            projection026_current_pA=projection026_trace,
            projection036_current_pA=projection036_trace,
            projection038_current_pA=projection038_trace,
            layer4_inhibitory_spike_times_ms=np.asarray(
                layer4_inhibitory.t / brian.ms
            ),
            layer4_spike_indices=np.asarray(layer4.i),
            layer4_spike_times_ms=np.asarray(layer4.t / brian.ms),
            winner_indices=winner_indices,
        )
        if record_layer4_target_balance_indices:
            state_times_ms = np.asarray(layer4e_inhibitory_state.t / brian.ms)
            mismatch_window = state_times_ms >= pre_match_duration_ms
            layer4_target_balance_bins = summarize_layer4_target_balance_bins(
                target_indices=record_layer4_target_balance_indices,
                mismatch_start_ms=pre_match_duration_ms,
                mismatch_duration_ms=mismatch_duration_ms,
                dt_ms=dt_ms,
                bin_width_ms=10.0,
                state_times_ms=state_times_ms[mismatch_window],
                projection035_current_pA_by_index=np.asarray(
                    layer4e_inhibitory_state.i_port_000 / brian.pA
                )[:, mismatch_window],
                projection036_current_pA_by_index=np.asarray(
                    layer4e_inhibitory_state.i_port_001 / brian.pA
                )[:, mismatch_window],
                projection037_current_pA_by_index=np.asarray(
                    layer4e_inhibitory_state.i_port_002 / brian.pA
                )[:, mismatch_window],
                projection038_current_pA_by_index=np.asarray(
                    layer4e_inhibitory_state.i_port_003 / brian.pA
                )[:, mismatch_window],
                layer4_spike_indices=np.asarray(layer4.i),
                layer4_spike_times_ms=np.asarray(layer4.t / brian.ms),
            )
        if record_layer4_target_timing_indices:
            state_times_ms = np.asarray(layer4e_inhibitory_state.t / brian.ms)
            layer4_target_timing_summaries = summarize_layer4_target_timing(
                target_indices=record_layer4_target_timing_indices,
                mismatch_start_ms=pre_match_duration_ms,
                window_start_from_mismatch_ms=float(layer4_target_timing_window_ms[0]),
                window_end_from_mismatch_ms=float(layer4_target_timing_window_ms[1]),
                dt_ms=dt_ms,
                bin_width_ms=layer4_target_timing_bin_width_ms,
                current_activity_threshold_pA=(
                    layer4_target_timing_current_threshold_pA
                ),
                state_times_ms=state_times_ms,
                soma_voltage_mV_by_index=np.asarray(
                    layer4e_inhibitory_state.v_soma / brian.mV
                ),
                projection035_current_pA_by_index=np.asarray(
                    layer4e_inhibitory_state.i_port_000 / brian.pA
                ),
                projection036_current_pA_by_index=np.asarray(
                    layer4e_inhibitory_state.i_port_001 / brian.pA
                ),
                projection037_current_pA_by_index=np.asarray(
                    layer4e_inhibitory_state.i_port_002 / brian.pA
                ),
                projection038_current_pA_by_index=np.asarray(
                    layer4e_inhibitory_state.i_port_003 / brian.pA
                ),
                projection035_gate_by_index=np.asarray(
                    layer4e_inhibitory_state.port_000_gate
                ),
                projection036_gate_by_index=np.asarray(
                    layer4e_inhibitory_state.port_001_gate
                ),
                layer4_spike_indices=np.asarray(layer4.i),
                layer4_spike_times_ms=np.asarray(layer4.t / brian.ms),
            )

    return Figure10ConditionResult(
        pre_match_duration_ms=pre_match_duration_ms,
        mismatch_duration_ms=mismatch_duration_ms,
        reset_pathway_enabled=reset_pathway_enabled,
        layer4_spike_indices=tuple(int(value) for value in np.asarray(layer4.i)),
        layer4_spike_times_ms=tuple(float(value) for value in np.asarray(layer4.t / brian.ms)),
        nonspecific_spike_times_ms=tuple(
            float(value) for value in np.asarray(nonspecific.t / brian.ms)
        ),
        layer5_spike_indices=tuple(int(value) for value in np.asarray(layer5.i)),
        layer5_spike_times_ms=tuple(float(value) for value in np.asarray(layer5.t / brian.ms)),
        layer6i_spike_indices=tuple(int(value) for value in np.asarray(layer6i.i)),
        layer6i_spike_times_ms=tuple(float(value) for value in np.asarray(layer6i.t / brian.ms)),
        layer4_inhibitory_spike_indices=(
            ()
            if layer4_inhibitory is None
            else tuple(int(value) for value in np.asarray(layer4_inhibitory.i))
        ),
        layer4_inhibitory_spike_times_ms=(
            ()
            if layer4_inhibitory is None
            else tuple(float(value) for value in np.asarray(layer4_inhibitory.t / brian.ms))
        ),
        convention_fingerprint=conventions.fingerprint,
        learned_state_provenance=learned_state_provenance,
        comparator_target_count=comparator_top_k_targets,
        top_down_current_mode=current_mode.value,
        layer6i_mismatch_gate_integral_ms_by_projection=gate_integrals,
        layer6i_mismatch_current_integral_pA_ms_by_projection=current_integrals,
        layer6i_mismatch_current_peak_pA_by_projection=current_peaks,
        layer6i_mismatch_event_transmitter_samples=transmitter_samples,
        layer6i_mismatch_projection025_gate_integral_ms_by_index=(projection025_gate_by_index),
        layer6i_mismatch_projection025_current_integral_pA_ms_by_index=(
            projection025_current_integral_by_index
        ),
        layer6i_mismatch_projection025_current_peak_pA_by_index=(
            projection025_current_peak_by_index
        ),
        layer6i_mismatch_soma_voltage_peak_mV_by_index=soma_voltage_peak_by_index,
        layer6i_mismatch_proximal_voltage_peak_mV_by_index=(proximal_voltage_peak_by_index),
        layer6i_selected_trace_summaries=selected_trace_summaries,
        layer6i_replay_trace_path=(
            None if layer6i_replay_trace_output is None else str(layer6i_replay_trace_output)
        ),
        layer6i_replay_trace_sha256=layer6i_replay_trace_sha256,
        layer4i_mismatch_projection026_gate_integral_ms=projection026_gate,
        layer4i_mismatch_projection026_current_integral_pA_ms=projection026_current,
        layer4i_mismatch_projection026_current_peak_pA=(
            None if projection026_trace is None else float(np.max(projection026_trace))
        ),
        layer4e_mismatch_projection036_gate_integral_ms=projection036_gate,
        layer4e_mismatch_projection036_current_integral_pA_ms=projection036_current,
        layer4e_mismatch_projection036_current_trough_pA=(
            None if projection036_trace is None else float(np.min(projection036_trace))
        ),
        layer4e_mismatch_projection038_gate_integral_ms=projection038_gate,
        layer4e_mismatch_projection038_current_integral_pA_ms=projection038_current,
        layer4e_mismatch_projection038_current_peak_pA=(
            None if projection038_trace is None else float(np.max(projection038_trace))
        ),
        layer4_balance_bins=layer4_balance_bins,
        layer4_target_balance_bins=layer4_target_balance_bins,
        layer4_target_timing_summaries=layer4_target_timing_summaries,
    )
