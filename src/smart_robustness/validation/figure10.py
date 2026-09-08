"""Qualitative Figure 10 mismatch-reset validation.

Grossberg and Versace (2008) do not tabulate a reset latency or probability.
This module therefore tests the published causal ordering and winner switch,
with an explicit pathway-disconnection negative control, rather than fitting an
unreported numerical trace.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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
class Figure10ConditionResult:
    """Spike evidence from one persistent pre-reset then mismatch episode."""

    pre_match_duration_ms: float
    mismatch_duration_ms: float
    reset_pathway_enabled: bool
    layer4_spike_indices: tuple[int, ...]
    layer4_spike_times_ms: tuple[float, ...]
    nonspecific_spike_times_ms: tuple[float, ...] = ()
    layer5_spike_times_ms: tuple[float, ...] = ()
    layer6i_spike_times_ms: tuple[float, ...] = ()
    convention_fingerprint: str | None = None
    learned_state_provenance: str | None = None
    comparator_target_count: int | None = None
    top_down_current_mode: str = TopDownCurrentMode.SUSTAINED_EPOCH.value
    layer6i_mismatch_gate_integral_ms_by_projection: tuple[tuple[str, float], ...] = ()
    layer6i_mismatch_current_integral_pA_ms_by_projection: tuple[tuple[str, float], ...] = ()
    layer6i_mismatch_current_peak_pA_by_projection: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.pre_match_duration_ms <= 0 or self.mismatch_duration_ms <= 0:
            raise ValueError("Figure 10 phase durations must be positive")
        if len(self.layer4_spike_indices) != len(self.layer4_spike_times_ms):
            raise ValueError("layer-4 spike indices and times must have equal length")

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
    sector.network.add(layer4, nonspecific, layer5, layer6i)
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
            ),
            record=True,
            name=(
                "figure10_intact_layer6i_state"
                if reset_pathway_enabled
                else "figure10_control_layer6i_state"
            ),
        )
        sector.network.add(layer6i_state)
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

    gate_integrals: tuple[tuple[str, float], ...] = ()
    current_integrals: tuple[tuple[str, float], ...] = ()
    current_peaks: tuple[tuple[str, float], ...] = ()
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

    return Figure10ConditionResult(
        pre_match_duration_ms=pre_match_duration_ms,
        mismatch_duration_ms=mismatch_duration_ms,
        reset_pathway_enabled=reset_pathway_enabled,
        layer4_spike_indices=tuple(int(value) for value in np.asarray(layer4.i)),
        layer4_spike_times_ms=tuple(float(value) for value in np.asarray(layer4.t / brian.ms)),
        nonspecific_spike_times_ms=tuple(
            float(value) for value in np.asarray(nonspecific.t / brian.ms)
        ),
        layer5_spike_times_ms=tuple(float(value) for value in np.asarray(layer5.t / brian.ms)),
        layer6i_spike_times_ms=tuple(float(value) for value in np.asarray(layer6i.t / brian.ms)),
        convention_fingerprint=conventions.fingerprint,
        learned_state_provenance=learned_state_provenance,
        comparator_target_count=comparator_top_k_targets,
        top_down_current_mode=current_mode.value,
        layer6i_mismatch_gate_integral_ms_by_projection=gate_integrals,
        layer6i_mismatch_current_integral_pA_ms_by_projection=current_integrals,
        layer6i_mismatch_current_peak_pA_by_projection=current_peaks,
    )
