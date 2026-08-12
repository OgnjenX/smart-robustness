"""Qualitative Figure 10 mismatch-reset validation.

Grossberg and Versace (2008) do not tabulate a reset latency or probability.
This module therefore tests the published causal ordering and winner switch,
with an explicit pathway-disconnection negative control, rather than fitting an
unreported numerical trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..protocols import (
    ClassicMatchMismatchCue,
    MatchCondition,
    apply_match_mismatch_cue,
    clear_match_mismatch_cue,
)
from .figure7 import apply_figure7_learned_state, paper_constrained_figure6_expectation

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
        return self.pre_reset_winner_index is not None and self.pre_reset_winner_spikes > 0

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
    winner: int | None = None
    winner_spikes = int(np.max(intact_pre))
    if winner_spikes > 0:
        winner = int(np.argmax(intact_pre))
    intact_post = intact.layer4_counts(after_mismatch=True)
    control_post = disconnected_control.layer4_counts(after_mismatch=True)
    if winner is None:
        intact_winner = control_winner = 0
        intact_alternatives = control_alternatives = 0
    else:
        intact_winner = int(intact_post[winner])
        control_winner = int(control_post[winner])
        alternative_mask = np.arange(81) != winner
        intact_alternatives = int(np.count_nonzero(intact_post[alternative_mask]))
        control_alternatives = int(np.count_nonzero(control_post[alternative_mask]))
    return Figure10ResetAssessment(
        pre_reset_winner_index=winner,
        pre_reset_winner_spikes=winner_spikes,
        intact_winner_post_spikes=intact_winner,
        control_winner_post_spikes=control_winner,
        intact_released_alternatives=intact_alternatives,
        control_released_alternatives=control_alternatives,
        intact_nonspecific_spikes=intact.mismatch_spike_count(
            intact.nonspecific_spike_times_ms
        ),
        intact_layer5_spikes=intact.mismatch_spike_count(intact.layer5_spike_times_ms),
        intact_layer6i_spikes=intact.mismatch_spike_count(intact.layer6i_spike_times_ms),
    )


def run_figure10_condition(
    *,
    top_down_current_pA: float,
    pre_match_duration_ms: float,
    mismatch_duration_ms: float,
    reset_pathway_enabled: bool,
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
    learned = paper_constrained_figure6_expectation(
        sector.projections, derive_from_source=cpp_standalone_directory is not None
    )
    apply_figure7_learned_state(
        sector.projections,
        learned,
        verify_runtime_bounds=cpp_standalone_directory is None,
    )
    layer4 = brian.SpikeMonitor(sector.populations["layer4_excitatory_v1"].group)
    nonspecific = brian.SpikeMonitor(sector.populations["thalamic_nonspecific"].group)
    layer5 = brian.SpikeMonitor(sector.populations["layer5_excitatory_v1"].group)
    layer6i = brian.SpikeMonitor(sector.populations["layer6i_excitatory_v1"].group)
    sector.network.add(layer4, nonspecific, layer5, layer6i)

    match = ClassicMatchMismatchCue(
        condition=MatchCondition.MATCH,
        top_down_current_pA=top_down_current_pA,
        duration_ms=pre_match_duration_ms,
    )
    apply_match_mismatch_cue(sector, match, brian=brian)
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
    apply_match_mismatch_cue(sector, mismatch, brian=brian)
    sector.network.run(mismatch_duration_ms * brian.ms)
    clear_match_mismatch_cue(sector, mismatch, brian=brian)
    if cpp_standalone_directory is not None:
        from ..standalone import build_and_run_cpp_standalone

        build_and_run_cpp_standalone(brian, cpp_standalone_directory)

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
    )
