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
from ..protocols import (
    BarOrientation,
    ClassicBarStimulus,
    apply_bar_stimulus,
    clear_bar_stimulus,
)

BOTTOM_UP_PROJECTION_ID = "modeldb112923.projection.035"
TOP_DOWN_WIDE_PROJECTION_ID = "modeldb112923.projection.005"
TOP_DOWN_NARROW_PROJECTION_ID = "modeldb112923.projection.007"
MINIMUM_TOP_DOWN_HORIZONTAL_CONTRAST = 0.01
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

    @property
    def horizontal_orientation_contrast(self) -> float:
        """Absolute horizontal-minus-vertical arm contrast, excluding the center.

        Ratio-based retention becomes ill-conditioned in the tiny Gaussian
        tails.  The published maps are spatial weight maps, so their direct
        arm contrast is the robust acceptance metric.
        """

        after = np.asarray(self.after)
        horizontal = float(np.mean(after[list(HORIZONTAL_ONLY_INDICES)]))
        vertical = float(np.mean(after[list(VERTICAL_ONLY_INDICES)]))
        return horizontal - vertical


@dataclass(frozen=True, slots=True)
class Figure6LearningResult:
    convention_fingerprint: str
    duration_ms: float
    population_spikes: dict[str, int]
    bottom_up: Figure6MapSummary
    top_down_wide: Figure6MapSummary
    top_down_narrow: Figure6MapSummary
    population_spike_indices: dict[str, tuple[int, ...]] | None = None
    population_spike_times_ms: dict[str, tuple[float, ...]] | None = None

    @property
    def top_down_combined(self) -> Figure6MapSummary:
        """Combined adaptive AMPA field plotted in Figure 6c.

        The archived relay has separate wide and narrow adaptive AMPA gates,
        while the paper shows one corticothalamic weight field.  Their summed
        before/after maps also match the figure's stated graphical scale.
        """

        return Figure6MapSummary(
            projection_id=f"{self.top_down_wide.projection_id}+{self.top_down_narrow.projection_id}",
            map_role="combined_outgoing_from_active_layer6ii",
            before=tuple(
                wide + narrow
                for wide, narrow in zip(
                    self.top_down_wide.before, self.top_down_narrow.before, strict=True
                )
            ),
            after=tuple(
                wide + narrow
                for wide, narrow in zip(
                    self.top_down_wide.after, self.top_down_narrow.after, strict=True
                )
            ),
        )

    @property
    def bottom_up_oriented(self) -> bool:
        return self.bottom_up.horizontal_orientation_contrast > 0

    @property
    def top_down_oriented(self) -> bool:
        return (
            self.top_down_combined.horizontal_orientation_contrast
            >= MINIMUM_TOP_DOWN_HORIZONTAL_CONTRAST
        )


@dataclass(frozen=True, slots=True)
class Figure6LearningProtocol:
    """The Figure 6b/c simultaneous 100-ms horizontal training episode."""

    warmup_ms: float = 0.0
    stimulus_ms: float = 100.0
    post_stimulus_ms: float = 0.0
    dt_ms: float = 0.01
    source_value: float = 120.0
    category_source_value: float = 70.0
    winning_layer4_index: int = 40
    active_category_index: int = 40
    layer6ii_ahp_scale: float = 1.0
    monitored_populations: tuple[str, ...] = (
        "thalamic_relay",
        "layer4_excitatory_v1",
        "layer6ii_excitatory_v1",
    )

    def __post_init__(self) -> None:
        if self.warmup_ms < 0:
            raise ValueError("warmup_ms cannot be negative")
        if self.stimulus_ms <= 0 or self.dt_ms <= 0:
            raise ValueError("stimulus_ms and dt_ms must be positive")
        if self.post_stimulus_ms < 0:
            raise ValueError("post_stimulus_ms cannot be negative")
        if not 0 <= self.winning_layer4_index < 81:
            raise ValueError("winning_layer4_index must address the 9x9 sheet")
        if not 0 <= self.active_category_index < 81:
            raise ValueError("active_category_index must address the 9x9 sheet")
        if self.layer6ii_ahp_scale < 0:
            raise ValueError("layer6ii_ahp_scale cannot be negative")


@dataclass(frozen=True, slots=True)
class Figure6LearningRun:
    result: Figure6LearningResult
    learned_weights: dict[str, tuple[float, ...]]


@dataclass(frozen=True, slots=True)
class Figure6TopDownTimingAssessment:
    category_spike_ms: float | None
    teaching_arrival_ms: float | None
    preceding_relay_spike_ms: float | None
    following_relay_spike_ms: float | None

    @property
    def preceding_post_minus_arrival_ms(self) -> float | None:
        if self.preceding_relay_spike_ms is None or self.teaching_arrival_ms is None:
            return None
        return self.preceding_relay_spike_ms - self.teaching_arrival_ms

    @property
    def following_post_minus_arrival_ms(self) -> float | None:
        if self.following_relay_spike_ms is None or self.teaching_arrival_ms is None:
            return None
        return self.following_relay_spike_ms - self.teaching_arrival_ms

    @property
    def causal_pair_in_learning_window(self) -> bool:
        delta = self.following_post_minus_arrival_ms
        return delta is not None and 0 <= delta <= 20


@dataclass(frozen=True, slots=True)
class Figure6LearningReachability:
    """Source-bounded check that a reported map is reachable in one episode."""

    initial_weight: float
    maximum_weight: float
    learning_rate_per_ms: float
    episode_ms: float
    observed_weight: float
    upper_bound: float

    @property
    def reachable(self) -> bool:
        return self.observed_weight <= self.upper_bound + 1e-12


def figure6_weight_reachability(
    *,
    initial_weight: float,
    maximum_weight: float,
    learning_rate_per_ms: float,
    episode_ms: float,
    observed_weight: float,
) -> Figure6LearningReachability:
    """Return a generous Equation-25 upper bound independent of spike timing.

    For presynaptically gated learning, both the ligand gate and its extra
    gating factor lie in [0, 1]. Dropping all decay/depression and setting both
    to one gives ``dw/dt <= lambda * (w_max - w)``. Its solution is therefore
    an absolute upper bound for every possible spike train in the episode.
    """

    if not 0 <= initial_weight <= maximum_weight:
        raise ValueError("initial weight must lie within [0, maximum]")
    if learning_rate_per_ms < 0 or episode_ms < 0 or observed_weight < 0:
        raise ValueError("rate, duration, and observed weight must be nonnegative")
    upper = maximum_weight - (maximum_weight - initial_weight) * np.exp(
        -learning_rate_per_ms * episode_ms
    )
    return Figure6LearningReachability(
        initial_weight=initial_weight,
        maximum_weight=maximum_weight,
        learning_rate_per_ms=learning_rate_per_ms,
        episode_ms=episode_ms,
        observed_weight=observed_weight,
        upper_bound=float(upper),
    )


def assess_figure6_top_down_timing(
    result: Figure6LearningResult,
    *,
    category_index: int = 40,
    relay_index: int = 40,
    axonal_delay_ms: float = 2.0,
) -> Figure6TopDownTimingAssessment:
    """Score whether layer-6II teaching arrives before a matched relay spike."""

    indices = result.population_spike_indices or {}
    times = result.population_spike_times_ms or {}
    category_times = tuple(
        time
        for index, time in zip(
            indices.get("layer6ii_excitatory_v1", ()),
            times.get("layer6ii_excitatory_v1", ()),
            strict=True,
        )
        if index == category_index
    )
    if not category_times:
        return Figure6TopDownTimingAssessment(None, None, None, None)
    category_spike = category_times[0]
    arrival = category_spike + axonal_delay_ms
    relay_times = tuple(
        time
        for index, time in zip(
            indices.get("thalamic_relay", ()),
            times.get("thalamic_relay", ()),
            strict=True,
        )
        if index == relay_index
    )
    preceding = tuple(time for time in relay_times if time < arrival)
    following = tuple(time for time in relay_times if time >= arrival)
    return Figure6TopDownTimingAssessment(
        category_spike_ms=category_spike,
        teaching_arrival_ms=arrival,
        preceding_relay_spike_ms=max(preceding) if preceding else None,
        following_relay_spike_ms=min(following) if following else None,
    )


def run_figure6_learning(
    *,
    conventions=None,
    protocol: Figure6LearningProtocol | None = None,
    brian=None,
) -> Figure6LearningRun:
    """Run and summarize the official simultaneous Figure 6b/c episode."""

    if brian is None:
        import brian2 as brian
    # Local imports avoid making lightweight analytical Figure 6 tests import
    # the full Brian2 sector assembly path.
    from ..classic_sector import build_first_order_connected_sector, figure6_runtime_conventions

    conventions = conventions or figure6_runtime_conventions()
    protocol = protocol or Figure6LearningProtocol()
    brian.start_scope()
    brian.defaultclock.dt = protocol.dt_ms * brian.ms
    sector = build_first_order_connected_sector(conventions=conventions, brian=brian)
    category_group = sector.populations["layer6ii_excitatory_v1"].group
    category_group.g_ahp_max = category_group.g_ahp_max[:] * protocol.layer6ii_ahp_scale
    monitored_ids = (
        BOTTOM_UP_PROJECTION_ID,
        TOP_DOWN_WIDE_PROJECTION_ID,
        TOP_DOWN_NARROW_PROJECTION_ID,
    )
    before = {
        projection_id: np.asarray(sector.projections[projection_id].w[:], dtype=float).copy()
        for projection_id in monitored_ids
    }
    monitors = {
        name: brian.SpikeMonitor(population.group, name=f"figure6_spikes_{name}")
        for name, population in sector.populations.items()
        if name in protocol.monitored_populations
    }
    unknown_monitors = set(protocol.monitored_populations) - set(monitors)
    if unknown_monitors:
        raise ValueError(f"unknown monitored populations: {sorted(unknown_monitors)}")
    sector.network.add(*monitors.values())
    if protocol.warmup_ms:
        sector.network.run(protocol.warmup_ms * brian.ms)
    warmup_counts = {name: int(monitor.num_spikes) for name, monitor in monitors.items()}
    stimulus = ClassicBarStimulus(
        BarOrientation.HORIZONTAL,
        duration_ms=protocol.stimulus_ms,
        source_value=protocol.source_value,
        category_source_value=protocol.category_source_value,
        include_archived_category_pixel=True,
    )
    apply_bar_stimulus(sector, stimulus)
    sector.network.run(protocol.stimulus_ms * brian.ms)
    if protocol.post_stimulus_ms:
        clear_bar_stimulus(sector, stimulus)
        sector.network.run(protocol.post_stimulus_ms * brian.ms)
    spike_counts = {
        name: int(monitor.num_spikes) - warmup_counts[name]
        for name, monitor in monitors.items()
    }
    spike_indices: dict[str, tuple[int, ...]] = {}
    spike_times_ms: dict[str, tuple[float, ...]] = {}
    for name, monitor in monitors.items():
        times = np.asarray(monitor.t / brian.ms)
        indices = np.asarray(monitor.i)
        mask = times >= protocol.warmup_ms
        spike_indices[name] = tuple(int(value) for value in indices[mask])
        spike_times_ms[name] = tuple(float(value - protocol.warmup_ms) for value in times[mask])
    result = summarize_figure6_learning(
        convention_fingerprint=conventions.fingerprint,
        duration_ms=protocol.stimulus_ms,
        population_spikes=spike_counts,
        projections=sector.projections,
        before_weights=before,
        winning_layer4_index=protocol.winning_layer4_index,
        active_category_index=protocol.active_category_index,
    )
    result = Figure6LearningResult(
        convention_fingerprint=result.convention_fingerprint,
        duration_ms=result.duration_ms,
        population_spikes=result.population_spikes,
        bottom_up=result.bottom_up,
        top_down_wide=result.top_down_wide,
        top_down_narrow=result.top_down_narrow,
        population_spike_indices=spike_indices,
        population_spike_times_ms=spike_times_ms,
    )
    learned = {
        projection_id: tuple(
            float(value) for value in np.asarray(sector.projections[projection_id].w[:])
        )
        for projection_id in monitored_ids
    }
    return Figure6LearningRun(result=result, learned_weights=learned)


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
