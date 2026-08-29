"""Reproducible Figure 6a SMART spike-timing learning curves."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from ..learning import (
    LEARNING_RULES,
    equilibrium_depression_scale,
    full_postsynaptic_learning_signal,
    gated_weight_derivative,
)
from ..modeldb_projections import MODELDB_FULL
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
L23_FEEDFORWARD_INHIBITION_ID = "modeldb112923.projection.031"
L23_FEEDFORWARD_EXCITATION_ID = "modeldb112923.projection.032"
L23_INTERNEURON_DRIVE_ID = "modeldb112923.projection.039"
MINIMUM_TOP_DOWN_HORIZONTAL_CONTRAST = 0.01
# Historical calibration treated Figure 6c's approximately 0.5--2.5 colorbar
# endpoint as a numeric map target. Artifact 155 proves that the raw matrix was
# not published and that the printed learning law cannot reach that endpoint
# from the stated baseline. Keep the gate for historical artifact comparison,
# but do not conflate it with the source-supported orientation/shape claim.
MINIMUM_TOP_DOWN_COMBINED_PEAK = 2.0
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
    def top_down_shape_oriented(self) -> bool:
        """Return the source-supported horizontal Figure 6c shape gate."""

        return (
            self.top_down_combined.horizontal_orientation_contrast
            >= MINIMUM_TOP_DOWN_HORIZONTAL_CONTRAST
        )

    @property
    def top_down_legacy_amplitude_gate_pass(self) -> bool:
        """Retain the retracted colorbar-endpoint gate for provenance."""

        return max(self.top_down_combined.after) >= MINIMUM_TOP_DOWN_COMBINED_PEAK

    @property
    def top_down_oriented(self) -> bool:
        """Historical combined shape-plus-amplitude gate.

        New source-strength assessments should use ``top_down_shape_oriented``
        and report absolute amplitude as not identifiable.
        """

        return self.top_down_shape_oriented and self.top_down_legacy_amplitude_gate_pass


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
class Figure6L23CurrentBalance:
    """Current and voltage landmarks at the first Figure 6 cortical bottleneck."""

    convention_fingerprint: str
    target_index: int
    network_finite: bool
    first_layer4_spike_ms: float | None
    first_layer23_interneuron_spike_ms: float | None
    first_layer23_excitatory_spike_ms: float | None
    excitation_gate_peak: float
    excitation_gate_peak_ms: float
    inhibition_gate_peak: float
    inhibition_gate_peak_ms: float
    excitation_current_peak_pA: float
    excitation_current_peak_ms: float
    inhibition_current_trough_pA: float
    inhibition_current_trough_ms: float
    proximal_voltage_peak_mV: float
    proximal_voltage_peak_ms: float
    soma_voltage_peak_mV: float
    soma_voltage_peak_ms: float
    soma_axial_current_peak_pA: float
    soma_axial_current_peak_ms: float


@dataclass(frozen=True, slots=True)
class Figure6RelayCurrentBalance:
    """Drive and intrinsic landmarks at the Figure 6 relay bottleneck."""

    convention_fingerprint: str
    network_mode: str
    disabled_projection_ids: tuple[str, ...]
    target_index: int
    network_finite: bool
    external_port_record_id: str
    external_effective_reversal_mV: float
    external_current_peak_pA: float
    external_current_peak_ms: float
    calcium_current_peak_pA: float
    calcium_current_peak_ms: float
    calcium_current_final_pA: float
    soma_voltage_peak_mV: float
    soma_voltage_peak_ms: float
    soma_voltage_post_event_minimum_mV: float
    soma_voltage_post_event_minimum_ms: float
    soma_voltage_final_mV: float
    soma_time_above_minus20_mV: float
    soma_time_above_0_mV: float
    soma_time_above_30_mV: float
    soma_time_above_leak_mV: float
    external_current_final_pA: float
    relay_event_times_ms: tuple[float, ...]
    target_layer4_event_times_ms: tuple[float, ...]

    @property
    def relay_repeats_during_stimulus(self) -> bool:
        return len(self.relay_event_times_ms) >= 2

    @property
    def relay_recruits_target_layer4(self) -> bool:
        return bool(self.target_layer4_event_times_ms)


@dataclass(frozen=True, slots=True)
class Figure6LearningPhaseWindow:
    """Exact learning decomposition between two presynaptic arrivals."""

    start_ms: float
    end_ms: float
    measured_delta: float
    positive_correlation_delta: float
    negative_correlation_delta: float
    baseline_delta: float


@dataclass(frozen=True, slots=True)
class Figure6LearningPhaseConnection:
    """Integrated Equation 25/28 terms for one corticothalamic connection."""

    projection_id: str
    target_index: int
    initial_weight: float
    final_weight: float
    measured_delta: float
    positive_correlation_delta: float
    negative_correlation_delta: float
    baseline_delta: float
    reconstructed_delta: float
    presynaptic_gate_integral_ms: float
    postsynaptic_positive_overlap_ms: float
    postsynaptic_negative_overlap_ms: float
    windows: tuple[Figure6LearningPhaseWindow, ...]


@dataclass(frozen=True, slots=True)
class Figure6LearningPhaseResult:
    """Connection-resolved top-down learning phase in the official episode."""

    convention_fingerprint: str
    dt_ms: float
    duration_ms: float
    category_event_times_ms: tuple[float, ...]
    relay_event_times_ms: dict[int, tuple[float, ...]]
    connections: tuple[Figure6LearningPhaseConnection, ...]


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
class Figure6CorticalRecruitmentAssessment:
    """First events along the paper's layer-4 to category pathway."""

    layer4_spike_ms: float | None
    layer23_spike_ms: float | None
    layer5_spike_ms: float | None
    layer6i_spike_ms: float | None
    layer6ii_spike_ms: float | None

    @property
    def feedforward_chain_complete(self) -> bool:
        """Whether the required layer-4 -> 2/3 -> 5 sequence is recruited."""

        sequence = (self.layer4_spike_ms, self.layer23_spike_ms, self.layer5_spike_ms)
        return all(value is not None for value in sequence) and bool(
            self.layer4_spike_ms < self.layer23_spike_ms < self.layer5_spike_ms
        )


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
    relay_times = tuple(
        time
        for index, time in zip(
            indices.get("thalamic_relay", ()),
            times.get("thalamic_relay", ()),
            strict=True,
        )
        if index == relay_index
    )
    causal_pairs = tuple(
        (relay_time - (category_time + axonal_delay_ms), category_time, relay_time)
        for category_time in category_times
        for relay_time in relay_times
        if relay_time >= category_time + axonal_delay_ms
    )
    if causal_pairs:
        _, category_spike, following_relay = min(causal_pairs)
    else:
        category_spike = category_times[0]
        following_relay = None
    arrival = category_spike + axonal_delay_ms
    preceding = tuple(time for time in relay_times if time < arrival)
    return Figure6TopDownTimingAssessment(
        category_spike_ms=category_spike,
        teaching_arrival_ms=arrival,
        preceding_relay_spike_ms=max(preceding) if preceding else None,
        following_relay_spike_ms=following_relay,
    )


def assess_figure6_cortical_recruitment(
    result: Figure6LearningResult,
) -> Figure6CorticalRecruitmentAssessment:
    """Report recruitment of the cortical chain described in Sections 2.1-2.2.

    Layer-6II also receives the archived blue external prime and layer-6I
    input, so its activity alone does not prove that layer 4 recruited the
    layer-2/3 -> layer-5 category pathway.
    """

    times = result.population_spike_times_ms or {}

    def first(population: str) -> float | None:
        values = times.get(population, ())
        return min(values) if values else None

    return Figure6CorticalRecruitmentAssessment(
        layer4_spike_ms=first("layer4_excitatory_v1"),
        layer23_spike_ms=first("layer23_excitatory_v1"),
        layer5_spike_ms=first("layer5_excitatory_v1"),
        layer6i_spike_ms=first("layer6i_excitatory_v1"),
        layer6ii_spike_ms=first("layer6ii_excitatory_v1"),
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


def run_figure6_l23_current_balance(
    *,
    conventions=None,
    protocol: Figure6LearningProtocol | None = None,
    target_index: int = 40,
    brian=None,
) -> Figure6L23CurrentBalance:
    """Trace the exact L4 excitation/feedforward-inhibition contest in layer 2/3.

    This diagnostic runs the published Figure 6 episode without modifying any
    projection. It records the two source-serialized currents on one layer-2/3
    pyramidal cell together with dendrite-to-soma axial transfer.
    """

    if brian is None:
        import brian2 as brian
    from ..classic_sector import build_first_order_connected_sector, figure6_runtime_conventions

    conventions = conventions or figure6_runtime_conventions()
    protocol = protocol or Figure6LearningProtocol()
    if not 0 <= target_index < 81:
        raise ValueError("target_index must address the 9x9 layer-2/3 sheet")
    brian.start_scope()
    brian.defaultclock.dt = protocol.dt_ms * brian.ms
    sector = build_first_order_connected_sector(conventions=conventions, brian=brian)
    target = sector.populations["layer23_excitatory_v1"]

    def port_name(record_id: str) -> str:
        return next(
            port.name for port in target.compiled.synaptic_ports if port.record_id == record_id
        )

    excitation_port = port_name(L23_FEEDFORWARD_EXCITATION_ID)
    inhibition_port = port_name(L23_FEEDFORWARD_INHIBITION_ID)
    variables = (
        "v_soma",
        "v_proximal_dendrite",
        "i_axial_inward_soma",
        f"{excitation_port}_gate",
        f"i_{excitation_port}",
        f"{inhibition_port}_gate",
        f"i_{inhibition_port}",
    )
    state = brian.StateMonitor(
        target.group,
        variables,
        record=[target_index],
        name="figure6_l23_current_balance_state",
    )
    spike_populations = (
        "layer4_excitatory_v1",
        "layer23_inhibitory_v1",
        "layer23_excitatory_v1",
    )
    spikes = {
        name: brian.SpikeMonitor(
            sector.populations[name].group,
            name=f"figure6_l23_current_balance_spikes_{name}",
        )
        for name in spike_populations
    }
    sector.network.add(state, *spikes.values())
    if protocol.warmup_ms:
        sector.network.run(protocol.warmup_ms * brian.ms)
    apply_bar_stimulus(
        sector,
        ClassicBarStimulus(
            BarOrientation.HORIZONTAL,
            duration_ms=protocol.stimulus_ms,
            source_value=protocol.source_value,
            category_source_value=protocol.category_source_value,
            include_archived_category_pixel=True,
        ),
    )
    sector.network.run(protocol.stimulus_ms * brian.ms)
    time_ms = np.asarray(state.t / brian.ms) - protocol.warmup_ms
    keep = time_ms >= 0
    time_ms = time_ms[keep]

    def trace(variable: str, unit) -> np.ndarray:
        return np.asarray(getattr(state, variable)[0][keep] / unit, dtype=float)

    def peak(values: np.ndarray, *, minimum: bool = False) -> tuple[float, float]:
        index = int(np.argmin(values) if minimum else np.argmax(values))
        return float(values[index]), float(time_ms[index])

    def first_spike(name: str) -> float | None:
        values = np.asarray(spikes[name].t / brian.ms) - protocol.warmup_ms
        values = values[values >= 0]
        return float(values[0]) if values.size else None

    excitation_gate_peak, excitation_gate_peak_ms = peak(trace(excitation_port + "_gate", 1))
    inhibition_gate_peak, inhibition_gate_peak_ms = peak(trace(inhibition_port + "_gate", 1))
    excitation_current_peak_pA, excitation_current_peak_ms = peak(
        trace("i_" + excitation_port, brian.pA)
    )
    inhibition_current_trough_pA, inhibition_current_trough_ms = peak(
        trace("i_" + inhibition_port, brian.pA), minimum=True
    )
    proximal_voltage_peak_mV, proximal_voltage_peak_ms = peak(
        trace("v_proximal_dendrite", brian.mV)
    )
    soma_voltage_peak_mV, soma_voltage_peak_ms = peak(trace("v_soma", brian.mV))
    soma_axial_current_peak_pA, soma_axial_current_peak_ms = peak(
        trace("i_axial_inward_soma", brian.pA)
    )
    network_finite = all(
        np.isfinite(np.asarray(getattr(population.group, f"v_{compartment}")[:] / brian.mV)).all()
        for population in sector.populations.values()
        for compartment in population.compartments
    )
    return Figure6L23CurrentBalance(
        convention_fingerprint=conventions.fingerprint,
        target_index=target_index,
        network_finite=bool(network_finite),
        first_layer4_spike_ms=first_spike("layer4_excitatory_v1"),
        first_layer23_interneuron_spike_ms=first_spike("layer23_inhibitory_v1"),
        first_layer23_excitatory_spike_ms=first_spike("layer23_excitatory_v1"),
        excitation_gate_peak=excitation_gate_peak,
        excitation_gate_peak_ms=excitation_gate_peak_ms,
        inhibition_gate_peak=inhibition_gate_peak,
        inhibition_gate_peak_ms=inhibition_gate_peak_ms,
        excitation_current_peak_pA=excitation_current_peak_pA,
        excitation_current_peak_ms=excitation_current_peak_ms,
        inhibition_current_trough_pA=inhibition_current_trough_pA,
        inhibition_current_trough_ms=inhibition_current_trough_ms,
        proximal_voltage_peak_mV=proximal_voltage_peak_mV,
        proximal_voltage_peak_ms=proximal_voltage_peak_ms,
        soma_voltage_peak_mV=soma_voltage_peak_mV,
        soma_voltage_peak_ms=soma_voltage_peak_ms,
        soma_axial_current_peak_pA=soma_axial_current_peak_pA,
        soma_axial_current_peak_ms=soma_axial_current_peak_ms,
    )


def run_figure6_relay_current_balance(
    *,
    conventions=None,
    protocol: Figure6LearningProtocol | None = None,
    target_index: int = 40,
    connected: bool = True,
    disabled_projection_ids: tuple[str, ...] = (),
    brian=None,
) -> Figure6RelayCurrentBalance:
    """Trace the source-defined bar drive and response of one relay cell.

    The assay leaves the complete first-order network connected and applies the
    same horizontal episode as :func:`run_figure6_learning`. It therefore
    localizes whether failure precedes or follows relay spike/release events
    without replacing the SMART circuit by an isolated-cell surrogate.
    """

    if brian is None:
        import brian2 as brian
    from ..classic_sector import (
        build_first_order_connected_sector,
        build_first_order_intrinsic_sector,
        figure6_runtime_conventions,
    )

    conventions = conventions or figure6_runtime_conventions()
    protocol = protocol or Figure6LearningProtocol()
    if not 0 <= target_index < 81:
        raise ValueError("target_index must address the 9x9 relay sheet")
    brian.start_scope()
    brian.defaultclock.dt = protocol.dt_ms * brian.ms
    if disabled_projection_ids and not connected:
        raise ValueError("projection controls require a connected Figure 6 sector")
    builder = build_first_order_connected_sector if connected else build_first_order_intrinsic_sector
    sector = builder(conventions=conventions, brian=brian)
    unknown_disabled = set(disabled_projection_ids) - set(sector.projections)
    if unknown_disabled:
        raise ValueError(f"unknown disabled projection IDs: {sorted(unknown_disabled)}")
    for projection_id in disabled_projection_ids:
        sector.network.remove(sector.projections[projection_id])
    relay = sector.populations["thalamic_relay"]
    if len(relay.compiled.external_input_ports) != 1:
        raise ValueError("Figure 6 relay must expose exactly one archived external-input port")
    external_port = relay.compiled.external_input_ports[0]
    calcium_variables = tuple(
        f"i_ca_{compartment.name}"
        for compartment in relay.cell_spec.compartments
        if compartment.g_ca_mS_cm2 is not None
    )
    if not calcium_variables:
        raise ValueError("Figure 6 relay must contain a T-type calcium compartment")
    variables = (
        "v_soma",
        f"e_{external_port.name}_effective",
        f"i_{external_port.name}",
        *calcium_variables,
    )
    state = brian.StateMonitor(
        relay.group,
        variables,
        record=[target_index],
        name="figure6_relay_current_balance_state",
    )
    relay_spikes = brian.SpikeMonitor(
        relay.group, name="figure6_relay_current_balance_relay_spikes"
    )
    layer4_spikes = brian.SpikeMonitor(
        sector.populations["layer4_excitatory_v1"].group,
        name="figure6_relay_current_balance_layer4_spikes",
    )
    sector.network.add(state, relay_spikes, layer4_spikes)
    if protocol.warmup_ms:
        sector.network.run(protocol.warmup_ms * brian.ms)
    apply_bar_stimulus(
        sector,
        ClassicBarStimulus(
            BarOrientation.HORIZONTAL,
            duration_ms=protocol.stimulus_ms,
            source_value=protocol.source_value,
            category_source_value=protocol.category_source_value,
            include_archived_category_pixel=True,
        ),
    )
    sector.network.run(protocol.stimulus_ms * brian.ms)
    time_ms = np.asarray(state.t / brian.ms) - protocol.warmup_ms
    keep = time_ms >= 0
    time_ms = time_ms[keep]

    def trace(variable: str, unit) -> np.ndarray:
        return np.asarray(getattr(state, variable)[0][keep] / unit, dtype=float)

    def maximum(variable: str, unit) -> tuple[float, float]:
        values = trace(variable, unit)
        index = int(np.argmax(values))
        return float(values[index]), float(time_ms[index])

    def target_events(monitor) -> tuple[float, ...]:
        indices = np.asarray(monitor.i, dtype=int)
        times = np.asarray(monitor.t / brian.ms) - protocol.warmup_ms
        selected = (indices == target_index) & (times >= 0)
        return tuple(float(value) for value in times[selected])

    external_current_peak_pA, external_current_peak_ms = maximum(
        f"i_{external_port.name}", brian.pA
    )
    calcium_current = sum(trace(variable, brian.pA) for variable in calcium_variables)
    calcium_peak_index = int(np.argmax(calcium_current))
    calcium_current_peak_pA = float(calcium_current[calcium_peak_index])
    calcium_current_peak_ms = float(time_ms[calcium_peak_index])
    soma_voltage_peak_mV, soma_voltage_peak_ms = maximum("v_soma", brian.mV)
    soma_voltage = trace("v_soma", brian.mV)
    relay_leak_mV = float(relay.group.e_l_soma[target_index] / brian.mV)
    sample_ms = protocol.dt_ms

    def time_above(threshold_mV: float) -> float:
        return float(np.count_nonzero(soma_voltage >= threshold_mV) * sample_ms)

    post_event = time_ms > soma_voltage_peak_ms
    post_event_indices = np.flatnonzero(post_event)
    post_event_minimum_index = int(
        post_event_indices[np.argmin(soma_voltage[post_event_indices])]
    )
    external_current = trace(f"i_{external_port.name}", brian.pA)
    effective_reversal = trace(f"e_{external_port.name}_effective", brian.mV)
    network_finite = all(
        np.isfinite(np.asarray(getattr(population.group, f"v_{compartment}")[:] / brian.mV)).all()
        for population in sector.populations.values()
        for compartment in population.compartments
    )
    return Figure6RelayCurrentBalance(
        convention_fingerprint=conventions.fingerprint,
        network_mode=(
            "connected_projection_control"
            if disabled_projection_ids
            else "connected" if connected else "intrinsic_only"
        ),
        disabled_projection_ids=disabled_projection_ids,
        target_index=target_index,
        network_finite=bool(network_finite),
        external_port_record_id=external_port.record_id,
        external_effective_reversal_mV=float(effective_reversal[0]),
        external_current_peak_pA=external_current_peak_pA,
        external_current_peak_ms=external_current_peak_ms,
        calcium_current_peak_pA=calcium_current_peak_pA,
        calcium_current_peak_ms=calcium_current_peak_ms,
        calcium_current_final_pA=float(calcium_current[-1]),
        soma_voltage_peak_mV=soma_voltage_peak_mV,
        soma_voltage_peak_ms=soma_voltage_peak_ms,
        soma_voltage_post_event_minimum_mV=float(soma_voltage[post_event_minimum_index]),
        soma_voltage_post_event_minimum_ms=float(time_ms[post_event_minimum_index]),
        soma_voltage_final_mV=float(soma_voltage[-1]),
        soma_time_above_minus20_mV=time_above(-20.0),
        soma_time_above_0_mV=time_above(0.0),
        soma_time_above_30_mV=time_above(30.0),
        soma_time_above_leak_mV=time_above(relay_leak_mV),
        external_current_final_pA=float(external_current[-1]),
        relay_event_times_ms=target_events(relay_spikes),
        target_layer4_event_times_ms=target_events(layer4_spikes),
    )


def run_figure6_top_down_learning_phase(
    *,
    conventions=None,
    protocol: Figure6LearningProtocol | None = None,
    source_index: int = 40,
    target_indices: tuple[int, ...] = HORIZONTAL_ONLY_INDICES + VERTICAL_ONLY_INDICES,
    brian=None,
) -> Figure6LearningPhaseResult:
    """Integrate the exact learning terms for selected Figure 6c synapses."""

    if brian is None:
        import brian2 as brian
    from ..classic_sector import build_first_order_connected_sector, figure6_runtime_conventions

    conventions = conventions or figure6_runtime_conventions()
    protocol = protocol or Figure6LearningProtocol()
    if not 0 <= source_index < 81:
        raise ValueError("source_index must address the 9x9 category sheet")
    if not target_indices or len(set(target_indices)) != len(target_indices):
        raise ValueError("target_indices must be nonempty and unique")
    if any(index < 0 or index >= 81 for index in target_indices):
        raise ValueError("target_indices must address the 9x9 relay sheet")
    brian.start_scope()
    brian.defaultclock.dt = protocol.dt_ms * brian.ms
    sector = build_first_order_connected_sector(
        conventions=conventions,
        instrument_learning_terms=True,
        brian=brian,
    )
    monitors: dict[str, tuple[Any, np.ndarray, Any]] = {}
    for projection_id in (TOP_DOWN_WIDE_PROJECTION_ID, TOP_DOWN_NARROW_PROJECTION_ID):
        projection = sector.projections[projection_id]
        source = np.asarray(projection.i[:], dtype=int)
        target = np.asarray(projection.j[:], dtype=int)
        selected = np.flatnonzero(
            (source == source_index) & np.isin(target, np.asarray(target_indices))
        )
        monitor = brian.StateMonitor(
            projection,
            (
                "w",
                "pre_signal",
                "last_post_spike",
                "learning_positive",
                "learning_negative",
                "learning_baseline",
            ),
            record=selected,
            name=f"figure6_learning_phase_{projection_id.rsplit('.', maxsplit=1)[-1]}",
        )
        monitors[projection_id] = (projection, selected, monitor)
    relay_targets = tuple(sorted(set(target_indices)))
    relay_state = brian.StateMonitor(
        sector.populations["thalamic_relay"].group,
        "v_soma",
        record=relay_targets,
        name="figure6_learning_phase_relay_state",
    )
    category_spikes = brian.SpikeMonitor(
        sector.populations["layer6ii_excitatory_v1"].group,
        name="figure6_learning_phase_category_spikes",
    )
    relay_spikes = brian.SpikeMonitor(
        sector.populations["thalamic_relay"].group,
        name="figure6_learning_phase_relay_spikes",
    )
    sector.network.add(
        *(monitor for _, _, monitor in monitors.values()),
        relay_state,
        category_spikes,
        relay_spikes,
    )
    if protocol.warmup_ms:
        sector.network.run(protocol.warmup_ms * brian.ms)
    apply_bar_stimulus(
        sector,
        ClassicBarStimulus(
            BarOrientation.HORIZONTAL,
            duration_ms=protocol.stimulus_ms,
            source_value=protocol.source_value,
            category_source_value=protocol.category_source_value,
            include_archived_category_pixel=True,
        ),
    )
    sector.network.run(protocol.stimulus_ms * brian.ms)

    def indexed_events(monitor, index: int) -> tuple[float, ...]:
        indices = np.asarray(monitor.i, dtype=int)
        times = np.asarray(monitor.t / brian.ms, dtype=float) - protocol.warmup_ms
        selected = (indices == index) & (times >= 0)
        return tuple(float(value) for value in times[selected])

    category_event_times = indexed_events(category_spikes, source_index)

    def interval_delta(
        trace: np.ndarray,
        final: float,
        start_sample: int,
        end_sample: int,
    ) -> float:
        end_value = final if end_sample >= len(trace) else float(trace[end_sample])
        return end_value - float(trace[start_sample])

    relay_voltage_by_target = {
        target: np.asarray(relay_state.v_soma[row] / brian.mV, dtype=float)
        for row, target in enumerate(relay_targets)
    }
    record_by_id = {record.id: record for record in MODELDB_FULL.projections}
    connections: list[Figure6LearningPhaseConnection] = []
    for projection_id, (projection, selected, monitor) in monitors.items():
        record = record_by_id[projection_id]
        if record.learning_rate is None or record.depotentiation_ms is None:
            raise ValueError(f"{projection_id} has incomplete learning parameters")
        time_ms = np.asarray(monitor.t / brian.ms, dtype=float) - protocol.warmup_ms
        keep = time_ms >= 0
        if np.count_nonzero(keep) < 2:
            raise ValueError("learning phase monitor did not record the stimulus epoch")
        step_ms = float(np.median(np.diff(time_ms[keep])))
        for row, synapse_index in enumerate(selected):
            weight = np.asarray(monitor.w[row][keep], dtype=float)
            pre = np.asarray(monitor.pre_signal[row][keep], dtype=float)
            target_index = int(np.asarray(projection.j[:], dtype=int)[synapse_index])
            last_post_spike_ms = (
                np.asarray(monitor.last_post_spike[row][keep] / brian.ms, dtype=float)
                - protocol.warmup_ms
            )
            post_elapsed_ms = time_ms[keep] - last_post_spike_ms
            post_voltage = relay_voltage_by_target[target_index][keep]
            if conventions.postsynaptic_learning_coordinate == "shifted_67_mV":
                post_voltage = post_voltage + 67.0
            elif conventions.postsynaptic_learning_coordinate == "relative_to_soma_leak":
                relay = sector.populations["thalamic_relay"].group
                post_voltage = post_voltage - float(relay.e_l_soma[target_index] / brian.mV)
            elif conventions.postsynaptic_learning_coordinate != "absolute_physical":
                raise ValueError(
                    "unsupported postsynaptic learning coordinate "
                    f"{conventions.postsynaptic_learning_coordinate!r}"
                )
            if (
                conventions.postsynaptic_depression_scale_convention
                == "serialized_projection_bounds"
            ):
                depression_scale = -float(record.asymptotic_weight) / float(record.weight)
            else:
                depression_scale = -np.asarray(
                    projection.w_baseline[synapse_index]
                ) / np.asarray(projection.w_maximum[synapse_index])
            above = post_voltage >= conventions.postsynaptic_learning_threshold_mV
            post = np.zeros_like(post_elapsed_ms)
            post[above] = depression_scale + 1.0
            early = (~above) & (post_elapsed_ms >= 0) & (post_elapsed_ms < 0.1)
            post[early] = depression_scale + 1.0 - post_elapsed_ms[early] / 0.1
            late = (
                (~above)
                & (post_elapsed_ms >= 0.1)
                & (post_elapsed_ms < record.depotentiation_ms + 0.1)
            )
            post[late] = depression_scale * (
                1.0 - (post_elapsed_ms[late] - 0.1) / record.depotentiation_ms
            )
            positive_trace = np.asarray(monitor.learning_positive[row][keep], dtype=float)
            negative_trace = np.asarray(monitor.learning_negative[row][keep], dtype=float)
            baseline_trace = np.asarray(monitor.learning_baseline[row][keep], dtype=float)
            positive_delta = float(
                projection.learning_positive[synapse_index] - positive_trace[0]
            )
            negative_delta = float(
                projection.learning_negative[synapse_index] - negative_trace[0]
            )
            baseline_delta = float(
                projection.learning_baseline[synapse_index] - baseline_trace[0]
            )
            delay_ms = float(projection.delay[synapse_index] / brian.ms)
            arrival_times = tuple(
                event_time + delay_ms
                for event_time in category_event_times
                if event_time + delay_ms < protocol.stimulus_ms
            )
            window_bounds = arrival_times + (protocol.stimulus_ms,)
            windows: list[Figure6LearningPhaseWindow] = []
            for start_ms, end_ms in pairwise(window_bounds):
                start_sample = int(np.searchsorted(time_ms[keep], start_ms, side="left"))
                end_sample = int(np.searchsorted(time_ms[keep], end_ms, side="left"))

                windows.append(
                    Figure6LearningPhaseWindow(
                        start_ms=start_ms,
                        end_ms=end_ms,
                        measured_delta=interval_delta(
                            weight,
                            float(projection.w[synapse_index]),
                            start_sample,
                            end_sample,
                        ),
                        positive_correlation_delta=interval_delta(
                            positive_trace,
                            float(projection.learning_positive[synapse_index]),
                            start_sample,
                            end_sample,
                        ),
                        negative_correlation_delta=interval_delta(
                            negative_trace,
                            float(projection.learning_negative[synapse_index]),
                            start_sample,
                            end_sample,
                        ),
                        baseline_delta=interval_delta(
                            baseline_trace,
                            float(projection.learning_baseline[synapse_index]),
                            start_sample,
                            end_sample,
                        ),
                    )
                )
            connections.append(
                Figure6LearningPhaseConnection(
                    projection_id=projection_id,
                    target_index=target_index,
                    initial_weight=float(weight[0]),
                    final_weight=float(projection.w[synapse_index]),
                    measured_delta=float(projection.w[synapse_index] - weight[0]),
                    positive_correlation_delta=positive_delta,
                    negative_correlation_delta=negative_delta,
                    baseline_delta=baseline_delta,
                    reconstructed_delta=positive_delta + negative_delta + baseline_delta,
                    presynaptic_gate_integral_ms=float(np.sum(pre) * step_ms),
                    postsynaptic_positive_overlap_ms=float(
                        np.sum(pre**2 * np.maximum(post, 0)) * step_ms
                    ),
                    postsynaptic_negative_overlap_ms=float(
                        np.sum(pre**2 * np.minimum(post, 0)) * step_ms
                    ),
                    windows=tuple(windows),
                )
            )

    return Figure6LearningPhaseResult(
        convention_fingerprint=conventions.fingerprint,
        dt_ms=protocol.dt_ms,
        duration_ms=protocol.stimulus_ms,
        category_event_times_ms=category_event_times,
        relay_event_times_ms={
            index: indexed_events(relay_spikes, index) for index in target_indices
        },
        connections=tuple(connections),
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
