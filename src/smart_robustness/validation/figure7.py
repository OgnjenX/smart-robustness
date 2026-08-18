"""Predeclared Figure 7 match/mismatch arousal metrics.

Figure 7 reports approximate firing rates for the single nonspecific thalamic
cell.  It does not provide a complete trace or a numeric reset latency, so the
rate result and the later qualitative reset result are deliberately separate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..modeldb_projections import MODELDB_FIRST_ORDER
from ..protocols import (
    ClassicMatchMismatchCue,
    MatchCondition,
    apply_bar_stimulus,
    apply_layer6ii_somatic_cue,
    apply_match_mismatch_cue,
    clear_bar_stimulus,
    clear_layer6ii_somatic_cue,
    clear_match_mismatch_cue,
)
from ..synapses import modeldb_topology_pairs
from .figure6 import TOP_DOWN_NARROW_PROJECTION_ID, TOP_DOWN_WIDE_PROJECTION_ID

FIGURE7_MATCH_RATE_HZ = 40.0
FIGURE7_MISMATCH_RATE_HZ = 70.0
FIGURE7_RATE_TOLERANCE_HZ = 10.0
FIGURE7_REQUIRED_LEARNED_PROJECTIONS = (
    TOP_DOWN_WIDE_PROJECTION_ID,
    TOP_DOWN_NARROW_PROJECTION_ID,
)
FIGURE7_RELAY_DIAGNOSTIC_INDICES = (22, 31, 38, 39, 40, 41, 42, 49, 58)
FIGURE14_V1_CORTICAL_POPULATIONS = (
    "layer23_excitatory_v1",
    "layer23_inhibitory_v1",
    "layer4_excitatory_v1",
    "layer4_inhibitory_v1",
    "layer5_excitatory_v1",
    "layer6i_excitatory_v1",
    "layer6ii_excitatory_v1",
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
    *,
    derive_from_source: bool = False,
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
        if derive_from_source:
            record = MODELDB_FIRST_ORDER.by_id(projection_id)
            source, target, spatial_factor = modeldb_topology_pairs(
                record,
                source_shape=(9, 9),
                target_shape=(9, 9),
                gaussian_weight_convention="source_peak",
            )
            assert record.asymptotic_weight is not None and record.weight is not None
            values = float(record.asymptotic_weight) * spatial_factor
            maximum = np.maximum(float(record.weight), values)
        else:
            values = np.asarray(projection.w[:], dtype=float).copy()
            source = np.asarray(projection.i[:], dtype=int)
            target = np.asarray(projection.j[:], dtype=int)
            maximum = np.asarray(projection.w_maximum[:], dtype=float)
        selected = source == reference.source_index
        values[selected] = np.minimum(desired[target[selected]] / 2.0, maximum[selected])
        learned[projection_id] = tuple(float(value) for value in values)
    return learned


def apply_figure7_learned_state(
    projections: Mapping[str, object],
    learned_weights: Mapping[str, tuple[float, ...] | np.ndarray],
    *,
    freeze_learning: bool = True,
    verify_runtime_bounds: bool = True,
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
        if verify_runtime_bounds:
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
    v1_cortical_spike_times_ms: tuple[float, ...] = ()
    v2_layer4_spike_indices: tuple[int, ...] = ()
    v2_layer4_spike_times_ms: tuple[float, ...] = ()
    v2_relay_spike_indices: tuple[int, ...] = ()
    v2_relay_spike_times_ms: tuple[float, ...] = ()
    convention_fingerprint: str | None = None
    top_down_current_pA: float | None = None
    top_down_cue_lead_ms: float = 0.0
    equilibration_ms: float = 0.0
    learned_state_provenance: str = "unspecified"
    network_scope: str = "first_order"
    relay_top_down_ampa_peak_by_index: tuple[tuple[int, float], ...] = ()
    relay_top_down_ampa_integral_ms_by_index: tuple[tuple[int, float], ...] = ()
    relay_top_down_nmda_peak_by_index: tuple[tuple[int, float], ...] = ()
    relay_distal_voltage_range_mV_by_index: tuple[tuple[int, float, float], ...] = ()
    relay_proximal_voltage_range_mV_by_index: tuple[tuple[int, float, float], ...] = ()
    relay_soma_voltage_range_mV_by_index: tuple[tuple[int, float, float], ...] = ()
    relay_trn_gaba_peak_by_index: tuple[tuple[int, float], ...] = ()
    relay_trn_gaba_integral_ms_by_index: tuple[tuple[int, float], ...] = ()
    trn_layer6ii_ampa_peak_by_index: tuple[tuple[int, float], ...] = ()
    trn_layer6ii_nmda_peak_by_index: tuple[tuple[int, float], ...] = ()
    trn_relay_ampa_peak_by_index: tuple[tuple[int, float], ...] = ()
    trn_proximal_voltage_range_mV_by_index: tuple[tuple[int, float, float], ...] = ()
    trn_soma_voltage_range_mV_by_index: tuple[tuple[int, float, float], ...] = ()
    trn_post_startup_soma_voltage_range_mV_by_index: tuple[
        tuple[int, float, float], ...
    ] = ()

    def __post_init__(self) -> None:
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        if self.top_down_cue_lead_ms < 0:
            raise ValueError("top_down_cue_lead_ms cannot be negative")
        if self.equilibration_ms < 0:
            raise ValueError("equilibration_ms cannot be negative")

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


@dataclass(frozen=True, slots=True)
class Figure7PathwayAssessment:
    """Score the causal relay-to-TRN ordering stated in the Figure 7 caption.

    The publication does not report numeric relay or TRN counts.  It does state
    that a match lets more specific-thalamic cells fire, which in turn makes
    TRN inhibition stronger than during mismatch.  These are therefore
    directional gates, kept separate from the approximate 40/70-Hz output
    targets.
    """

    match_active_relay_cells: int
    mismatch_active_relay_cells: int
    match_trn_spikes: int
    mismatch_trn_spikes: int

    @property
    def relay_subset_pass(self) -> bool:
        return self.match_active_relay_cells > self.mismatch_active_relay_cells

    @property
    def trn_order_pass(self) -> bool:
        return self.match_trn_spikes > self.mismatch_trn_spikes

    @property
    def reproduced_pathway(self) -> bool:
        return self.relay_subset_pass and self.trn_order_pass


@dataclass(frozen=True, slots=True)
class Figure7ReproductionAssessment:
    """Combined behavioral and causal validation for the Figure 7 result."""

    arousal: Figure7ArousalAssessment
    pathway: Figure7PathwayAssessment

    @property
    def reproduced(self) -> bool:
        return self.arousal.reproduced_arousal and self.pathway.reproduced_pathway


def _validate_figure7_pair(
    match: Figure7ConditionResult,
    mismatch: Figure7ConditionResult,
) -> None:
    if match.condition is not MatchCondition.MATCH:
        raise ValueError("match result must use the match condition")
    if mismatch.condition is not MatchCondition.MISMATCH:
        raise ValueError("mismatch result must use the mismatch condition")
    if match.duration_ms != mismatch.duration_ms:
        raise ValueError("match and mismatch results must use the same duration")


def assess_figure7_arousal(
    match: Figure7ConditionResult,
    mismatch: Figure7ConditionResult,
    *,
    tolerance_hz: float = FIGURE7_RATE_TOLERANCE_HZ,
) -> Figure7ArousalAssessment:
    """Score only the published Figure 7 nonspecific-thalamus rate claim."""

    _validate_figure7_pair(match, mismatch)
    if tolerance_hz < 0:
        raise ValueError("tolerance_hz cannot be negative")
    return Figure7ArousalAssessment(
        match_rate_hz=match.nonspecific_rate_hz,
        mismatch_rate_hz=mismatch.nonspecific_rate_hz,
        tolerance_hz=tolerance_hz,
    )


def assess_figure7_pathway(
    match: Figure7ConditionResult,
    mismatch: Figure7ConditionResult,
) -> Figure7PathwayAssessment:
    """Score the caption's match-greater-than-mismatch relay/TRN mechanism."""

    _validate_figure7_pair(match, mismatch)
    return Figure7PathwayAssessment(
        match_active_relay_cells=len(set(match.relay_spike_indices)),
        mismatch_active_relay_cells=len(set(mismatch.relay_spike_indices)),
        match_trn_spikes=len(match.trn_spike_times_ms),
        mismatch_trn_spikes=len(mismatch.trn_spike_times_ms),
    )


def assess_figure7_reproduction(
    match: Figure7ConditionResult,
    mismatch: Figure7ConditionResult,
    *,
    tolerance_hz: float = FIGURE7_RATE_TOLERANCE_HZ,
) -> Figure7ReproductionAssessment:
    """Require both the published output rates and their stated mechanism."""

    return Figure7ReproductionAssessment(
        arousal=assess_figure7_arousal(match, mismatch, tolerance_hz=tolerance_hz),
        pathway=assess_figure7_pathway(match, mismatch),
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
    relay_clamp_compartment: str = "proximal_dendrite",
    include_higher_order_loop: bool = False,
    record_relay_diagnostics: bool = False,
    record_v1_cortical_spikes: bool = False,
    projection_weight_scales: Mapping[str, float] | None = None,
    top_down_cue_lead_ms: float = 0.0,
    equilibration_ms: float = 0.0,
    cpp_standalone_directory: str | Path | None = None,
    brian=None,
) -> Figure7ConditionResult:
    """Run one source-labeled Figure 7 match or mismatch condition."""

    if learned_weights is not None and use_paper_constrained_reference:
        raise ValueError("pass learned_weights or request the paper-constrained reference, not both")
    if learned_weights is None and not use_paper_constrained_reference:
        raise ValueError("Figure 7 requires an explicit learned expectation state")
    if duration_ms <= 0 or dt_ms <= 0:
        raise ValueError("duration_ms and dt_ms must be positive")
    if top_down_cue_lead_ms < 0:
        raise ValueError("top_down_cue_lead_ms cannot be negative")
    if equilibration_ms < 0:
        raise ValueError("equilibration_ms cannot be negative")
    if brian is None:
        import brian2 as brian
    if cpp_standalone_directory is not None:
        brian.device.reinit()
        brian.set_device(
            "cpp_standalone",
            directory=str(Path(cpp_standalone_directory).resolve()),
            build_on_run=False,
        )
    from ..classic_sector import (
        build_first_order_connected_sector,
        build_first_order_voltage_clamp_sector,
        build_full_smart_network,
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
    if exact_relay_voltage_clamp and include_higher_order_loop:
        raise ValueError("the exact relay-clamp audit is only defined for the first-order assay")
    if exact_relay_voltage_clamp and top_down_cue_lead_ms > 0:
        raise ValueError("the relay-clamp audit cannot represent a cue-only lead interval")
    if exact_relay_voltage_clamp:
        sector = build_first_order_voltage_clamp_sector(
            clamped_relay_indices=orientation.active_indices,
            holding_mV=orientation.expected_holding_mV,
            compartment=relay_clamp_compartment,
            conventions=conventions,
            brian=brian,
        )
    elif include_higher_order_loop:
        sector = build_full_smart_network(conventions=conventions, brian=brian)
    else:
        sector = build_first_order_connected_sector(conventions=conventions, brian=brian)
    if use_paper_constrained_reference:
        learned_weights = paper_constrained_figure6_expectation(
            sector.projections,
            paper_reference,
            derive_from_source=cpp_standalone_directory is not None,
        )
        provenance = "paper-constrained-figure6c-reference"
    else:
        provenance = "simulated-learned-weight-snapshot"
    assert learned_weights is not None
    apply_figure7_learned_state(
        sector.projections,
        learned_weights,
        verify_runtime_bounds=cpp_standalone_directory is None,
    )
    if projection_weight_scales:
        unknown = set(projection_weight_scales) - set(sector.projections)
        if unknown:
            raise ValueError(f"unknown projection scale IDs: {sorted(unknown)}")
        for projection_id, scale in projection_weight_scales.items():
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("projection weight scales must be finite and positive")
            projection = sector.projections[projection_id]
            blocks = getattr(projection, "blocks", (projection,))
            for block in blocks:
                # A symbolic assignment is evaluated by Brian after the
                # source-defined spatial weights have been initialized.  It
                # therefore works in both runtime and C++ standalone, where
                # reading state arrays before the build is unsupported.
                block.w = f"w*({float(scale)!r})"
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
    monitors = [nonspecific, layer4, relay, trn, category]
    cortical_spike_monitors: dict[str, object] = {}
    if record_v1_cortical_spikes:
        cortical_spike_monitors = {
            population_name: brian.SpikeMonitor(
                sector.populations[population_name].group,
                name=f"figure14_{condition.value}_{population_name}_spikes",
            )
            for population_name in FIGURE14_V1_CORTICAL_POPULATIONS
        }
        monitors.extend(cortical_spike_monitors.values())
    relay_state = None
    trn_state = None
    if record_relay_diagnostics:
        relay_state = brian.StateMonitor(
            sector.populations["thalamic_relay"].group,
            (
                "port_003_gate",
                "port_000_gate",
                "port_001_gate",
                "port_004_gate",
                "port_005_gate",
                "port_006_gate",
                "port_007_gate",
                "v_distal_dendrite",
                "v_proximal_dendrite",
                "v_soma",
            ),
            record=FIGURE7_RELAY_DIAGNOSTIC_INDICES,
            name=f"figure7_{condition.value}_relay_pathway_state",
        )
        trn_state = brian.StateMonitor(
            sector.populations["trn"].group,
            (
                "port_001_gate",
                "port_002_gate",
                "port_004_gate",
                "v_proximal_dendrite",
                "v_soma",
            ),
            record=FIGURE7_RELAY_DIAGNOSTIC_INDICES,
            name=f"figure7_{condition.value}_trn_pathway_state",
        )
        monitors.extend((relay_state, trn_state))
    v2_layer4 = None
    v2_relay = None
    if include_higher_order_loop:
        v2_layer4 = brian.SpikeMonitor(
            sector.populations["layer4_excitatory_v2"].group,
            name=f"figure7_{condition.value}_v2_layer4_spikes",
        )
        v2_relay = brian.SpikeMonitor(
            sector.populations["thalamic_relay_v2"].group,
            name=f"figure7_{condition.value}_v2_relay_spikes",
        )
        monitors.extend((v2_layer4, v2_relay))
    sector.network.add(*monitors)
    # KInNeSS documentation distinguishes the serialized leak equilibrium
    # from the actual membrane rest when voltage-gated channels are present.
    # A source-unreported settling interval is therefore an explicit protocol
    # discriminator, never an implicit part of the classic trial.
    if equilibration_ms > 0:
        sector.network.run(equilibration_ms * brian.ms)
    cue = ClassicMatchMismatchCue(
        condition=condition,
        top_down_current_pA=top_down_current_pA,
        duration_ms=duration_ms,
    )
    if top_down_cue_lead_ms > 0:
        apply_layer6ii_somatic_cue(
            sector,
            current_pA=cue.top_down_current_pA,
            cell_index=cue.top_down_cell_index,
            brian=brian,
        )
        sector.network.run(top_down_cue_lead_ms * brian.ms)
        apply_bar_stimulus(
            sector,
            cue.bottom_up_stimulus,
            apply_relay_input=not exact_relay_voltage_clamp,
        )
    else:
        apply_match_mismatch_cue(
            sector, cue, apply_relay_input=not exact_relay_voltage_clamp, brian=brian
        )
    sector.network.run(duration_ms * brian.ms)
    if top_down_cue_lead_ms > 0:
        clear_bar_stimulus(sector, cue.bottom_up_stimulus)
        clear_layer6ii_somatic_cue(sector, brian=brian)
    else:
        clear_match_mismatch_cue(sector, cue, brian=brian)
    if cpp_standalone_directory is not None:
        from ..standalone import build_and_run_cpp_standalone

        build_and_run_cpp_standalone(brian, cpp_standalone_directory)
    ampa_peak: tuple[tuple[int, float], ...] = ()
    ampa_integral: tuple[tuple[int, float], ...] = ()
    nmda_peak: tuple[tuple[int, float], ...] = ()
    voltage_range: tuple[tuple[int, float, float], ...] = ()
    proximal_voltage_range: tuple[tuple[int, float, float], ...] = ()
    soma_voltage_range: tuple[tuple[int, float, float], ...] = ()
    relay_trn_gaba_peak: tuple[tuple[int, float], ...] = ()
    relay_trn_gaba_integral: tuple[tuple[int, float], ...] = ()
    trn_layer6ii_ampa_peak: tuple[tuple[int, float], ...] = ()
    trn_layer6ii_nmda_peak: tuple[tuple[int, float], ...] = ()
    trn_relay_ampa_peak: tuple[tuple[int, float], ...] = ()
    trn_voltage_range: tuple[tuple[int, float, float], ...] = ()
    trn_soma_voltage_range: tuple[tuple[int, float, float], ...] = ()
    trn_post_startup_soma_voltage_range: tuple[tuple[int, float, float], ...] = ()
    if relay_state is not None:
        ampa = np.asarray(relay_state.port_005_gate) + np.asarray(
            relay_state.port_007_gate
        )
        nmda = np.asarray(relay_state.port_003_gate) + np.asarray(
            relay_state.port_006_gate
        )
        voltage_mV = np.asarray(relay_state.v_distal_dendrite / brian.mV)
        times_ms = np.asarray(relay_state.t / brian.ms)
        stimulus_start_ms = equilibration_ms + top_down_cue_lead_ms
        diagnostic_window = times_ms >= stimulus_start_ms
        times_ms = times_ms[diagnostic_window] - stimulus_start_ms
        ampa = ampa[:, diagnostic_window]
        nmda = nmda[:, diagnostic_window]
        voltage_mV = voltage_mV[:, diagnostic_window]
        ampa_peak = tuple(
            (index, float(np.max(values)))
            for index, values in zip(FIGURE7_RELAY_DIAGNOSTIC_INDICES, ampa, strict=True)
        )
        ampa_integral = tuple(
            (index, float(np.trapz(values, times_ms)))
            for index, values in zip(FIGURE7_RELAY_DIAGNOSTIC_INDICES, ampa, strict=True)
        )
        nmda_peak = tuple(
            (index, float(np.max(values)))
            for index, values in zip(FIGURE7_RELAY_DIAGNOSTIC_INDICES, nmda, strict=True)
        )
        voltage_range = tuple(
            (index, float(np.min(values)), float(np.max(values)))
            for index, values in zip(
                FIGURE7_RELAY_DIAGNOSTIC_INDICES, voltage_mV, strict=True
            )
        )
        proximal_voltage_mV = np.asarray(relay_state.v_proximal_dendrite / brian.mV)[
            :, diagnostic_window
        ]
        proximal_voltage_range = tuple(
            (index, float(np.min(values)), float(np.max(values)))
            for index, values in zip(
                FIGURE7_RELAY_DIAGNOSTIC_INDICES, proximal_voltage_mV, strict=True
            )
        )
        soma_voltage_mV = np.asarray(relay_state.v_soma / brian.mV)[:, diagnostic_window]
        soma_voltage_range = tuple(
            (index, float(np.min(values)), float(np.max(values)))
            for index, values in zip(
                FIGURE7_RELAY_DIAGNOSTIC_INDICES, soma_voltage_mV, strict=True
            )
        )
        relay_trn_gaba = (
            np.asarray(relay_state.port_000_gate)[:, diagnostic_window]
            + np.asarray(relay_state.port_001_gate)[:, diagnostic_window]
            + np.asarray(relay_state.port_004_gate)[:, diagnostic_window]
        )
        relay_trn_gaba_peak = tuple(
            (index, float(np.max(values)))
            for index, values in zip(
                FIGURE7_RELAY_DIAGNOSTIC_INDICES, relay_trn_gaba, strict=True
            )
        )
        relay_trn_gaba_integral = tuple(
            (index, float(np.trapz(values, times_ms)))
            for index, values in zip(
                FIGURE7_RELAY_DIAGNOSTIC_INDICES, relay_trn_gaba, strict=True
            )
        )
    if trn_state is not None:
        trn_layer6ii_ampa_peak = tuple(
            (index, float(np.max(values)))
            for index, values in zip(
                FIGURE7_RELAY_DIAGNOSTIC_INDICES,
                np.asarray(trn_state.port_004_gate)[:, diagnostic_window],
                strict=True,
            )
        )
        trn_layer6ii_nmda_peak = tuple(
            (index, float(np.max(values)))
            for index, values in zip(
                FIGURE7_RELAY_DIAGNOSTIC_INDICES,
                np.asarray(trn_state.port_001_gate)[:, diagnostic_window],
                strict=True,
            )
        )
        trn_relay_ampa_peak = tuple(
            (index, float(np.max(values)))
            for index, values in zip(
                FIGURE7_RELAY_DIAGNOSTIC_INDICES,
                np.asarray(trn_state.port_002_gate)[:, diagnostic_window],
                strict=True,
            )
        )
        trn_voltage_mV = np.asarray(trn_state.v_proximal_dendrite / brian.mV)[
            :, diagnostic_window
        ]
        trn_voltage_range = tuple(
            (index, float(np.min(values)), float(np.max(values)))
            for index, values in zip(
                FIGURE7_RELAY_DIAGNOSTIC_INDICES, trn_voltage_mV, strict=True
            )
        )
        trn_soma_voltage_mV = np.asarray(trn_state.v_soma / brian.mV)[
            :, diagnostic_window
        ]
        trn_soma_voltage_range = tuple(
            (index, float(np.min(values)), float(np.max(values)))
            for index, values in zip(
                FIGURE7_RELAY_DIAGNOSTIC_INDICES, trn_soma_voltage_mV, strict=True
            )
        )
        post_startup_window = times_ms >= 5.0
        trn_post_startup_soma_voltage_range = tuple(
            (index, float(np.min(values)), float(np.max(values)))
            for index, values in zip(
                FIGURE7_RELAY_DIAGNOSTIC_INDICES,
                trn_soma_voltage_mV[:, post_startup_window],
                strict=True,
            )
        )
    def stimulus_times(monitor) -> tuple[float, ...]:
        stimulus_start_ms = equilibration_ms + top_down_cue_lead_ms
        return tuple(
            float(value - stimulus_start_ms)
            for value in monitor.t / brian.ms
            if float(value) >= stimulus_start_ms
        )

    def stimulus_indices(monitor) -> tuple[int, ...]:
        stimulus_start_ms = equilibration_ms + top_down_cue_lead_ms
        return tuple(
            int(index)
            for index, value in zip(monitor.i, monitor.t / brian.ms, strict=True)
            if float(value) >= stimulus_start_ms
        )

    result = Figure7ConditionResult(
        condition=condition,
        duration_ms=duration_ms,
        nonspecific_spike_times_ms=stimulus_times(nonspecific),
        layer4_spike_times_ms=stimulus_times(layer4),
        relay_spike_indices=stimulus_indices(relay),
        relay_spike_times_ms=stimulus_times(relay),
        trn_spike_indices=stimulus_indices(trn),
        trn_spike_times_ms=stimulus_times(trn),
        category_spike_indices=stimulus_indices(category),
        category_spike_times_ms=stimulus_times(category),
        v1_cortical_spike_times_ms=tuple(
            sorted(
                spike_time
                for monitor in cortical_spike_monitors.values()
                for spike_time in stimulus_times(monitor)
            )
        ),
        v2_layer4_spike_indices=(
            () if v2_layer4 is None else stimulus_indices(v2_layer4)
        ),
        v2_layer4_spike_times_ms=(
            () if v2_layer4 is None else stimulus_times(v2_layer4)
        ),
        v2_relay_spike_indices=(
            () if v2_relay is None else stimulus_indices(v2_relay)
        ),
        v2_relay_spike_times_ms=(
            () if v2_relay is None else stimulus_times(v2_relay)
        ),
        convention_fingerprint=conventions.fingerprint,
        top_down_current_pA=top_down_current_pA,
        top_down_cue_lead_ms=top_down_cue_lead_ms,
        equilibration_ms=equilibration_ms,
        learned_state_provenance=provenance,
        network_scope="full_two_area" if include_higher_order_loop else "first_order",
        relay_top_down_ampa_peak_by_index=ampa_peak,
        relay_top_down_ampa_integral_ms_by_index=ampa_integral,
        relay_top_down_nmda_peak_by_index=nmda_peak,
        relay_distal_voltage_range_mV_by_index=voltage_range,
        relay_proximal_voltage_range_mV_by_index=proximal_voltage_range,
        relay_soma_voltage_range_mV_by_index=soma_voltage_range,
        relay_trn_gaba_peak_by_index=relay_trn_gaba_peak,
        relay_trn_gaba_integral_ms_by_index=relay_trn_gaba_integral,
        trn_layer6ii_ampa_peak_by_index=trn_layer6ii_ampa_peak,
        trn_layer6ii_nmda_peak_by_index=trn_layer6ii_nmda_peak,
        trn_relay_ampa_peak_by_index=trn_relay_ampa_peak,
        trn_proximal_voltage_range_mV_by_index=trn_voltage_range,
        trn_soma_voltage_range_mV_by_index=trn_soma_voltage_range,
        trn_post_startup_soma_voltage_range_mV_by_index=(
            trn_post_startup_soma_voltage_range
        ),
    )
    if cpp_standalone_directory is not None:
        brian.device.reinit()
        brian.set_device("runtime")
    return result
