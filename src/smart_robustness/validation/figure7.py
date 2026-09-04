"""Source-constrained Figure 7 match/mismatch arousal metrics.

The rendered Figure 7c panel fixes a 100-ms comparison with 40-Hz match and
70-Hz mismatch nonspecific-thalamus output. The caption also fixes the spatial
relay and TRN-order mechanism; no complete numeric trace is published.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from ..modeldb_projections import MODELDB_FIRST_ORDER
from ..protocols import (
    BarOrientation,
    ClassicBarStimulus,
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


class TopDownCurrentMode(StrEnum):
    """Published or source-adjacent interpretations of cue-current duration."""

    SUSTAINED_EPOCH = "sustained_epoch"
    UNTIL_CUED_CELL_FIRST_EVENT = "until_cued_cell_first_event"
    UNTIL_CUED_CELL_EVENT_LIMIT = "until_cued_cell_event_limit"

FIGURE7_REQUIRED_LEARNED_PROJECTIONS = (
    TOP_DOWN_WIDE_PROJECTION_ID,
    TOP_DOWN_NARROW_PROJECTION_ID,
)
FIGURE7_TOP_DOWN_RELAY_PROJECTION_IDS = (
    "modeldb112923.projection.003",
    "modeldb112923.projection.005",
    "modeldb112923.projection.006",
    "modeldb112923.projection.007",
)
FIGURE7_RELAY_DIAGNOSTIC_INDICES = (22, 31, 38, 39, 40, 41, 42, 49, 58)
FIGURE7_MATCH_RELAY_INDICES = frozenset((38, 39, 40, 41, 42))
FIGURE7_MISMATCH_INPUT_INDICES = frozenset((22, 31, 40, 49, 58))
FIGURE7_RELAY_OVERLAP_INDICES = FIGURE7_MATCH_RELAY_INDICES & FIGURE7_MISMATCH_INPUT_INDICES
FIGURE14_V1_CORTICAL_POPULATIONS = (
    "layer23_excitatory_v1",
    "layer23_inhibitory_v1",
    "layer4_excitatory_v1",
    "layer4_inhibitory_v1",
    "layer5_excitatory_v1",
    "layer6i_excitatory_v1",
    "layer6ii_excitatory_v1",
)


def learned_expectation_support_by_target(
    learned_weights: Mapping[str, tuple[float, ...] | np.ndarray],
    *,
    source_index: int = 40,
) -> np.ndarray:
    """Resolve one learned category's normalized target support.

    This is an explicitly reconstructed mesoscopic comparator signal, not a
    recovered SMART source equation. It uses only the adaptive projection
    weights learned by the selected layer-6II category and their archived
    ModelDB topology; it does not inspect the trial condition or bar labels.
    """

    if not 0 <= source_index < 81:
        raise ValueError("source_index must address the 9x9 sheet")
    missing = set(FIGURE7_REQUIRED_LEARNED_PROJECTIONS) - set(learned_weights)
    if missing:
        raise ValueError(f"missing learned Figure 7 projections: {sorted(missing)}")

    support = np.zeros(81, dtype=float)
    for projection_id in FIGURE7_REQUIRED_LEARNED_PROJECTIONS:
        record = MODELDB_FIRST_ORDER.by_id(projection_id)
        source, target, _ = modeldb_topology_pairs(
            record,
            source_shape=(9, 9),
            target_shape=(9, 9),
            gaussian_weight_convention="source_peak",
        )
        values = np.asarray(learned_weights[projection_id], dtype=float)
        if values.shape != source.shape:
            raise ValueError(
                f"{projection_id}: expected {source.shape} weights, got {values.shape}"
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"{projection_id}: learned weights must be finite and nonnegative")
        selected = source == source_index
        np.add.at(support, target[selected], values[selected])

    peak = float(np.max(support))
    if peak <= 0:
        raise ValueError("selected source has no positive learned expectation support")
    return support / peak


def comparator_relay_input_gains(
    learned_weights: Mapping[str, tuple[float, ...] | np.ndarray],
    *,
    floor: float,
    source_index: int = 40,
) -> np.ndarray:
    """Blend learned local coincidence support with a uniform relay floor."""

    if not np.isfinite(floor) or not 0.0 <= floor <= 1.0:
        raise ValueError("comparator floor must be finite and lie in [0, 1]")
    support = learned_expectation_support_by_target(
        learned_weights, source_index=source_index
    )
    return floor + (1.0 - floor) * support


def half_max_comparator_relay_input_gains(
    learned_weights: Mapping[str, tuple[float, ...] | np.ndarray],
    *,
    source_index: int = 40,
) -> np.ndarray:
    """Return a saturated gate at the learned field's standard half maximum."""

    support = learned_expectation_support_by_target(
        learned_weights, source_index=source_index
    )
    return (support >= 0.5).astype(float)


def top_k_comparator_relay_input_gains(
    learned_weights: Mapping[str, tuple[float, ...] | np.ndarray],
    *,
    target_count: int,
    source_index: int = 40,
) -> np.ndarray:
    """Select a fixed-cardinality learned field with deterministic tie order."""

    if isinstance(target_count, bool) or not isinstance(target_count, int):
        raise TypeError("comparator target_count must be an integer")
    if not 1 <= target_count <= 81:
        raise ValueError("comparator target_count must lie in [1, 81]")
    support = learned_expectation_support_by_target(
        learned_weights, source_index=source_index
    )
    selected = np.lexsort((np.arange(81), -support))[:target_count]
    gains = np.zeros(81, dtype=float)
    gains[selected] = 1.0
    return gains


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


def expand_figure7_source_expectation_toward_bounds(
    learned_weights: Mapping[str, tuple[float, ...] | np.ndarray],
    *,
    headroom_fraction: float,
    source_index: int = 40,
) -> tuple[dict[str, tuple[float, ...]], float]:
    """Scale one learned category's on-center while preserving its shape.

    ``headroom_fraction=0`` returns the supplied learned state, while ``1``
    applies the largest common multiplicative factor that does not exceed any
    source-derived adaptive-weight maximum. Zero-weight connections remain
    zero and every nonselected source row remains unchanged.
    """

    if not 0.0 <= headroom_fraction <= 1.0:
        raise ValueError("headroom_fraction must lie in [0, 1]")
    if not 0 <= source_index < 81:
        raise ValueError("source_index must address the 9x9 sheet")
    missing = set(FIGURE7_REQUIRED_LEARNED_PROJECTIONS) - set(learned_weights)
    if missing:
        raise ValueError(f"missing learned Figure 7 projections: {sorted(missing)}")

    resolved: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    factor_limits: list[float] = []
    for projection_id in FIGURE7_REQUIRED_LEARNED_PROJECTIONS:
        record = MODELDB_FIRST_ORDER.by_id(projection_id)
        source, _, spatial_factor = modeldb_topology_pairs(
            record,
            source_shape=(9, 9),
            target_shape=(9, 9),
            gaussian_weight_convention="source_peak",
        )
        assert record.asymptotic_weight is not None and record.weight is not None
        maximum = np.maximum(
            float(record.weight),
            float(record.asymptotic_weight) * spatial_factor,
        )
        values = np.asarray(learned_weights[projection_id], dtype=float).copy()
        if values.shape != maximum.shape:
            raise ValueError(
                f"{projection_id}: expected {maximum.shape} weights, got {values.shape}"
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"{projection_id}: learned weights must be finite and nonnegative")
        if np.any(values > maximum + 1e-12):
            raise ValueError(f"{projection_id}: learned weights exceed source-derived maxima")
        selected_positive = (source == source_index) & (values > 0)
        if np.any(selected_positive):
            factor_limits.extend(
                float(value)
                for value in maximum[selected_positive] / values[selected_positive]
            )
        resolved[projection_id] = (values, source, maximum)
    if not factor_limits:
        raise ValueError("selected source has no positive learned expectation weights")
    maximum_common_factor = min(factor_limits)
    applied_factor = 1.0 + headroom_fraction * (maximum_common_factor - 1.0)
    expanded: dict[str, tuple[float, ...]] = {}
    for projection_id, (values, source, maximum) in resolved.items():
        selected = source == source_index
        values[selected] = np.minimum(values[selected] * applied_factor, maximum[selected])
        expanded[projection_id] = tuple(float(value) for value in values)
    return expanded, float(applied_factor)


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


def restrict_figure7_top_down_relay_sources(
    projections: Mapping[str, object],
    source_indices: frozenset[int],
) -> None:
    """Diagnostic mask for selected-category relay-directed feedback.

    This is not part of the recovered SMART source. It isolates whether
    bottom-up recruitment of nonselected layer-6II cells defeats the intended
    learned on-center/off-surround match computation.
    """

    if not source_indices or any(not 0 <= index < 81 for index in source_indices):
        raise ValueError("top-down relay source indices must address the 9x9 sheet")
    for projection_id in FIGURE7_TOP_DOWN_RELAY_PROJECTION_IDS:
        projection = projections[projection_id]
        for block in getattr(projection, "blocks", (projection,)):
            presynaptic = np.asarray(block.i[:], dtype=int)
            weights = np.asarray(block.w[:], dtype=float).copy()
            weights[~np.isin(presynaptic, tuple(source_indices))] = 0.0
            block.w = weights


@dataclass(frozen=True, slots=True)
class Figure7ConditionResult:
    condition: MatchCondition
    duration_ms: float
    nonspecific_spike_times_ms: tuple[float, ...]
    layer4_spike_indices: tuple[int, ...] = ()
    layer4_spike_times_ms: tuple[float, ...] = ()
    relay_spike_indices: tuple[int, ...] = ()
    relay_spike_times_ms: tuple[float, ...] = ()
    trn_spike_indices: tuple[int, ...] = ()
    trn_spike_times_ms: tuple[float, ...] = ()
    category_spike_indices: tuple[int, ...] = ()
    category_spike_times_ms: tuple[float, ...] = ()
    equilibration_nonspecific_spike_times_ms: tuple[float, ...] = ()
    equilibration_layer4_spike_indices: tuple[int, ...] = ()
    equilibration_layer4_spike_times_ms: tuple[float, ...] = ()
    equilibration_relay_spike_indices: tuple[int, ...] = ()
    equilibration_relay_spike_times_ms: tuple[float, ...] = ()
    equilibration_trn_spike_indices: tuple[int, ...] = ()
    equilibration_trn_spike_times_ms: tuple[float, ...] = ()
    equilibration_category_spike_indices: tuple[int, ...] = ()
    equilibration_category_spike_times_ms: tuple[float, ...] = ()
    cue_lead_category_spike_indices: tuple[int, ...] = ()
    cue_lead_category_spike_times_ms: tuple[float, ...] = ()
    cue_lead_nonspecific_spike_times_ms: tuple[float, ...] = ()
    cue_lead_trn_spike_indices: tuple[int, ...] = ()
    cue_lead_trn_spike_times_ms: tuple[float, ...] = ()
    cue_lead_relay_spike_indices: tuple[int, ...] = ()
    cue_lead_relay_spike_times_ms: tuple[float, ...] = ()
    v1_cortical_spike_times_ms: tuple[float, ...] = ()
    v2_layer4_spike_indices: tuple[int, ...] = ()
    v2_layer4_spike_times_ms: tuple[float, ...] = ()
    v2_relay_spike_indices: tuple[int, ...] = ()
    v2_relay_spike_times_ms: tuple[float, ...] = ()
    convention_fingerprint: str | None = None
    top_down_current_pA: float | None = None
    top_down_current_mode: str = TopDownCurrentMode.SUSTAINED_EPOCH.value
    top_down_current_event_limit: int | None = None
    top_down_current_termination_time_ms: float | None = None
    top_down_relay_source_indices: tuple[int, ...] | None = None
    top_down_cue_lead_ms: float = 0.0
    equilibration_ms: float = 0.0
    learned_state_provenance: str = "unspecified"
    comparator_relay_floor: float | None = None
    comparator_source_index: int | None = None
    comparator_transform: str | None = None
    comparator_target_count: int | None = None
    network_scope: str = "first_order"
    relay_top_down_ampa_peak_by_index: tuple[tuple[int, float], ...] = ()
    relay_top_down_ampa_integral_ms_by_index: tuple[tuple[int, float], ...] = ()
    relay_top_down_nmda_peak_by_index: tuple[tuple[int, float], ...] = ()
    relay_distal_voltage_range_mV_by_index: tuple[tuple[int, float, float], ...] = ()
    relay_proximal_voltage_range_mV_by_index: tuple[tuple[int, float, float], ...] = ()
    relay_soma_voltage_range_mV_by_index: tuple[tuple[int, float, float], ...] = ()
    relay_trn_gaba_peak_by_index: tuple[tuple[int, float], ...] = ()
    relay_trn_gaba_integral_ms_by_index: tuple[tuple[int, float], ...] = ()
    relay_driven_current_range_pA_by_index_and_source: tuple[
        tuple[int, str, float, float], ...
    ] = ()
    relay_event_current_samples_pA: tuple[
        tuple[int, float, str, float], ...
    ] = ()
    relay_pre_event_current_samples_pA: tuple[
        tuple[int, float, float, str, float], ...
    ] = ()
    relay_pre_event_voltage_samples_mV: tuple[
        tuple[int, float, float, str, float], ...
    ] = ()
    relay_pre_event_trn_gaba_gate_samples: tuple[
        tuple[int, float, float, float], ...
    ] = ()
    trn_layer6ii_ampa_peak_by_index: tuple[tuple[int, float], ...] = ()
    trn_layer6ii_nmda_peak_by_index: tuple[tuple[int, float], ...] = ()
    trn_relay_ampa_peak_by_index: tuple[tuple[int, float], ...] = ()
    trn_proximal_voltage_range_mV_by_index: tuple[tuple[int, float, float], ...] = ()
    trn_soma_voltage_range_mV_by_index: tuple[tuple[int, float, float], ...] = ()
    trn_post_startup_soma_voltage_range_mV_by_index: tuple[
        tuple[int, float, float], ...
    ] = ()
    trn_detector_voltage_range_mV_by_index: tuple[
        tuple[int, float, float], ...
    ] = ()
    trn_detector_post_first_event_voltage_range_mV_by_index: tuple[
        tuple[int, float, float], ...
    ] = ()
    trn_detector_threshold_upcrossings_by_index: tuple[tuple[int, int], ...] = ()
    trn_detector_zero_downcrossings_by_index: tuple[tuple[int, int], ...] = ()
    trn_detector_arm_transitions_by_index: tuple[tuple[int, int], ...] = ()
    trn_detector_release_transitions_by_index: tuple[tuple[int, int], ...] = ()
    trn_detector_final_armed_by_index: tuple[tuple[int, float], ...] = ()
    trn_driven_current_range_pA_by_index_and_source: tuple[
        tuple[int, str, float, float], ...
    ] = ()
    nonspecific_trn_gaba_peak: float | None = None
    nonspecific_trn_gaba_integral_ms: float | None = None
    nonspecific_post_startup_trn_gaba_peak: float | None = None
    nonspecific_layer6ii_ampa_peak: float | None = None
    nonspecific_layer6ii_nmda_peak: float | None = None
    nonspecific_direct_input_current_range_pA: tuple[float, float] | None = None
    nonspecific_trn_current_range_pA: tuple[float, float] | None = None
    nonspecific_layer6ii_current_range_pA: tuple[float, float] | None = None
    nonspecific_voltage_range_mV_by_compartment: tuple[
        tuple[str, float, float], ...
    ] = ()
    nonspecific_positive_soma_local_maxima_ms_mV: tuple[
        tuple[float, float], ...
    ] = ()
    nonspecific_positive_detector_local_maxima_ms_mV: tuple[
        tuple[float, float], ...
    ] = ()
    nonspecific_detector_voltage_range_mV: tuple[float, float] | None = None
    nonspecific_detector_threshold_upcrossings: int | None = None
    nonspecific_detector_zero_downcrossings: int | None = None
    nonspecific_detector_arm_transitions: int | None = None
    nonspecific_detector_release_transitions: int | None = None
    nonspecific_detector_final_armed: float | None = None

    def __post_init__(self) -> None:
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        if self.top_down_cue_lead_ms < 0:
            raise ValueError("top_down_cue_lead_ms cannot be negative")
        if self.equilibration_ms < 0:
            raise ValueError("equilibration_ms cannot be negative")
        TopDownCurrentMode(self.top_down_current_mode)
        if self.top_down_current_event_limit is not None and (
            self.top_down_current_event_limit < 1
        ):
            raise ValueError("top-down current event limit must be positive")
        if self.top_down_current_termination_time_ms is not None and not (
            0
            <= self.top_down_current_termination_time_ms
            <= self.top_down_cue_lead_ms + self.duration_ms
        ):
            raise ValueError("top-down current termination must lie within the cue/trial")
        if self.top_down_relay_source_indices is not None and (
            not self.top_down_relay_source_indices
            or any(not 0 <= index < 81 for index in self.top_down_relay_source_indices)
        ):
            raise ValueError("top-down relay source indices must address the 9x9 sheet")
        if self.comparator_relay_floor is not None and (
            not np.isfinite(self.comparator_relay_floor)
            or not 0.0 <= self.comparator_relay_floor <= 1.0
        ):
            raise ValueError("comparator relay floor must be finite and lie in [0, 1]")
        if self.comparator_source_index is not None and not (
            0 <= self.comparator_source_index < 81
        ):
            raise ValueError("comparator source index must address the 9x9 sheet")
        if self.comparator_target_count is not None and not (
            1 <= self.comparator_target_count <= 81
        ):
            raise ValueError("comparator target count must lie in [1, 81]")

    @property
    def nonspecific_rate_hz(self) -> float:
        return len(self.nonspecific_spike_times_ms) * 1000.0 / self.duration_ms


@dataclass(frozen=True, slots=True)
class Figure7ArousalAssessment:
    match_rate_hz: float
    mismatch_rate_hz: float
    duration_ms: float

    MATCH_TARGET_HZ = 40.0
    MISMATCH_TARGET_HZ = 70.0
    TARGET_DURATION_MS = 100.0

    @property
    def mismatch_disinhibition_pass(self) -> bool:
        return self.mismatch_rate_hz > self.match_rate_hz

    @property
    def target_duration_pass(self) -> bool:
        return self.duration_ms == self.TARGET_DURATION_MS

    @property
    def match_numeric_target_pass(self) -> bool:
        return self.match_rate_hz == self.MATCH_TARGET_HZ

    @property
    def mismatch_numeric_target_pass(self) -> bool:
        return self.mismatch_rate_hz == self.MISMATCH_TARGET_HZ

    @property
    def numeric_rate_pass(self) -> bool:
        return (
            self.target_duration_pass
            and self.match_numeric_target_pass
            and self.mismatch_numeric_target_pass
        )

    @property
    def reproduced_arousal(self) -> bool:
        return self.mismatch_disinhibition_pass and self.numeric_rate_pass


@dataclass(frozen=True, slots=True)
class Figure7PathwayAssessment:
    """Score the causal relay-to-TRN ordering stated in the Figure 7 caption.

    The publication does not report numeric relay or TRN counts.  It does state
    that a match lets more specific-thalamic cells fire, which in turn makes
    TRN inhibition stronger than during mismatch.  These are therefore
    directional gates, with absolute output rates retained only as diagnostics.
    """

    match_active_relay_cells: int
    mismatch_active_relay_cells: int
    match_trn_spikes: int
    mismatch_trn_spikes: int
    match_active_relay_indices: frozenset[int]
    mismatch_active_relay_indices: frozenset[int]

    @property
    def relay_subset_pass(self) -> bool:
        return self.match_active_relay_cells > self.mismatch_active_relay_cells

    @property
    def relay_spatial_match_pass(self) -> bool:
        return self.match_active_relay_indices == FIGURE7_MATCH_RELAY_INDICES

    @property
    def relay_mismatch_overlap_pass(self) -> bool:
        return (
            bool(self.mismatch_active_relay_indices)
            and self.mismatch_active_relay_indices <= FIGURE7_RELAY_OVERLAP_INDICES
        )

    @property
    def trn_order_pass(self) -> bool:
        return self.match_trn_spikes > self.mismatch_trn_spikes

    @property
    def reproduced_pathway(self) -> bool:
        return (
            self.relay_subset_pass
            and self.relay_spatial_match_pass
            and self.relay_mismatch_overlap_pass
            and self.trn_order_pass
        )


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
) -> Figure7ArousalAssessment:
    """Score Figure 7c's exact 100-ms 40-Hz/70-Hz arousal traces."""

    _validate_figure7_pair(match, mismatch)
    return Figure7ArousalAssessment(
        match_rate_hz=match.nonspecific_rate_hz,
        mismatch_rate_hz=mismatch.nonspecific_rate_hz,
        duration_ms=match.duration_ms,
    )


def assess_figure7_pathway(
    match: Figure7ConditionResult,
    mismatch: Figure7ConditionResult,
) -> Figure7PathwayAssessment:
    """Score the caption's match-greater-than-mismatch relay/TRN mechanism."""

    _validate_figure7_pair(match, mismatch)
    match_indices = frozenset(match.relay_spike_indices)
    mismatch_indices = frozenset(mismatch.relay_spike_indices)
    return Figure7PathwayAssessment(
        match_active_relay_cells=len(match_indices),
        mismatch_active_relay_cells=len(mismatch_indices),
        match_trn_spikes=len(match.trn_spike_times_ms),
        mismatch_trn_spikes=len(mismatch.trn_spike_times_ms),
        match_active_relay_indices=match_indices,
        mismatch_active_relay_indices=mismatch_indices,
    )


def assess_figure7_reproduction(
    match: Figure7ConditionResult,
    mismatch: Figure7ConditionResult,
) -> Figure7ReproductionAssessment:
    """Require both directional arousal and the stated relay/TRN mechanism."""

    return Figure7ReproductionAssessment(
        arousal=assess_figure7_arousal(match, mismatch),
        pathway=assess_figure7_pathway(match, mismatch),
    )


def run_figure7_condition(
    *,
    condition: MatchCondition,
    top_down_current_pA: float,
    learned_weights: Mapping[str, tuple[float, ...] | np.ndarray] | None = None,
    use_paper_constrained_reference: bool = False,
    pretrain_with_figure6_episode: bool = False,
    paper_reference: Figure6ReferenceExpectation | None = None,
    conventions=None,
    duration_ms: float = 100.0,
    dt_ms: float = 0.01,
    exact_relay_voltage_clamp: bool = False,
    relay_clamp_compartment: str = "proximal_dendrite",
    include_higher_order_loop: bool = False,
    record_relay_diagnostics: bool = False,
    relay_pre_event_offsets_ms: tuple[float, ...] = (),
    record_v1_cortical_spikes: bool = False,
    projection_weight_scales: Mapping[str, float] | None = None,
    persistent_projection_weight_scales: Mapping[str, float] | None = None,
    top_down_relay_source_indices: frozenset[int] | None = None,
    comparator_relay_floor: float | None = None,
    comparator_half_max_gate: bool = False,
    comparator_top_k_targets: int | None = None,
    comparator_source_index: int = 40,
    top_down_current_mode: TopDownCurrentMode | str = (
        TopDownCurrentMode.SUSTAINED_EPOCH
    ),
    top_down_current_event_limit: int | None = None,
    top_down_cue_lead_ms: float = 0.0,
    equilibration_ms: float = 0.0,
    cpp_standalone_directory: str | Path | None = None,
    brian=None,
) -> Figure7ConditionResult:
    """Run one source-labeled Figure 7 match or mismatch condition."""

    learning_sources = sum(
        (
            learned_weights is not None,
            use_paper_constrained_reference,
            pretrain_with_figure6_episode,
        )
    )
    if learning_sources != 1:
        raise ValueError(
            "Figure 7 requires exactly one learned-weight snapshot, paper-constrained "
            "reference, or same-network Figure 6 episode"
        )
    if duration_ms <= 0 or dt_ms <= 0:
        raise ValueError("duration_ms and dt_ms must be positive")
    if top_down_cue_lead_ms < 0:
        raise ValueError("top_down_cue_lead_ms cannot be negative")
    if equilibration_ms < 0:
        raise ValueError("equilibration_ms cannot be negative")
    current_mode = TopDownCurrentMode(top_down_current_mode)
    if current_mode is TopDownCurrentMode.UNTIL_CUED_CELL_EVENT_LIMIT:
        if top_down_current_event_limit is None or top_down_current_event_limit < 1:
            raise ValueError("event-limited top-down current requires a positive limit")
    elif top_down_current_event_limit is not None:
        raise ValueError("top-down current event limit requires event-limited mode")
    if top_down_relay_source_indices is not None and cpp_standalone_directory is not None:
        raise ValueError("selected-category source masking is a numpy diagnostic only")
    if comparator_relay_floor is not None and cpp_standalone_directory is not None:
        raise ValueError("the reconstructed comparator is a numpy calibration only")
    if (
        comparator_half_max_gate or comparator_top_k_targets is not None
    ) and cpp_standalone_directory is not None:
        raise ValueError("the reconstructed comparator is a numpy calibration only")
    comparator_transform_count = sum(
        (
            comparator_relay_floor is not None,
            comparator_half_max_gate,
            comparator_top_k_targets is not None,
        )
    )
    if comparator_transform_count > 1:
        raise ValueError("select only one reconstructed comparator transform")
    if (
        comparator_transform_count
    ) and pretrain_with_figure6_episode:
        raise ValueError(
            "the reconstructed comparator requires an explicit learned-weight snapshot"
        )
    if record_relay_diagnostics and duration_ms <= 45.0:
        raise ValueError("Figure 7 pathway diagnostics require duration_ms > 45")
    if relay_pre_event_offsets_ms and not record_relay_diagnostics:
        raise ValueError("pre-event samples require relay diagnostics")
    if (
        any(not np.isfinite(offset) or offset <= 0 for offset in relay_pre_event_offsets_ms)
        or len(set(relay_pre_event_offsets_ms)) != len(relay_pre_event_offsets_ms)
    ):
        raise ValueError("pre-event offsets must be unique, finite, and positive")
    overlapping_scales = set(projection_weight_scales or ()) & set(
        persistent_projection_weight_scales or ()
    )
    if overlapping_scales:
        raise ValueError(
            "projection scale mappings overlap: "
            f"{sorted(overlapping_scales)}"
        )
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
    if exact_relay_voltage_clamp and pretrain_with_figure6_episode:
        raise ValueError("same-network Figure 6 pretraining requires an unclamped relay")
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

    def apply_projection_scales(scales: Mapping[str, float] | None) -> None:
        if not scales:
            return
        unknown = set(scales) - set(sector.projections)
        if unknown:
            raise ValueError(f"unknown projection scale IDs: {sorted(unknown)}")
        for projection_id, scale in scales.items():
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("projection weight scales must be finite and positive")
            projection = sector.projections[projection_id]
            blocks = getattr(projection, "blocks", (projection,))
            for block in blocks:
                # Symbolic assignment also works before a deferred standalone
                # build, when state arrays cannot yet be read.
                block.w = f"w*({float(scale)!r})"

    # Persistent scales define one altered network and therefore precede an
    # optional same-network Figure 6 episode. The historical post-learning
    # scales below remain available for recognition-only diagnostics.
    apply_projection_scales(persistent_projection_weight_scales)
    pretraining_elapsed_ms = 0.0
    if pretrain_with_figure6_episode:
        from .figure6 import Figure6LearningProtocol

        training = Figure6LearningProtocol()
        category_group = sector.populations["layer6ii_excitatory_v1"].group
        if training.layer6ii_ahp_scale != 1.0:
            # Keep this assignment symbolic: C++ standalone cannot read a
            # state array before the deferred build has executed.
            category_group.g_ahp_max = (
                f"g_ahp_max*({float(training.layer6ii_ahp_scale)!r})"
            )
        if training.warmup_ms:
            sector.network.run(training.warmup_ms * brian.ms)
        training_stimulus = ClassicBarStimulus(
            BarOrientation.HORIZONTAL,
            duration_ms=training.stimulus_ms,
            source_value=training.source_value,
            category_source_value=training.category_source_value,
            include_archived_category_pixel=True,
        )
        apply_bar_stimulus(sector, training_stimulus)
        sector.network.run(training.stimulus_ms * brian.ms)
        clear_bar_stimulus(sector, training_stimulus)
        if training.post_stimulus_ms:
            sector.network.run(training.post_stimulus_ms * brian.ms)
        pretraining_elapsed_ms = (
            training.warmup_ms + training.stimulus_ms + training.post_stimulus_ms
        )
        for projection_id in FIGURE7_REQUIRED_LEARNED_PROJECTIONS:
            sector.projections[projection_id].modifiable = 0
        provenance = "same-network-figure6-episode"
    elif use_paper_constrained_reference:
        learned_weights = paper_constrained_figure6_expectation(
            sector.projections,
            paper_reference,
            derive_from_source=cpp_standalone_directory is not None,
        )
        provenance = "paper-constrained-figure6c-reference"
    else:
        provenance = "simulated-learned-weight-snapshot"
    if not pretrain_with_figure6_episode:
        assert learned_weights is not None
        apply_figure7_learned_state(
            sector.projections,
            learned_weights,
            verify_runtime_bounds=cpp_standalone_directory is None,
        )
    relay_input_gains = None
    if comparator_relay_floor is not None:
        assert learned_weights is not None
        relay_input_gains = comparator_relay_input_gains(
            learned_weights,
            floor=comparator_relay_floor,
            source_index=comparator_source_index,
        )
    elif comparator_half_max_gate:
        assert learned_weights is not None
        relay_input_gains = half_max_comparator_relay_input_gains(
            learned_weights,
            source_index=comparator_source_index,
        )
    elif comparator_top_k_targets is not None:
        assert learned_weights is not None
        relay_input_gains = top_k_comparator_relay_input_gains(
            learned_weights,
            target_count=comparator_top_k_targets,
            source_index=comparator_source_index,
        )
    apply_projection_scales(projection_weight_scales)
    if top_down_relay_source_indices is not None:
        restrict_figure7_top_down_relay_sources(
            sector.projections, top_down_relay_source_indices
        )
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
    trn_calcium_variables: tuple[str, ...] = ()
    nonspecific_state = None
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
                "i_port_000",
                "i_port_001",
                "i_port_002",
                "i_port_003",
                "i_port_004",
                "i_port_005",
                "i_port_006",
                "i_port_007",
                "i_external_001",
                "i_ca_distal_dendrite",
                "i_ca_proximal_dendrite",
                "i_axial_inward_soma",
                "i_na_soma",
                "i_k_soma",
                "v_distal_dendrite",
                "v_proximal_dendrite",
                "v_soma",
            ),
            record=FIGURE7_RELAY_DIAGNOSTIC_INDICES,
            name=f"figure7_{condition.value}_relay_pathway_state",
        )
        trn_group = sector.populations["trn"].group
        trn_calcium_variables = tuple(
            sorted(name for name in trn_group.variables if name.startswith("i_ca_"))
        )
        trn_state = brian.StateMonitor(
            trn_group,
            (
                "port_001_gate",
                "port_002_gate",
                "port_004_gate",
                "i_port_000",
                "i_port_001",
                "i_port_002",
                "i_port_003",
                "i_port_004",
                "i_axial_inward_soma",
                "i_na_soma",
                "i_k_soma",
                "v_proximal_dendrite",
                "v_soma",
                "spike_detector_voltage",
                "armed",
                *trn_calcium_variables,
            ),
            record=FIGURE7_RELAY_DIAGNOSTIC_INDICES,
            name=f"figure7_{condition.value}_trn_pathway_state",
        )
        nonspecific_state = brian.StateMonitor(
            sector.populations["thalamic_nonspecific"].group,
            (
                "port_000_gate",
                "port_001_gate",
                "port_002_gate",
                "port_003_gate",
                "port_004_gate",
                "i_port_000",
                "i_port_001",
                "i_port_002",
                "i_port_003",
                "i_port_004",
                "i_external_001",
                "v_distal_dendrite",
                "v_proximal_dendrite",
                "v_soma",
                "spike_detector_voltage",
                "armed",
            ),
            record=True,
            name=f"figure7_{condition.value}_nonspecific_pathway_state",
        )
        monitors.extend((relay_state, trn_state, nonspecific_state))
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
    if current_mode is TopDownCurrentMode.UNTIL_CUED_CELL_FIRST_EVENT:
        category_group = sector.populations[cue.top_down_population].group
        category_group.clear_drive_on_spike = 0
        category_group.clear_drive_on_spike[cue.top_down_cell_index] = 1
    elif current_mode is TopDownCurrentMode.UNTIL_CUED_CELL_EVENT_LIMIT:
        category_group = sector.populations[cue.top_down_population].group
        category_group.drive_spikes_until_clear = 0
        category_group.drive_spikes_until_clear[cue.top_down_cell_index] = int(
            top_down_current_event_limit
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
            relay_input_gains=relay_input_gains,
        )
    else:
        apply_match_mismatch_cue(
            sector,
            cue,
            apply_relay_input=not exact_relay_voltage_clamp,
            relay_input_gains=relay_input_gains,
            brian=brian,
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
    relay_driven_current_range: tuple[tuple[int, str, float, float], ...] = ()
    relay_event_current_samples: tuple[tuple[int, float, str, float], ...] = ()
    relay_pre_event_current_samples: tuple[
        tuple[int, float, float, str, float], ...
    ] = ()
    relay_pre_event_voltage_samples: tuple[
        tuple[int, float, float, str, float], ...
    ] = ()
    relay_pre_event_trn_gaba_gate_samples: tuple[
        tuple[int, float, float, float], ...
    ] = ()
    trn_layer6ii_ampa_peak: tuple[tuple[int, float], ...] = ()
    trn_layer6ii_nmda_peak: tuple[tuple[int, float], ...] = ()
    trn_relay_ampa_peak: tuple[tuple[int, float], ...] = ()
    trn_voltage_range: tuple[tuple[int, float, float], ...] = ()
    trn_soma_voltage_range: tuple[tuple[int, float, float], ...] = ()
    trn_post_startup_soma_voltage_range: tuple[tuple[int, float, float], ...] = ()
    trn_detector_voltage_range: tuple[tuple[int, float, float], ...] = ()
    trn_detector_post_first_event_voltage_range: tuple[
        tuple[int, float, float], ...
    ] = ()
    trn_detector_threshold_upcrossings: tuple[tuple[int, int], ...] = ()
    trn_detector_zero_downcrossings: tuple[tuple[int, int], ...] = ()
    trn_detector_arm_transitions: tuple[tuple[int, int], ...] = ()
    trn_detector_release_transitions: tuple[tuple[int, int], ...] = ()
    trn_detector_final_armed: tuple[tuple[int, float], ...] = ()
    trn_driven_current_range: tuple[tuple[int, str, float, float], ...] = ()
    nonspecific_trn_gaba_peak = None
    nonspecific_trn_gaba_integral_ms = None
    nonspecific_post_startup_trn_gaba_peak = None
    nonspecific_layer6ii_ampa_peak = None
    nonspecific_layer6ii_nmda_peak = None
    nonspecific_direct_input_current_range_pA = None
    nonspecific_trn_current_range_pA = None
    nonspecific_layer6ii_current_range_pA = None
    nonspecific_voltage_range_mV_by_compartment: tuple[
        tuple[str, float, float], ...
    ] = ()
    nonspecific_positive_soma_local_maxima_ms_mV: tuple[
        tuple[float, float], ...
    ] = ()
    nonspecific_positive_detector_local_maxima_ms_mV: tuple[
        tuple[float, float], ...
    ] = ()
    nonspecific_detector_voltage_range_mV = None
    nonspecific_detector_threshold_upcrossings = None
    nonspecific_detector_zero_downcrossings = None
    nonspecific_detector_arm_transitions = None
    nonspecific_detector_release_transitions = None
    nonspecific_detector_final_armed = None
    if relay_state is not None:
        ampa = np.asarray(relay_state.port_005_gate) + np.asarray(
            relay_state.port_007_gate
        )
        nmda = np.asarray(relay_state.port_003_gate) + np.asarray(
            relay_state.port_006_gate
        )
        voltage_mV = np.asarray(relay_state.v_distal_dendrite / brian.mV)
        times_ms = np.asarray(relay_state.t / brian.ms)
        stimulus_start_ms = (
            pretraining_elapsed_ms + equilibration_ms + top_down_cue_lead_ms
        )
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
        relay_driven_window = times_ms >= 40.0
        relay_current_sources_pA = {
            "direct_image_input": np.asarray(relay_state.i_external_001 / brian.pA)[
                :, diagnostic_window
            ],
            "top_down_excitation": sum(
                np.asarray(getattr(relay_state, f"i_port_{index:03d}") / brian.pA)[
                    :, diagnostic_window
                ]
                for index in (3, 5, 6, 7)
            ),
            "trn_gaba": sum(
                np.asarray(getattr(relay_state, f"i_port_{index:03d}") / brian.pA)[
                    :, diagnostic_window
                ]
                for index in (0, 1, 4)
            ),
            "interneuron_gaba": np.asarray(relay_state.i_port_002 / brian.pA)[
                :, diagnostic_window
            ],
            "distal_calcium": np.asarray(relay_state.i_ca_distal_dendrite / brian.pA)[
                :, diagnostic_window
            ],
            "proximal_calcium": np.asarray(
                relay_state.i_ca_proximal_dendrite / brian.pA
            )[:, diagnostic_window],
            "soma_axial": np.asarray(relay_state.i_axial_inward_soma / brian.pA)[
                :, diagnostic_window
            ],
            "soma_sodium": np.asarray(relay_state.i_na_soma / brian.pA)[
                :, diagnostic_window
            ],
            "soma_potassium": np.asarray(relay_state.i_k_soma / brian.pA)[
                :, diagnostic_window
            ],
        }
        relay_driven_current_range = tuple(
            (index, source, float(np.min(values)), float(np.max(values)))
            for source, traces in relay_current_sources_pA.items()
            for index, values in zip(
                FIGURE7_RELAY_DIAGNOSTIC_INDICES,
                traces[:, relay_driven_window],
                strict=True,
            )
        )
        diagnostic_row = {
            index: row for row, index in enumerate(FIGURE7_RELAY_DIAGNOSTIC_INDICES)
        }
        relay_event_current_samples = tuple(
            (
                event_index,
                event_time_ms,
                source,
                float(traces[diagnostic_row[event_index], sample]),
            )
            for event_index, absolute_event_time_ms in zip(
                relay.i, relay.t / brian.ms, strict=True
            )
            if int(event_index) in diagnostic_row
            and float(absolute_event_time_ms) >= stimulus_start_ms
            for event_time_ms in [float(absolute_event_time_ms) - stimulus_start_ms]
            for sample in [int(np.argmin(np.abs(times_ms - event_time_ms)))]
            for source, traces in relay_current_sources_pA.items()
        )
        relay_voltage_sources_mV = {
            "distal_dendrite": voltage_mV,
            "proximal_dendrite": proximal_voltage_mV,
            "soma": soma_voltage_mV,
        }
        relay_pre_event_current_samples = tuple(
            (
                event_index,
                event_time_ms,
                float(offset_ms),
                source,
                float(traces[diagnostic_row[event_index], sample]),
            )
            for event_index, absolute_event_time_ms in zip(
                relay.i, relay.t / brian.ms, strict=True
            )
            if int(event_index) in diagnostic_row
            and float(absolute_event_time_ms) >= stimulus_start_ms
            for event_time_ms in [float(absolute_event_time_ms) - stimulus_start_ms]
            for offset_ms in relay_pre_event_offsets_ms
            for sample in [
                int(np.argmin(np.abs(times_ms - (event_time_ms - offset_ms))))
            ]
            for source, traces in relay_current_sources_pA.items()
        )
        relay_pre_event_voltage_samples = tuple(
            (
                event_index,
                event_time_ms,
                float(offset_ms),
                source,
                float(traces[diagnostic_row[event_index], sample]),
            )
            for event_index, absolute_event_time_ms in zip(
                relay.i, relay.t / brian.ms, strict=True
            )
            if int(event_index) in diagnostic_row
            and float(absolute_event_time_ms) >= stimulus_start_ms
            for event_time_ms in [float(absolute_event_time_ms) - stimulus_start_ms]
            for offset_ms in relay_pre_event_offsets_ms
            for sample in [
                int(np.argmin(np.abs(times_ms - (event_time_ms - offset_ms))))
            ]
            for source, traces in relay_voltage_sources_mV.items()
        )
        relay_pre_event_trn_gaba_gate_samples = tuple(
            (
                event_index,
                event_time_ms,
                float(offset_ms),
                float(relay_trn_gaba[diagnostic_row[event_index], sample]),
            )
            for event_index, absolute_event_time_ms in zip(
                relay.i, relay.t / brian.ms, strict=True
            )
            if int(event_index) in diagnostic_row
            and float(absolute_event_time_ms) >= stimulus_start_ms
            for event_time_ms in [float(absolute_event_time_ms) - stimulus_start_ms]
            for offset_ms in relay_pre_event_offsets_ms
            for sample in [
                int(np.argmin(np.abs(times_ms - (event_time_ms - offset_ms))))
            ]
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
        trn_detector_voltage_mV = np.asarray(
            trn_state.spike_detector_voltage / brian.mV
        )[:, diagnostic_window]
        trn_detector_armed = np.asarray(trn_state.armed)[:, diagnostic_window]
        trn_detector_voltage_range = tuple(
            (index, float(np.min(values)), float(np.max(values)))
            for index, values in zip(
                FIGURE7_RELAY_DIAGNOSTIC_INDICES,
                trn_detector_voltage_mV,
                strict=True,
            )
        )
        detector_threshold_mV = (
            conventions.trn_spike_event_threshold_mV
            if conventions.trn_spike_event_threshold_mV is not None
            else conventions.spike_event_threshold_mV
        )
        trn_detector_threshold_upcrossings = tuple(
            (
                index,
                int(
                    np.count_nonzero(
                        (values[:-1] <= detector_threshold_mV)
                        & (values[1:] > detector_threshold_mV)
                    )
                ),
            )
            for index, values in zip(
                FIGURE7_RELAY_DIAGNOSTIC_INDICES,
                trn_detector_voltage_mV,
                strict=True,
            )
        )
        trn_detector_zero_downcrossings = tuple(
            (
                index,
                int(
                    np.count_nonzero(
                        (values[:-1] >= 0.0) & (values[1:] < 0.0)
                    )
                ),
            )
            for index, values in zip(
                FIGURE7_RELAY_DIAGNOSTIC_INDICES,
                trn_detector_voltage_mV,
                strict=True,
            )
        )
        trn_detector_arm_transitions = tuple(
            (
                index,
                int(
                    np.count_nonzero(
                        (values[:-1] <= 0.5) & (values[1:] > 0.5)
                    )
                ),
            )
            for index, values in zip(
                FIGURE7_RELAY_DIAGNOSTIC_INDICES,
                trn_detector_armed,
                strict=True,
            )
        )
        trn_detector_release_transitions = tuple(
            (
                index,
                int(
                    np.count_nonzero(
                        (values[:-1] > 0.5) & (values[1:] <= 0.5)
                    )
                ),
            )
            for index, values in zip(
                FIGURE7_RELAY_DIAGNOSTIC_INDICES,
                trn_detector_armed,
                strict=True,
            )
        )
        trn_detector_final_armed = tuple(
            (index, float(values[-1]))
            for index, values in zip(
                FIGURE7_RELAY_DIAGNOSTIC_INDICES,
                trn_detector_armed,
                strict=True,
            )
        )
        trn_detector_post_first_event_voltage_range = tuple(
            (
                index,
                float(np.min(trn_detector_voltage_mV[row, times_ms > first_event_ms])),
                float(np.max(trn_detector_voltage_mV[row, times_ms > first_event_ms])),
            )
            for row, index in enumerate(FIGURE7_RELAY_DIAGNOSTIC_INDICES)
            for first_event_ms in [
                next(
                    (
                        float(value - stimulus_start_ms)
                        for event_index, value in zip(
                            trn.i, trn.t / brian.ms, strict=True
                        )
                        if int(event_index) == index
                        and float(value) >= stimulus_start_ms
                    ),
                    None,
                )
            ]
            if first_event_ms is not None
            and np.any(times_ms > first_event_ms)
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
        driven_window = times_ms >= 45.0
        current_sources_pA = {
            "relay_ampa": np.asarray(trn_state.i_port_002 / brian.pA)[
                :, diagnostic_window
            ],
            "layer6ii_excitation": (
                np.asarray(trn_state.i_port_001 / brian.pA)[:, diagnostic_window]
                + np.asarray(trn_state.i_port_004 / brian.pA)[:, diagnostic_window]
            ),
            "recurrent_gaba": (
                np.asarray(trn_state.i_port_000 / brian.pA)[:, diagnostic_window]
                + np.asarray(trn_state.i_port_003 / brian.pA)[:, diagnostic_window]
            ),
            "soma_axial": np.asarray(trn_state.i_axial_inward_soma / brian.pA)[
                :, diagnostic_window
            ],
            "soma_sodium": np.asarray(trn_state.i_na_soma / brian.pA)[
                :, diagnostic_window
            ],
            "soma_potassium": np.asarray(trn_state.i_k_soma / brian.pA)[
                :, diagnostic_window
            ],
        }
        for variable in trn_calcium_variables:
            compartment = variable.removeprefix("i_ca_").removesuffix("_dendrite")
            current_sources_pA[f"{compartment}_calcium"] = np.asarray(
                getattr(trn_state, variable) / brian.pA
            )[:, diagnostic_window]
        trn_driven_current_range = tuple(
            (index, source, float(np.min(values)), float(np.max(values)))
            for source, traces in current_sources_pA.items()
            for index, values in zip(
                FIGURE7_RELAY_DIAGNOSTIC_INDICES,
                traces[:, driven_window],
                strict=True,
            )
        )
    if nonspecific_state is not None:
        nonspecific_trn_gaba = sum(
            np.asarray(getattr(nonspecific_state, f"port_{index:03d}_gate"))[0]
            for index in range(3)
        )[diagnostic_window]
        nonspecific_layer6ii_ampa = np.asarray(nonspecific_state.port_003_gate)[0][
            diagnostic_window
        ]
        nonspecific_layer6ii_nmda = np.asarray(nonspecific_state.port_004_gate)[0][
            diagnostic_window
        ]
        nonspecific_trn_current_pA = sum(
            np.asarray(getattr(nonspecific_state, f"i_port_{index:03d}") / brian.pA)[0]
            for index in range(3)
        )[diagnostic_window]
        nonspecific_layer6ii_current_pA = sum(
            np.asarray(getattr(nonspecific_state, f"i_port_{index:03d}") / brian.pA)[0]
            for index in (3, 4)
        )[diagnostic_window]
        nonspecific_direct_input_pA = np.asarray(
            nonspecific_state.i_external_001 / brian.pA
        )[0][diagnostic_window]
        nonspecific_trn_gaba_peak = float(np.max(nonspecific_trn_gaba))
        nonspecific_trn_gaba_integral_ms = float(
            np.trapz(nonspecific_trn_gaba, times_ms)
        )
        nonspecific_post_startup_trn_gaba_peak = float(
            np.max(nonspecific_trn_gaba[post_startup_window])
        )
        nonspecific_layer6ii_ampa_peak = float(np.max(nonspecific_layer6ii_ampa))
        nonspecific_layer6ii_nmda_peak = float(np.max(nonspecific_layer6ii_nmda))
        nonspecific_direct_input_current_range_pA = (
            float(np.min(nonspecific_direct_input_pA)),
            float(np.max(nonspecific_direct_input_pA)),
        )
        nonspecific_trn_current_range_pA = (
            float(np.min(nonspecific_trn_current_pA)),
            float(np.max(nonspecific_trn_current_pA)),
        )
        nonspecific_layer6ii_current_range_pA = (
            float(np.min(nonspecific_layer6ii_current_pA)),
            float(np.max(nonspecific_layer6ii_current_pA)),
        )
        nonspecific_voltage_range_mV_by_compartment = tuple(
            (
                compartment,
                float(np.min(values)),
                float(np.max(values)),
            )
            for compartment, values in (
                (
                    "distal_dendrite",
                    np.asarray(nonspecific_state.v_distal_dendrite / brian.mV)[0][
                        diagnostic_window
                    ],
                ),
                (
                    "proximal_dendrite",
                    np.asarray(nonspecific_state.v_proximal_dendrite / brian.mV)[0][
                        diagnostic_window
                    ],
                ),
                (
                    "soma",
                    np.asarray(nonspecific_state.v_soma / brian.mV)[0][diagnostic_window],
                ),
            )
        )
        nonspecific_soma_voltage_mV = np.asarray(
            nonspecific_state.v_soma / brian.mV
        )[0][diagnostic_window]
        nonspecific_detector_voltage_mV = np.asarray(
            nonspecific_state.spike_detector_voltage / brian.mV
        )[0][diagnostic_window]
        nonspecific_detector_armed = np.asarray(nonspecific_state.armed)[0][
            diagnostic_window
        ]

        def positive_local_maxima(
            values: np.ndarray,
        ) -> tuple[tuple[float, float], ...]:
            peak_indices = np.flatnonzero(
                (values[1:-1] > values[:-2])
                & (values[1:-1] >= values[2:])
                & (values[1:-1] > 0.0)
            ) + 1
            return tuple(
                (float(times_ms[index]), float(values[index]))
                for index in peak_indices
            )

        nonspecific_positive_soma_local_maxima_ms_mV = positive_local_maxima(
            nonspecific_soma_voltage_mV
        )
        nonspecific_positive_detector_local_maxima_ms_mV = positive_local_maxima(
            nonspecific_detector_voltage_mV
        )
        nonspecific_detector_voltage_range_mV = (
            float(np.min(nonspecific_detector_voltage_mV)),
            float(np.max(nonspecific_detector_voltage_mV)),
        )
        detector_threshold_mV = conventions.spike_event_threshold_mV
        nonspecific_detector_threshold_upcrossings = int(
            np.count_nonzero(
                (nonspecific_detector_voltage_mV[:-1] <= detector_threshold_mV)
                & (nonspecific_detector_voltage_mV[1:] > detector_threshold_mV)
            )
        )
        nonspecific_detector_zero_downcrossings = int(
            np.count_nonzero(
                (nonspecific_detector_voltage_mV[:-1] >= 0.0)
                & (nonspecific_detector_voltage_mV[1:] < 0.0)
            )
        )
        nonspecific_detector_arm_transitions = int(
            np.count_nonzero(
                (nonspecific_detector_armed[:-1] <= 0.5)
                & (nonspecific_detector_armed[1:] > 0.5)
            )
        )
        nonspecific_detector_release_transitions = int(
            np.count_nonzero(
                (nonspecific_detector_armed[:-1] > 0.5)
                & (nonspecific_detector_armed[1:] <= 0.5)
            )
        )
        nonspecific_detector_final_armed = float(nonspecific_detector_armed[-1])
    def stimulus_times(monitor) -> tuple[float, ...]:
        stimulus_start_ms = (
            pretraining_elapsed_ms + equilibration_ms + top_down_cue_lead_ms
        )
        return tuple(
            float(value - stimulus_start_ms)
            for value in monitor.t / brian.ms
            if float(value) >= stimulus_start_ms
        )

    def stimulus_indices(monitor) -> tuple[int, ...]:
        stimulus_start_ms = (
            pretraining_elapsed_ms + equilibration_ms + top_down_cue_lead_ms
        )
        return tuple(
            int(index)
            for index, value in zip(monitor.i, monitor.t / brian.ms, strict=True)
            if float(value) >= stimulus_start_ms
        )

    def cue_lead_times(monitor) -> tuple[float, ...]:
        cue_start_ms = pretraining_elapsed_ms + equilibration_ms
        cue_end_ms = cue_start_ms + top_down_cue_lead_ms
        return tuple(
            float(value - cue_start_ms)
            for value in monitor.t / brian.ms
            if cue_start_ms <= float(value) < cue_end_ms
        )

    def cue_lead_indices(monitor) -> tuple[int, ...]:
        cue_start_ms = pretraining_elapsed_ms + equilibration_ms
        cue_end_ms = cue_start_ms + top_down_cue_lead_ms
        return tuple(
            int(index)
            for index, value in zip(monitor.i, monitor.t / brian.ms, strict=True)
            if cue_start_ms <= float(value) < cue_end_ms
        )

    def equilibration_times(monitor) -> tuple[float, ...]:
        start_ms = pretraining_elapsed_ms
        end_ms = start_ms + equilibration_ms
        return tuple(
            float(value - start_ms)
            for value in monitor.t / brian.ms
            if start_ms <= float(value) < end_ms
        )

    def equilibration_indices(monitor) -> tuple[int, ...]:
        start_ms = pretraining_elapsed_ms
        end_ms = start_ms + equilibration_ms
        return tuple(
            int(index)
            for index, value in zip(monitor.i, monitor.t / brian.ms, strict=True)
            if start_ms <= float(value) < end_ms
        )

    category_stimulus_indices = stimulus_indices(category)
    category_stimulus_times = stimulus_times(category)
    current_termination_time_ms = None
    if current_mode in {
        TopDownCurrentMode.UNTIL_CUED_CELL_FIRST_EVENT,
        TopDownCurrentMode.UNTIL_CUED_CELL_EVENT_LIMIT,
    }:
        termination_event_limit = (
            1
            if current_mode is TopDownCurrentMode.UNTIL_CUED_CELL_FIRST_EVENT
            else int(top_down_current_event_limit)
        )
        selected_event_times_from_cue_start = [
            time
            for index, time in zip(
                cue_lead_indices(category), cue_lead_times(category), strict=True
            )
            if index == cue.top_down_cell_index
        ] + [
            top_down_cue_lead_ms + time
            for index, time in zip(
                category_stimulus_indices, category_stimulus_times, strict=True
            )
            if index == cue.top_down_cell_index
        ]
        if len(selected_event_times_from_cue_start) >= termination_event_limit:
            current_termination_time_ms = selected_event_times_from_cue_start[
                termination_event_limit - 1
            ]
    result = Figure7ConditionResult(
        condition=condition,
        duration_ms=duration_ms,
        nonspecific_spike_times_ms=stimulus_times(nonspecific),
        layer4_spike_indices=stimulus_indices(layer4),
        layer4_spike_times_ms=stimulus_times(layer4),
        relay_spike_indices=stimulus_indices(relay),
        relay_spike_times_ms=stimulus_times(relay),
        trn_spike_indices=stimulus_indices(trn),
        trn_spike_times_ms=stimulus_times(trn),
        category_spike_indices=category_stimulus_indices,
        category_spike_times_ms=category_stimulus_times,
        equilibration_nonspecific_spike_times_ms=equilibration_times(nonspecific),
        equilibration_layer4_spike_indices=equilibration_indices(layer4),
        equilibration_layer4_spike_times_ms=equilibration_times(layer4),
        equilibration_relay_spike_indices=equilibration_indices(relay),
        equilibration_relay_spike_times_ms=equilibration_times(relay),
        equilibration_trn_spike_indices=equilibration_indices(trn),
        equilibration_trn_spike_times_ms=equilibration_times(trn),
        equilibration_category_spike_indices=equilibration_indices(category),
        equilibration_category_spike_times_ms=equilibration_times(category),
        cue_lead_category_spike_indices=cue_lead_indices(category),
        cue_lead_category_spike_times_ms=cue_lead_times(category),
        cue_lead_nonspecific_spike_times_ms=cue_lead_times(nonspecific),
        cue_lead_trn_spike_indices=cue_lead_indices(trn),
        cue_lead_trn_spike_times_ms=cue_lead_times(trn),
        cue_lead_relay_spike_indices=cue_lead_indices(relay),
        cue_lead_relay_spike_times_ms=cue_lead_times(relay),
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
        top_down_current_mode=current_mode.value,
        top_down_current_event_limit=top_down_current_event_limit,
        top_down_current_termination_time_ms=current_termination_time_ms,
        top_down_relay_source_indices=(
            None
            if top_down_relay_source_indices is None
            else tuple(sorted(top_down_relay_source_indices))
        ),
        top_down_cue_lead_ms=top_down_cue_lead_ms,
        equilibration_ms=equilibration_ms,
        learned_state_provenance=provenance,
        comparator_relay_floor=comparator_relay_floor,
        comparator_source_index=(
            comparator_source_index
            if comparator_transform_count
            else None
        ),
        comparator_transform=(
            "linear_floor"
            if comparator_relay_floor is not None
            else "half_max_binary"
            if comparator_half_max_gate
            else "top_k_binary"
            if comparator_top_k_targets is not None
            else None
        ),
        comparator_target_count=comparator_top_k_targets,
        network_scope="full_two_area" if include_higher_order_loop else "first_order",
        relay_top_down_ampa_peak_by_index=ampa_peak,
        relay_top_down_ampa_integral_ms_by_index=ampa_integral,
        relay_top_down_nmda_peak_by_index=nmda_peak,
        relay_distal_voltage_range_mV_by_index=voltage_range,
        relay_proximal_voltage_range_mV_by_index=proximal_voltage_range,
        relay_soma_voltage_range_mV_by_index=soma_voltage_range,
        relay_trn_gaba_peak_by_index=relay_trn_gaba_peak,
        relay_trn_gaba_integral_ms_by_index=relay_trn_gaba_integral,
        relay_driven_current_range_pA_by_index_and_source=(
            relay_driven_current_range
        ),
        relay_event_current_samples_pA=relay_event_current_samples,
        relay_pre_event_current_samples_pA=relay_pre_event_current_samples,
        relay_pre_event_voltage_samples_mV=relay_pre_event_voltage_samples,
        relay_pre_event_trn_gaba_gate_samples=(
            relay_pre_event_trn_gaba_gate_samples
        ),
        trn_layer6ii_ampa_peak_by_index=trn_layer6ii_ampa_peak,
        trn_layer6ii_nmda_peak_by_index=trn_layer6ii_nmda_peak,
        trn_relay_ampa_peak_by_index=trn_relay_ampa_peak,
        trn_proximal_voltage_range_mV_by_index=trn_voltage_range,
        trn_soma_voltage_range_mV_by_index=trn_soma_voltage_range,
        trn_post_startup_soma_voltage_range_mV_by_index=(
            trn_post_startup_soma_voltage_range
        ),
        trn_detector_voltage_range_mV_by_index=trn_detector_voltage_range,
        trn_detector_post_first_event_voltage_range_mV_by_index=(
            trn_detector_post_first_event_voltage_range
        ),
        trn_detector_threshold_upcrossings_by_index=(
            trn_detector_threshold_upcrossings
        ),
        trn_detector_zero_downcrossings_by_index=trn_detector_zero_downcrossings,
        trn_detector_arm_transitions_by_index=trn_detector_arm_transitions,
        trn_detector_release_transitions_by_index=trn_detector_release_transitions,
        trn_detector_final_armed_by_index=trn_detector_final_armed,
        trn_driven_current_range_pA_by_index_and_source=trn_driven_current_range,
        nonspecific_trn_gaba_peak=nonspecific_trn_gaba_peak,
        nonspecific_trn_gaba_integral_ms=nonspecific_trn_gaba_integral_ms,
        nonspecific_post_startup_trn_gaba_peak=(
            nonspecific_post_startup_trn_gaba_peak
        ),
        nonspecific_layer6ii_ampa_peak=nonspecific_layer6ii_ampa_peak,
        nonspecific_layer6ii_nmda_peak=nonspecific_layer6ii_nmda_peak,
        nonspecific_direct_input_current_range_pA=(
            nonspecific_direct_input_current_range_pA
        ),
        nonspecific_trn_current_range_pA=nonspecific_trn_current_range_pA,
        nonspecific_layer6ii_current_range_pA=(
            nonspecific_layer6ii_current_range_pA
        ),
        nonspecific_voltage_range_mV_by_compartment=(
            nonspecific_voltage_range_mV_by_compartment
        ),
        nonspecific_positive_soma_local_maxima_ms_mV=(
            nonspecific_positive_soma_local_maxima_ms_mV
        ),
        nonspecific_positive_detector_local_maxima_ms_mV=(
            nonspecific_positive_detector_local_maxima_ms_mV
        ),
        nonspecific_detector_voltage_range_mV=(
            nonspecific_detector_voltage_range_mV
        ),
        nonspecific_detector_threshold_upcrossings=(
            nonspecific_detector_threshold_upcrossings
        ),
        nonspecific_detector_zero_downcrossings=(
            nonspecific_detector_zero_downcrossings
        ),
        nonspecific_detector_arm_transitions=(
            nonspecific_detector_arm_transitions
        ),
        nonspecific_detector_release_transitions=(
            nonspecific_detector_release_transitions
        ),
        nonspecific_detector_final_armed=nonspecific_detector_final_armed,
    )
    if cpp_standalone_directory is not None:
        brian.device.reinit()
        brian.set_device("runtime")
    return result
