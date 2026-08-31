from __future__ import annotations

from enum import StrEnum
from typing import Any

import numpy as np

from .modeldb_projections import ModelDBProjection
from .models.compartmental_hh import CompartmentalPopulation, SpikeEventCoordinate
from .projections import ProjectionRecord, TopologyKind


class ModifiableWeightInitialization(StrEnum):
    """Initial value assigned to a source-serialized modifiable projection."""

    SOURCE_SERIALIZED_WEIGHT = "source_serialized_weight"
    ASYMPTOTIC_BASELINE = "asymptotic_baseline"
    FIGURE6_PATHWAY_SPECIFIC = "figure6_pathway_specific"


class GaussianWeightConvention(StrEnum):
    """Interpretation of KInNeSS ``connectFromMany`` Gaussian weights."""

    NORMALIZED_DENSITY = "normalized_density"
    SOURCE_PEAK = "source_peak"


class GaussianSpreadConvention(StrEnum):
    """Interpretation of archived ``sigma_x``/``sigma_y`` kernel fields."""

    STANDARD_DEVIATION = "standard_deviation"
    VARIANCE = "variance"


class RingKernelConvention(StrEnum):
    """Executable interpretations of KInNeSS's undocumented ``ring`` flag."""

    CENTER_EXCLUDED_GAUSSIAN = "center_excluded_gaussian"
    RADIAL_ANNULUS = "radial_annulus"


KINNESS_GAUSSIAN_WEIGHT_CUTOFF = 0.001
"""Minimum KInNeSS Gaussian connection weight retained by the legacy editor."""


class GaussianLearningBoundsConvention(StrEnum):
    """Whether a Gaussian scales only initial weights or also learning bounds."""

    PROJECTION_LEVEL = "projection_level"
    SPATIALLY_SCALED = "spatially_scaled"
    FIGURE6_PATHWAY_SPECIFIC = "figure6_pathway_specific"


class PostsynapticDepressionScaleConvention(StrEnum):
    """Source selected for Equation 6's negative postsynaptic scale ``D``."""

    LOCAL_LEARNING_BOUNDS = "local_learning_bounds"
    SERIALIZED_PROJECTION_BOUNDS = "serialized_projection_bounds"


def connect_conductance(
    pre: Any,
    post: Any,
    *,
    receptor: str,
    probability: float,
    weight_ns: float,
    delay_ms: float,
    depletion: float,
    recovery_ms: float,
    brian: Any,
    name: str,
) -> Any:
    """Conductance synapse with Grossberg-style recoverable transmitter resource.

    The post-synaptic receptor port is a normalized dual exponential. Resource
    dynamics implement dz/dt=(1-z)/tau and z <- z*(1-epsilon) on an event.
    """
    if receptor not in {"exc", "inh"}:
        raise ValueError("receptor must be 'exc' or 'inh'")
    targets = (
        ("g_exc_rise_post", "g_exc_decay_post")
        if receptor == "exc"
        else ("g_inh_rise_post", "g_inh_decay_post")
    )
    syn = brian.Synapses(
        pre,
        post,
        model="dz/dt = (1-z)/tau_rec : 1 (clock-driven)\n"
        "w : siemens (constant)\n"
        "epsilon : 1 (constant)\n"
        "tau_rec : second (constant)",
        on_pre=f"{targets[0]} += w*z\n{targets[1]} += w*z\nz *= (1-epsilon)",
        method="exact",
        name=name,
    )
    syn.connect(p=probability)
    syn.w = weight_ns * brian.nsiemens
    syn.delay = delay_ms * brian.ms
    syn.z = 1.0
    syn.epsilon = depletion
    syn.tau_rec = recovery_ms * brian.ms
    return syn


def topology_pairs(
    record: ProjectionRecord,
    *,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic pre/post pairs and source-declared spatial factors."""

    source_size = source_shape[0] * source_shape[1]
    target_size = target_shape[0] * target_shape[1]
    kind = record.parsed.topology.kind
    if kind is TopologyKind.ONE_TO_ONE:
        if source_size != target_size:
            raise ValueError(f"{record.id}: one-to-one populations have different sizes")
        indices = np.arange(source_size, dtype=int)
        return indices, indices.copy(), np.ones(source_size)
    if kind is TopologyKind.ALL_TO_ONE:
        pre = np.repeat(np.arange(source_size, dtype=int), target_size)
        post = np.tile(np.arange(target_size, dtype=int), source_size)
        return pre, post, np.ones(pre.size)

    sigma = record.parsed.topology.sigma
    if sigma is None or sigma <= 0:
        raise ValueError(f"{record.id}: Gaussian topology requires sigma")
    pre = np.repeat(np.arange(source_size, dtype=int), target_size)
    post = np.tile(np.arange(target_size, dtype=int), source_size)
    pre_y, pre_x = np.divmod(pre, source_shape[1])
    post_y, post_x = np.divmod(post, target_shape[1])
    distance_squared = (pre_x - post_x) ** 2 + (pre_y - post_y) ** 2
    factor = np.exp(-distance_squared / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    return pre, post, factor


def connect_classic_projection(
    record: ProjectionRecord,
    *,
    pre: CompartmentalPopulation,
    post: CompartmentalPopulation,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    brian: Any,
) -> Any:
    """Connect one audited chemical projection to its dedicated target port."""

    port = next(
        (port for port in post.compiled.synaptic_ports if port.record_id == record.id),
        None,
    )
    if port is None:
        raise ValueError(f"{record.id}: target population has no compiled receptor port")
    base_weight = record.parsed.weight_density_1e6_cm2
    if base_weight is None:
        raise ValueError(f"{record.id}: chemical projection has no event weight")
    depletion = record.parsed.depletion
    epsilon = depletion.epsilon if depletion is not None else 0.0
    recovery_ms = depletion.tau_ms if depletion is not None else 1.0
    update = f"{port.name}_rise_post += w*z"
    if port.rise_ms != port.fall_ms:
        update += f"\n{port.name}_fall_post += w*z"
    update += "\nz *= (1-epsilon)"
    synapse = brian.Synapses(
        pre.group,
        post.group,
        model=(
            "dz/dt=(1-z)/tau_rec : 1 (clock-driven)\n"
            "w : 1 (constant)\n"
            "epsilon : 1 (constant)\n"
            "tau_rec : second (constant)"
        ),
        on_pre=update,
        method="exact",
        name=f"smart_{port.name}_{pre.group.name}_to_{post.group.name}",
    )
    source_indices, target_indices, spatial_factor = topology_pairs(
        record, source_shape=source_shape, target_shape=target_shape
    )
    synapse.connect(i=source_indices, j=target_indices)
    synapse.w = float(base_weight) * spatial_factor
    synapse.z = 1.0
    synapse.epsilon = epsilon
    synapse.tau_rec = recovery_ms * brian.ms
    synapse.delay = float(record.parsed.delay_ms or 0.0) * brian.ms
    return synapse


def connect_gap_junction(
    record: ProjectionRecord,
    *,
    pre: CompartmentalPopulation,
    post: CompartmentalPopulation,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    brian: Any,
) -> Any:
    """Connect one source-serialized electrical projection."""

    port = next(
        (port for port in post.compiled.gap_junction_ports if port.record_id == record.id),
        None,
    )
    if port is None:
        raise ValueError(f"{record.id}: target population has no gap-junction port")
    target_compartment = post.cell_spec.compartment(port.compartment)
    total_nS = port.conductance_density_mS_cm2 * target_compartment.lateral_area_cm2 * 1e6
    source_indices, target_indices, spatial_factor = topology_pairs(
        record, source_shape=source_shape, target_shape=target_shape
    )
    not_self = source_indices != target_indices
    source_indices = source_indices[not_self]
    target_indices = target_indices[not_self]
    spatial_factor = spatial_factor[not_self]
    synapse = brian.Synapses(
        pre.group,
        post.group,
        model=(
            "g : siemens (constant)\n"
            f"i_{port.name}_post=g*(v_{port.compartment}_pre-v_{port.compartment}_post) "
            ": amp (summed)"
        ),
        name=f"smart_{port.name}_{pre.group.name}_electrical",
    )
    synapse.connect(i=source_indices, j=target_indices)
    synapse.g = total_nS * spatial_factor * brian.nsiemens
    return synapse


def modeldb_topology_pairs(
    record: ModelDBProjection,
    *,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    gaussian_weight_convention: GaussianWeightConvention | str = (
        GaussianWeightConvention.SOURCE_PEAK
    ),
    gaussian_spread_convention: GaussianSpreadConvention | str = (
        GaussianSpreadConvention.STANDARD_DEVIATION
    ),
    ring_kernel_convention: RingKernelConvention | str = (
        RingKernelConvention.CENTER_EXCLUDED_GAUSSIAN
    ),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the connect method and spatial metadata serialized by KInNeSS."""

    source_size = source_shape[0] * source_shape[1]
    target_size = target_shape[0] * target_shape[1]
    if record.method == "connectFromOne":
        if source_size != target_size:
            raise ValueError(f"{record.id}: connectFromOne population sizes differ")
        indices = np.arange(source_size, dtype=int)
        return indices, indices.copy(), np.ones(source_size)
    pre = np.repeat(np.arange(source_size, dtype=int), target_size)
    post = np.tile(np.arange(target_size, dtype=int), source_size)
    if record.method == "connectFromAll":
        return pre, post, np.ones(pre.size)
    if record.method != "connectFromMany" or record.kernel is None:
        raise ValueError(f"{record.id}: unsupported or incomplete KInNeSS method")
    kernel = record.kernel
    if kernel.sigma_x is None or kernel.sigma_y is None:
        raise ValueError(f"{record.id}: KInNeSS Gaussian kernel lacks sigma")
    pre_y, pre_x = np.divmod(pre, source_shape[1])
    post_y, post_x = np.divmod(post, target_shape[1])
    pre_x = pre_x - (source_shape[1] - 1) / 2
    pre_y = pre_y - (source_shape[0] - 1) / 2
    post_x = post_x - (target_shape[1] - 1) / 2
    post_y = post_y - (target_shape[0] - 1) / 2
    dx = np.abs(pre_x - post_x)
    dy = np.abs(pre_y - post_y)
    if kernel.border_effect == "wrap" and source_shape == target_shape:
        dx = np.minimum(dx, source_shape[1] - dx)
        dy = np.minimum(dy, source_shape[0] - dy)
    # The archived KInNeSS user manual states that ``Weight`` is the Gaussian
    # peak and that shoulders whose resulting weight is below 0.001 are cut.
    # The normalized-density branch is retained only to reproduce the earlier
    # paper-figure inference as an explicitly rejected audit alternative.
    convention = GaussianWeightConvention(gaussian_weight_convention)
    spread_convention = GaussianSpreadConvention(gaussian_spread_convention)
    variance_x = (
        kernel.sigma_x**2
        if spread_convention is GaussianSpreadConvention.STANDARD_DEVIATION
        else kernel.sigma_x
    )
    variance_y = (
        kernel.sigma_y**2
        if spread_convention is GaussianSpreadConvention.STANDARD_DEVIATION
        else kernel.sigma_y
    )
    scaled_radius_squared = dx**2 / (2 * variance_x) + dy**2 / (2 * variance_y)
    factor = np.exp(-scaled_radius_squared)
    ring_convention = RingKernelConvention(ring_kernel_convention)
    if kernel.ring:
        if ring_convention is RingKernelConvention.CENTER_EXCLUDED_GAUSSIAN:
            # This is the historical project interpretation: preserve the
            # Gaussian shoulders and remove only the colocated sample.
            factor[(dx == 0) & (dy == 0)] = 0
        else:
            # Parameter-free annular sensitivity: r^2 exp(-r^2) has a zero
            # center and a unit peak at the elliptical radius fixed by the
            # serialized sigmas. This is not claimed as recovered KInNeSS
            # behavior; it is the sole preregistered alternative geometry.
            factor = scaled_radius_squared * np.exp(1 - scaled_radius_squared)
    if convention is GaussianWeightConvention.NORMALIZED_DENSITY:
        factor /= 2 * np.pi * np.sqrt(variance_x * variance_y)
    peak_weight = float(record.weight) if record.weight is not None else 1.0
    retained = peak_weight * factor >= KINNESS_GAUSSIAN_WEIGHT_CUTOFF
    return pre[retained], post[retained], factor[retained]


def connect_modeldb_projection(
    record: ModelDBProjection,
    *,
    pre: CompartmentalPopulation,
    post: CompartmentalPopulation,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    modifiable_weight_initialization: str,
    gaussian_weight_convention: GaussianWeightConvention | str,
    gaussian_spread_convention: GaussianSpreadConvention | str = (
        GaussianSpreadConvention.STANDARD_DEVIATION
    ),
    ring_kernel_convention: RingKernelConvention | str = (
        RingKernelConvention.CENTER_EXCLUDED_GAUSSIAN
    ),
    gaussian_learning_bounds_convention: GaussianLearningBoundsConvention | str,
    spike_event_coordinate: str = "absolute_physical",
    spike_event_threshold_mV: float = 30.0,
    postsynaptic_learning_coordinate: str | None = None,
    postsynaptic_learning_threshold_mV: float | None = None,
    postsynaptic_learning_timestamp: str = "emitted_event",
    postsynaptic_depression_scale_convention: str = "local_learning_bounds",
    postsynaptic_signal_convention: str = "paper_equation6_literal",
    instrument_learning_terms: bool = False,
    topology_override: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    brian: Any,
) -> Any:
    """Connect one exact ModelDB chemical record to its dedicated port."""

    port = next(
        (port for port in post.compiled.synaptic_ports if port.record_id == record.id),
        None,
    )
    if port is None or record.weight is None:
        raise ValueError(f"{record.id}: incomplete compiled ModelDB projection")
    transmitter_scale = "transmitter_pre" if pre.compiled.depletion_enabled else "1"
    update = (
        "previous_arrival = last_arrival\n"
        "previous_amplitude = last_amplitude\n"
        "last_arrival = t\n"
        "last_amplitude = 1"
    )
    if port.rise_ms == port.fall_ms:
        last_ratio = f"clip(last_elapsed/({port.rise_ms}*ms), 0, 100)"
        previous_ratio = f"clip(previous_elapsed/({port.rise_ms}*ms), 0, 100)"
        last_wave = (
            f"last_amplitude*exp(1)*({last_ratio})*exp(-({last_ratio}))"
        )
        previous_wave = (
            f"previous_amplitude*exp(1)*({previous_ratio})*exp(-({previous_ratio}))"
        )
    else:
        last_wave = (
            f"last_amplitude*{port.normalization}"
            f"*(exp(-clip(last_elapsed/({port.fall_ms}*ms), 0, 100))"
            f"-exp(-clip(last_elapsed/({port.rise_ms}*ms), 0, 100)))"
        )
        previous_wave = (
            f"previous_amplitude*{port.normalization}"
            f"*(exp(-clip(previous_elapsed/({port.fall_ms}*ms), 0, 100))"
            f"-exp(-clip(previous_elapsed/({port.rise_ms}*ms), 0, 100)))"
        )
    model = (
        "w_baseline : 1 (constant)\n"
        "w_maximum : 1 (constant)\n"
        "modifiable : 1 (constant)\n"
        "last_arrival : second\n"
        "previous_arrival : second\n"
        "last_amplitude : 1\n"
        "previous_amplitude : 1\n"
        "last_elapsed=t-last_arrival : second\n"
        "previous_elapsed=t-previous_arrival : second\n"
        f"last_wave={last_wave} : 1\n"
        f"previous_wave={previous_wave} : 1\n"
        "pre_signal=last_wave+previous_wave-last_wave*previous_wave : 1\n"
        # KInNeSS Equation 16 multiplies the ongoing ligand gate g_ij(t) by
        # neurotransmitter availability z_j(t). It does not snapshot z into
        # the event amplitude at emission or arrival.
        f"{port.name}_gate_post=w*pre_signal*{transmitter_scale} : 1 (summed)"
    )
    on_post = None
    if record.modifiable:
        if record.learning_rate is None or record.depotentiation_ms is None:
            raise ValueError(f"{record.id}: incomplete ModelDB learning parameters")
        if record.learning_rule == "Presynaptically gated":
            learning_gate = "pre_signal"
        elif record.learning_rule == "Postsynaptically gated":
            learning_gate = "post_signal**2"
        elif record.learning_rule == "Dual AND gated":
            learning_gate = "pre_signal*post_signal**2"
        else:
            raise ValueError(f"{record.id}: unsupported learning rule {record.learning_rule!r}")
        if postsynaptic_signal_convention == "paper_equation6_literal":
            positive_window = 0.1
            negative_window = float(record.depotentiation_ms)
        elif postsynaptic_signal_convention == "kinness_equation27_transition_time":
            positive_window = float(record.depotentiation_ms)
            negative_window = 25.0
        else:
            raise ValueError(
                "unsupported postsynaptic signal convention "
                f"{postsynaptic_signal_convention!r}"
            )
        event_coordinate = SpikeEventCoordinate(
            spike_event_coordinate
            if postsynaptic_learning_coordinate is None
            else postsynaptic_learning_coordinate
        )
        if event_coordinate is SpikeEventCoordinate.ABSOLUTE_PHYSICAL:
            post_voltage = "v_soma_post"
        elif event_coordinate is SpikeEventCoordinate.SHIFTED_67_MV:
            post_voltage = "v_soma_post+67*mV"
        else:
            post_voltage = "v_soma_post-e_l_soma_post"
        learning_term_instrumentation = ""
        if instrument_learning_terms:
            learning_term_instrumentation = (
                f"\ndlearning_positive/dt=({record.learning_rate}/ms)*({learning_gate})"
                "*clip(correlation_drive, 0, 1e9) : 1 (clock-driven)"
                f"\ndlearning_negative/dt=({record.learning_rate}/ms)*({learning_gate})"
                "*clip(correlation_drive, -1e9, 0) : 1 (clock-driven)"
                f"\ndlearning_baseline/dt=({record.learning_rate}/ms)*({learning_gate})"
                "*baseline_drive : 1 (clock-driven)"
            )
        depression_convention = PostsynapticDepressionScaleConvention(
            postsynaptic_depression_scale_convention
        )
        if (
            depression_convention
            is PostsynapticDepressionScaleConvention.LOCAL_LEARNING_BOUNDS
        ):
            depression_scale = "-w_baseline/w_maximum"
        else:
            serialized_baseline = float(
                record.weight
                if record.asymptotic_weight is None
                else record.asymptotic_weight
            )
            depression_scale = f"{-serialized_baseline / float(record.weight)!r}"
        learning_threshold_mV = (
            float(spike_event_threshold_mV)
            if postsynaptic_learning_threshold_mV is None
            else float(postsynaptic_learning_threshold_mV)
        )
        if postsynaptic_learning_timestamp == "upward_threshold":
            post_elapsed = "t-last_spike_onset_post"
            timestamp_state = ""
            on_post = None
        elif postsynaptic_learning_timestamp == "emitted_event":
            post_elapsed = "t-last_post_spike"
            timestamp_state = "\nlast_post_spike : second"
            on_post = "last_post_spike = t"
        else:
            raise ValueError(
                "unsupported postsynaptic learning timestamp "
                f"{postsynaptic_learning_timestamp!r}"
            )
        model += (
            f"\npost_elapsed={post_elapsed} : second"
            f"\ndepression_scale={depression_scale} : 1"
            "\nx_post_above=depression_scale+1 : 1"
            f"\nx_post_early=depression_scale+1-post_elapsed/({positive_window}*ms) : 1"
            f"\nx_post_late=depression_scale*(1-(post_elapsed-{positive_window}*ms)/({negative_window}*ms)) : 1"
            f"\npost_signal=int({post_voltage} >= {learning_threshold_mV}*mV)*x_post_above"
            f" + int({post_voltage} < {learning_threshold_mV}*mV and post_elapsed >= 0*ms and post_elapsed < {positive_window}*ms)*x_post_early"
            f" + int({post_voltage} < {learning_threshold_mV}*mV and post_elapsed >= {positive_window}*ms and post_elapsed < {positive_window + negative_window}*ms)*x_post_late : 1"
            "\ncorrelation_drive=pre_signal*post_signal*(w_maximum-w) : 1"
            "\nbaseline_drive=w_baseline-w : 1"
            f"\ndw/dt=({record.learning_rate}/ms)*({learning_gate})"
            "*(correlation_drive+baseline_drive) : 1 (clock-driven)"
            f"{learning_term_instrumentation}"
            f"{timestamp_state}"
        )
    else:
        model = "w : 1\n" + model
    synapse = brian.Synapses(
        pre.group,
        post.group,
        model=model,
        on_pre=update,
        on_post=on_post,
        method="rk4",
        name=f"modeldb_{port.name}_{pre.group.name}_to_{post.group.name}",
    )
    if topology_override is None:
        source_indices, target_indices, spatial_factor = modeldb_topology_pairs(
            record,
            source_shape=source_shape,
            target_shape=target_shape,
            gaussian_weight_convention=gaussian_weight_convention,
            gaussian_spread_convention=gaussian_spread_convention,
            ring_kernel_convention=ring_kernel_convention,
        )
    else:
        source_indices, target_indices, spatial_factor = (
            np.asarray(values) for values in topology_override
        )
        source_indices = source_indices.astype(int)
        target_indices = target_indices.astype(int)
        spatial_factor = spatial_factor.astype(float)
        if not (source_indices.shape == target_indices.shape == spatial_factor.shape):
            raise ValueError("topology override arrays must have identical shapes")
        if np.any(source_indices < 0) or np.any(source_indices >= int(pre.group.N)):
            raise ValueError("topology override source index outside partition")
        if np.any(target_indices < 0) or np.any(target_indices >= int(post.group.N)):
            raise ValueError("topology override target index outside partition")
    synapse.connect(i=source_indices, j=target_indices)
    initialization = ModifiableWeightInitialization(modifiable_weight_initialization)
    bounds_convention = GaussianLearningBoundsConvention(
        gaussian_learning_bounds_convention
    )
    asymptote = record.asymptotic_weight
    baseline = float(record.weight if asymptote is None else asymptote)
    maximum = float(record.weight)
    initialize_at_baseline = initialization is ModifiableWeightInitialization.ASYMPTOTIC_BASELINE
    if initialization is ModifiableWeightInitialization.FIGURE6_PATHWAY_SPECIFIC:
        # Figure 6b starts the bottom-up relay->L4 filter at its serialized
        # Gaussian weights, whereas Figure 6c depicts weak corticothalamic
        # expectations that grow toward their serialized maxima. These are the
        # only three adaptive records in the first-order archive.
        initialize_at_baseline = record.id in {
            "modeldb112923.projection.005",
            "modeldb112923.projection.007",
        }
    initial = baseline if record.modifiable and initialize_at_baseline else maximum
    synapse.w = initial * spatial_factor
    top_down_figure6_bounds = (
        bounds_convention is GaussianLearningBoundsConvention.FIGURE6_PATHWAY_SPECIFIC
        and record.id
        in {"modeldb112923.projection.005", "modeldb112923.projection.007"}
    )
    projection_level_bounds = bounds_convention in {
        GaussianLearningBoundsConvention.PROJECTION_LEVEL,
        GaussianLearningBoundsConvention.FIGURE6_PATHWAY_SPECIFIC,
    }
    if record.modifiable and projection_level_bounds:
        # Equation 25's w0 and upper bound are projection-level parameters,
        # whereas Section 4.9's Gaussian calculates the initialized synaptic
        # weights. A normalized narrow Gaussian can peak above its amplitude;
        # retain that legal initial state by lifting only the affected local
        # upper bounds.
        # Figure 6c's weak circular pre-map identifies a Gaussian-scaled local
        # decorrelation baseline, while its learned peak approaches the
        # projection-level maximum. A uniform 0.05 tail baseline otherwise
        # grows inactive directions and erases orientation. Figure 6b retains
        # the projection-level baseline independently constrained earlier.
        synapse.w_baseline = baseline * spatial_factor if top_down_figure6_bounds else baseline
        synapse.w_maximum = np.maximum(maximum, initial * spatial_factor)
    else:
        synapse.w_baseline = baseline * spatial_factor
        synapse.w_maximum = maximum * spatial_factor
    synapse.modifiable = float(record.modifiable)
    # Zero amplitudes make these timestamps inactive. Keeping them finite is
    # essential because Brian evaluates algebraic branches eagerly: historical
    # -1e9-second sentinels overflowed exponentials even when multiplied by 0.
    synapse.last_arrival = 0 * brian.second
    synapse.previous_arrival = 0 * brian.second
    synapse.last_amplitude = 0
    synapse.previous_amplitude = 0
    if record.modifiable and postsynaptic_learning_timestamp == "emitted_event":
        # Start strictly outside the serialized post-spike learning window.
        inactive_window = (
            float(record.depotentiation_ms) + 25.0
            if postsynaptic_signal_convention == "kinness_equation27_transition_time"
            else float(record.depotentiation_ms) + 0.1
        )
        synapse.last_post_spike = -(inactive_window + 1.0) * brian.ms
    synapse.delay = float(record.delay_ms or 0.0) * brian.ms
    return synapse


def kinness_gap_total_conductance_nS(
    axial_conductance: float, *, diameter_mm: float, length_mm: float
) -> float:
    """KInNeSS framework Equation 8, converted to total compartment nS."""

    diameter_cm = diameter_mm * 0.1
    length_cm = length_mm * 0.1
    density_mS_cm2 = axial_conductance * diameter_cm / (4 * length_cm**2)
    lateral_area_cm2 = np.pi * diameter_cm * length_cm
    return float(density_mS_cm2 * lateral_area_cm2 * 1e6)


def connect_modeldb_gap_junction(
    record: ModelDBProjection,
    *,
    pre: CompartmentalPopulation,
    post: CompartmentalPopulation,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    gaussian_weight_convention: GaussianWeightConvention | str = (
        GaussianWeightConvention.SOURCE_PEAK
    ),
    gaussian_spread_convention: GaussianSpreadConvention | str = (
        GaussianSpreadConvention.STANDARD_DEVIATION
    ),
    ring_kernel_convention: RingKernelConvention | str = (
        RingKernelConvention.CENTER_EXCLUDED_GAUSSIAN
    ),
    topology_override: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    brian: Any,
) -> Any:
    """Connect one ModelDB electrical projection using KInNeSS Equation 8."""

    port = next(
        (port for port in post.compiled.gap_junction_ports if port.record_id == record.id),
        None,
    )
    if port is None or record.channel_conductance_mS_cm2 is None:
        raise ValueError(f"{record.id}: incomplete ModelDB gap-junction record")
    target = post.cell_spec.compartment(port.compartment)
    source_compartment = record.source_compartment or port.compartment
    source = pre.cell_spec.compartment(source_compartment)
    total_nS = kinness_gap_total_conductance_nS(
        record.channel_conductance_mS_cm2,
        diameter_mm=target.diameter_mm,
        length_mm=target.length_mm,
    )
    if topology_override is None:
        source_indices, target_indices, spatial_factor = modeldb_topology_pairs(
            record,
            source_shape=source_shape,
            target_shape=target_shape,
            gaussian_weight_convention=gaussian_weight_convention,
            gaussian_spread_convention=gaussian_spread_convention,
            ring_kernel_convention=ring_kernel_convention,
        )
    else:
        source_indices, target_indices, spatial_factor = (
            np.asarray(values) for values in topology_override
        )
        source_indices = source_indices.astype(int)
        target_indices = target_indices.astype(int)
        spatial_factor = spatial_factor.astype(float)
        if not (source_indices.shape == target_indices.shape == spatial_factor.shape):
            raise ValueError("gap topology override arrays must have identical shapes")
        if np.any(source_indices < 0) or np.any(source_indices >= int(pre.group.N)):
            raise ValueError("gap topology override source index outside partition")
        if np.any(target_indices < 0) or np.any(target_indices >= int(post.group.N)):
            raise ValueError("gap topology override target index outside partition")
    not_self = source_indices != target_indices
    source_indices = source_indices[not_self]
    target_indices = target_indices[not_self]
    spatial_factor = spatial_factor[not_self]
    synapse = brian.Synapses(
        pre.group,
        post.group,
        model=(
            "g : siemens (constant)\n"
            f"i_{port.name}_post=g*(v_{source.name}_pre-v_{target.name}_post) : amp (summed)"
        ),
        name=f"modeldb_{port.name}_{pre.group.name}_electrical",
    )
    synapse.connect(i=source_indices, j=target_indices)
    synapse.g = total_nS * spatial_factor * brian.nsiemens
    return synapse
