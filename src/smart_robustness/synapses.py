from __future__ import annotations

from typing import Any

import numpy as np

from .modeldb_projections import ModelDBProjection
from .models.compartmental_hh import CompartmentalPopulation
from .projections import ProjectionRecord, TopologyKind


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
    # KInNeSS serializes ``weight`` as the peak/maximal receptor-density
    # weight. The Gaussian therefore scales that peak and must remain in [0, 1];
    # it is not a probability-density kernel with a 1/(2*pi*sigma_x*sigma_y)
    # prefactor.
    factor = np.exp(-(dx**2 / (2 * kernel.sigma_x**2) + dy**2 / (2 * kernel.sigma_y**2)))
    if kernel.ring:
        # KInNeSS serializes no radius for a ring kernel. Its executable UI
        # convention is retained here as a center-excluding Gaussian candidate
        # until legacy source or a benchmark trace resolves the exact stencil.
        factor[(dx == 0) & (dy == 0)] = 0
    return pre, post, factor


def connect_modeldb_projection(
    record: ModelDBProjection,
    *,
    pre: CompartmentalPopulation,
    post: CompartmentalPopulation,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    brian: Any,
) -> Any:
    """Connect one exact ModelDB chemical record to its dedicated port."""

    port = next(
        (port for port in post.compiled.synaptic_ports if port.record_id == record.id),
        None,
    )
    if port is None or record.weight is None:
        raise ValueError(f"{record.id}: incomplete compiled ModelDB projection")
    resource = "*transmitter_pre" if pre.compiled.depletion_enabled else ""
    update = f"{port.name}_rise_post += w{resource}"
    if port.rise_ms != port.fall_ms:
        update += f"\n{port.name}_fall_post += w{resource}"
    model = "w_baseline : 1 (constant)\nw_maximum : 1 (constant)\nmodifiable : 1 (constant)"
    on_post = None
    if record.modifiable:
        if record.learning_rate is None or record.depotentiation_ms is None:
            raise ValueError(f"{record.id}: incomplete ModelDB learning parameters")
        if record.learning_rule == "Presynaptically gated":
            learning_gate = "pre_signal**2"
        elif record.learning_rule == "Postsynaptically gated":
            learning_gate = "post_signal**2"
        elif record.learning_rule == "Dual AND gated":
            learning_gate = "pre_signal*post_signal**2"
        else:
            raise ValueError(f"{record.id}: unsupported learning rule {record.learning_rule!r}")
        post_window = record.depotentiation_ms
        model += (
            f"\ndx_learning_rise/dt=-x_learning_rise/({port.rise_ms}*ms) : 1 (clock-driven)"
            f"\ndx_learning_fall/dt=-x_learning_fall/({port.fall_ms}*ms) : 1 (clock-driven)"
            f"\npre_signal={port.normalization}*(x_learning_fall-x_learning_rise) : 1"
            "\npost_elapsed=t-last_post_spike : second"
            "\ndepression_scale=-w_baseline/w_maximum : 1"
            "\nx_post_above=depression_scale+1 : 1"
            "\nx_post_early=depression_scale+1-post_elapsed/(0.1*ms) : 1"
            f"\nx_post_late=depression_scale*(1-(post_elapsed-0.1*ms)/({post_window}*ms)) : 1"
            f"\npost_signal=int(v_soma_post >= -20*mV)*x_post_above"
            " + int(v_soma_post < -20*mV and post_elapsed >= 0*ms and post_elapsed < 0.1*ms)*x_post_early"
            f" + int(v_soma_post < -20*mV and post_elapsed >= 0.1*ms and post_elapsed < {post_window + 0.1}*ms)*x_post_late : 1"
            f"\ndw/dt=({record.learning_rate}/ms)*({learning_gate})"
            "*(pre_signal*post_signal*w_maximum+w_baseline-w) : 1 (clock-driven)"
            "\nlast_post_spike : second"
        )
        update += "\nx_learning_rise += 1\nx_learning_fall += 1"
        on_post = "last_post_spike = t"
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
    source_indices, target_indices, spatial_factor = modeldb_topology_pairs(
        record, source_shape=source_shape, target_shape=target_shape
    )
    synapse.connect(i=source_indices, j=target_indices)
    asymptote = record.asymptotic_weight
    baseline = float(record.weight if asymptote is None else asymptote)
    maximum = float(record.weight)
    synapse.w = (baseline if record.modifiable else maximum) * spatial_factor
    synapse.w_baseline = baseline * spatial_factor
    synapse.w_maximum = maximum * spatial_factor
    synapse.modifiable = float(record.modifiable)
    if record.modifiable:
        synapse.x_learning_rise = 0
        synapse.x_learning_fall = 0
        synapse.last_post_spike = -1e9 * brian.second
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
    source_indices, target_indices, spatial_factor = modeldb_topology_pairs(
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
            f"i_{port.name}_post=g*(v_{source.name}_pre-v_{target.name}_post) : amp (summed)"
        ),
        name=f"modeldb_{port.name}_{pre.group.name}_electrical",
    )
    synapse.connect(i=source_indices, j=target_indices)
    synapse.g = total_nS * spatial_factor * brian.nsiemens
    return synapse
