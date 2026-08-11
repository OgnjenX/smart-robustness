from __future__ import annotations

from typing import Any

import numpy as np

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
