"""Registered post-2008 direct visual-thalamus to layer-5 intervention."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from types import MappingProxyType
from typing import Any

from ..classic_sector import FirstOrderRuntimeConventions
from ..modeldb_projections import MODELDB_FIRST_ORDER, ModelDBProjection
from ..synapses import connect_modeldb_projection, modeldb_topology_pairs
from .compartmental_hh import create_compartmental_hh_population
from .ports import modeldb_chemical_port

DIRECT_VISUAL_THALAMUS_LAYER5_ID = (
    "post2008.direct_visual_thalamus_to_layer5_excitatory.ampa_v1"
)
DIRECT_VISUAL_THALAMUS_LAYER5_TARGET = "layer5_excitatory_v1"
DIRECT_VISUAL_THALAMUS_LAYER5_COMPARTMENT = "proximal_dendrite"
DIRECT_VISUAL_THALAMUS_LAYER5_TEMPLATE_ID = "modeldb112923.projection.035"
REGISTERED_DIRECT_VISUAL_THALAMUS_LAYER5_RATIOS = (
    0.0,
    0.03125,
    0.0625,
    0.125,
    0.25,
    0.5,
    1.0,
)


def _validated_ratio(value: float) -> float:
    ratio = float(value)
    if not math.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
        raise ValueError("direct visual-thalamus L5 ratio must be finite and in [0, 1]")
    return ratio


def direct_visual_thalamus_layer5_record(strength_ratio: float) -> ModelDBProjection:
    """Return the sealed fixed-weight projection record for one nonzero ratio."""

    ratio = _validated_ratio(strength_ratio)
    if ratio == 0.0:
        raise ValueError("ratio zero is the unchanged control and has no added record")
    template = MODELDB_FIRST_ORDER.by_id(DIRECT_VISUAL_THALAMUS_LAYER5_TEMPLATE_ID)
    if template.weight is None:
        raise RuntimeError("projection 035 lacks its serialized peak weight")
    fixed_weight = float(template.weight) * ratio
    method_attributes = dict(template.method_attributes)
    method_attributes["weight"] = str(fixed_weight)
    method_attributes["assymptoticWeight"] = str(fixed_weight)
    return replace(
        template,
        id=DIRECT_VISUAL_THALAMUS_LAYER5_ID,
        target_population=DIRECT_VISUAL_THALAMUS_LAYER5_TARGET,
        target_compartment=DIRECT_VISUAL_THALAMUS_LAYER5_COMPARTMENT,
        channel_name="POST2008 direct dLGN to L5 AMPA",
        weight=fixed_weight,
        asymptotic_weight=fixed_weight,
        projection_attributes=MappingProxyType({"modifiable": "false"}),
        method_attributes=MappingProxyType(method_attributes),
    )


def make_direct_visual_thalamus_layer5_population_factory(
    record: ModelDBProjection,
    base_population_factory: Callable[..., Any] | None = None,
) -> Callable[..., Any]:
    """Add the dedicated receptor port only to the registered L5 population."""

    if record.id != DIRECT_VISUAL_THALAMUS_LAYER5_ID:
        raise ValueError("unexpected direct visual-thalamus L5 record ID")
    base = base_population_factory or create_compartmental_hh_population

    def factory(*, name: str, size: int, params: dict[str, Any], brian: Any):
        transformed = params
        if params.get("cell_class") == DIRECT_VISUAL_THALAMUS_LAYER5_TARGET:
            transformed = dict(params)
            ports = tuple(params.get("synaptic_ports", ()))
            if any(port.record_id == record.id for port in ports):
                raise ValueError("direct visual-thalamus L5 port is already present")
            transformed["synaptic_ports"] = ports + (
                modeldb_chemical_port(record, len(ports)),
            )
        return base(name=name, size=size, params=transformed, brian=brian)

    return factory


def make_direct_visual_thalamus_layer5_sector_builder(
    *,
    strength_ratio: float,
    base_builder: Callable[..., Any],
) -> Callable[..., Any]:
    """Wrap a frozen sector builder and add exactly one registered projection."""

    ratio = _validated_ratio(strength_ratio)
    if ratio == 0.0:
        return base_builder
    record = direct_visual_thalamus_layer5_record(ratio)

    def builder(*args: Any, **kwargs: Any):
        if args:
            raise TypeError("the intervention sector builder accepts keyword arguments only")
        brian = kwargs.get("brian")
        if brian is None:
            import brian2 as brian

            kwargs["brian"] = brian
        conventions = kwargs.get("conventions")
        if conventions is None:
            conventions = FirstOrderRuntimeConventions(
                gate_initialization_convention=kwargs.get(
                    "gate_initialization_convention",
                    "steady_state_at_initial_voltage",
                )
            )
        kwargs["population_factory"] = (
            make_direct_visual_thalamus_layer5_population_factory(
                record,
                kwargs.get("population_factory"),
            )
        )
        sector = base_builder(**kwargs)
        if DIRECT_VISUAL_THALAMUS_LAYER5_ID in sector.projections:
            raise ValueError("direct visual-thalamus L5 projection is already present")
        template = MODELDB_FIRST_ORDER.by_id(
            DIRECT_VISUAL_THALAMUS_LAYER5_TEMPLATE_ID
        )
        topology_override = modeldb_topology_pairs(
            template,
            source_shape=(9, 9),
            target_shape=(9, 9),
            gaussian_weight_convention=conventions.gaussian_weight_convention,
            gaussian_spread_convention=conventions.gaussian_spread_convention,
            ring_kernel_convention=conventions.ring_kernel_convention,
        )
        projection = connect_modeldb_projection(
            record,
            pre=sector.populations[record.source_population],
            post=sector.populations[record.target_population],
            source_shape=(9, 9),
            target_shape=(9, 9),
            modifiable_weight_initialization=(
                conventions.modifiable_weight_initialization
            ),
            gaussian_weight_convention=conventions.gaussian_weight_convention,
            gaussian_spread_convention=conventions.gaussian_spread_convention,
            ring_kernel_convention=conventions.ring_kernel_convention,
            ring_peak_radius_scale=1.0,
            gaussian_learning_bounds_convention=(
                conventions.gaussian_learning_bounds_convention
            ),
            spike_event_coordinate=conventions.spike_event_coordinate,
            spike_event_threshold_mV=conventions.spike_event_threshold_mV,
            postsynaptic_learning_threshold_mV=(
                conventions.postsynaptic_learning_threshold_mV
            ),
            postsynaptic_learning_coordinate=(
                conventions.postsynaptic_learning_coordinate
            ),
            postsynaptic_learning_timestamp=(
                conventions.postsynaptic_learning_timestamp
            ),
            postsynaptic_signal_convention=(
                conventions.postsynaptic_signal_convention
            ),
            postsynaptic_depression_scale_convention=(
                conventions.postsynaptic_depression_scale_convention
            ),
            topology_override=topology_override,
            brian=brian,
        )
        sector.projections[record.id] = projection
        sector.network.add(projection)
        return sector

    return builder
