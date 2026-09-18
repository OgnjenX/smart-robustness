"""Registered conductance-only routing arm for active-apical feedback study."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np

from ..modeldb_projections import MODELDB_FULL, ModelDBKernel
from ..synapses import RingKernelConvention, modeldb_topology_pairs

ACTIVE_APICAL_FEEDBACK_TEMPLATE_ID = "modeldb112923.projection.016"
ACTIVE_APICAL_FEEDBACK_SOURCE = "layer6ii_excitatory_v2"
ACTIVE_APICAL_FEEDBACK_TARGET = "layer5_excitatory_v1"
ACTIVE_APICAL_FEEDBACK_COMPARTMENT = "distal_dendrite"
REGISTERED_OFFSET_ROUTE_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _validated_route_fraction(value: float) -> float:
    fraction = float(value)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("offset route fraction must be finite and in [0, 1]")
    return fraction


def _dense_topology(
    topology: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    pre, post, factor = topology
    dense = np.zeros((81, 81), dtype=float)
    dense[np.asarray(pre, dtype=int), np.asarray(post, dtype=int)] = np.asarray(
        factor, dtype=float
    )
    return dense


def active_apical_feedback_topology(
    offset_route_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend aligned and nearest-surround feedback at fixed row-summed drive.

    The aligned component is the archived projection-016 Gaussian. The offset
    component uses the same serialized sigma and wrap geometry with the
    preregistered parameter-free radial-annulus interpretation. Every source
    row in the annular component is normalized to the archived row sum before
    blending, so this intervention changes routing rather than total available
    projection weight.
    """

    fraction = _validated_route_fraction(offset_route_fraction)
    template = MODELDB_FULL.by_id(ACTIVE_APICAL_FEEDBACK_TEMPLATE_ID)
    if template.kernel is None:
        raise RuntimeError("projection 016 lacks its serialized spatial kernel")
    local = _dense_topology(
        modeldb_topology_pairs(
            template,
            source_shape=(9, 9),
            target_shape=(9, 9),
        )
    )
    annular_record = replace(
        template,
        kernel=ModelDBKernel(
            sigma_x=template.kernel.sigma_x,
            sigma_y=template.kernel.sigma_y,
            width=template.kernel.width,
            height=template.kernel.height,
            ring=True,
            wrap=template.kernel.wrap,
            border_effect=template.kernel.border_effect,
        ),
    )
    annular = _dense_topology(
        modeldb_topology_pairs(
            annular_record,
            source_shape=(9, 9),
            target_shape=(9, 9),
            ring_kernel_convention=RingKernelConvention.RADIAL_ANNULUS,
        )
    )
    local_row_sum = np.sum(local, axis=1)
    annular_row_sum = np.sum(annular, axis=1)
    if np.any(local_row_sum <= 0) or np.any(annular_row_sum <= 0):
        raise RuntimeError("feedback routing produced an empty source row")
    annular *= (local_row_sum / annular_row_sum)[:, None]
    mixed = (1.0 - fraction) * local + fraction * annular
    pre, post = np.nonzero(mixed > 0)
    factor = mixed[pre, post]
    return (
        pre.astype(int, copy=False),
        post.astype(int, copy=False),
        factor.astype(float, copy=False),
    )


def make_active_apical_feedback_full_builder(
    *,
    offset_route_fraction: float,
    base_builder: Callable[..., Any],
) -> Callable[..., Any]:
    """Wrap the full SMART builder with only projection-016 routing changed."""

    fraction = _validated_route_fraction(offset_route_fraction)
    if fraction == 0.0:
        return base_builder
    topology = active_apical_feedback_topology(fraction)

    def builder(*args: Any, **kwargs: Any):
        if args:
            raise TypeError("the active-apical feedback builder accepts keywords only")
        overrides = dict(kwargs.pop("projection_topology_overrides", {}) or {})
        if ACTIVE_APICAL_FEEDBACK_TEMPLATE_ID in overrides:
            raise ValueError("projection 016 already has a topology override")
        overrides[ACTIVE_APICAL_FEEDBACK_TEMPLATE_ID] = topology
        return base_builder(
            projection_topology_overrides=overrides,
            **kwargs,
        )

    return builder
