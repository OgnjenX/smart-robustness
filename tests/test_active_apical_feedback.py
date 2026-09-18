from __future__ import annotations

import numpy as np
import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.classic_sector import build_full_smart_network
from smart_robustness.modeldb_projections import MODELDB_FULL
from smart_robustness.models.active_apical_feedback import (
    ACTIVE_APICAL_FEEDBACK_COMPARTMENT,
    ACTIVE_APICAL_FEEDBACK_SOURCE,
    ACTIVE_APICAL_FEEDBACK_TARGET,
    ACTIVE_APICAL_FEEDBACK_TEMPLATE_ID,
    REGISTERED_OFFSET_ROUTE_FRACTIONS,
    active_apical_feedback_topology,
    make_active_apical_feedback_full_builder,
)


def _dense(topology) -> np.ndarray:
    pre, post, factor = topology
    matrix = np.zeros((81, 81), dtype=float)
    matrix[pre, post] = factor
    return matrix


def test_feedback_registration_targets_existing_distal_v2_to_v1_path() -> None:
    record = MODELDB_FULL.by_id(ACTIVE_APICAL_FEEDBACK_TEMPLATE_ID)
    assert record.source_population == ACTIVE_APICAL_FEEDBACK_SOURCE
    assert record.target_population == ACTIVE_APICAL_FEEDBACK_TARGET
    assert record.target_compartment == ACTIVE_APICAL_FEEDBACK_COMPARTMENT


@pytest.mark.parametrize("fraction", (-0.1, float("nan"), float("inf"), 1.1))
def test_feedback_route_rejects_values_outside_registered_domain(fraction: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        active_apical_feedback_topology(fraction)


def test_offset_route_preserves_each_source_row_sum_and_moves_weight_off_center() -> None:
    aligned = _dense(active_apical_feedback_topology(0.0))
    offset = _dense(active_apical_feedback_topology(1.0))
    np.testing.assert_allclose(np.sum(offset, axis=1), np.sum(aligned, axis=1))
    assert aligned[40, 40] > 0
    assert offset[40, 40] == 0
    assert offset[40, 39] > aligned[40, 39]


@pytest.mark.parametrize("fraction", REGISTERED_OFFSET_ROUTE_FRACTIONS)
def test_every_registered_route_fraction_preserves_total_available_drive(
    fraction: float,
) -> None:
    control = _dense(active_apical_feedback_topology(0.0))
    candidate = _dense(active_apical_feedback_topology(fraction))
    np.testing.assert_allclose(
        np.sum(candidate, axis=1),
        np.sum(control, axis=1),
        rtol=1e-12,
        atol=1e-12,
    )


def test_zero_route_returns_exact_unchanged_full_builder() -> None:
    assert (
        make_active_apical_feedback_full_builder(
            offset_route_fraction=0.0,
            base_builder=build_full_smart_network,
        )
        is build_full_smart_network
    )


def test_nonzero_route_changes_only_projection016_topology() -> None:
    brian.start_scope()
    brian.prefs.codegen.target = "numpy"
    builder = make_active_apical_feedback_full_builder(
        offset_route_fraction=1.0,
        base_builder=build_full_smart_network,
    )
    sector = builder(
        projection_ids=frozenset({ACTIVE_APICAL_FEEDBACK_TEMPLATE_ID}),
        brian=brian,
    )
    projection = sector.projections[ACTIVE_APICAL_FEEDBACK_TEMPLATE_ID]
    expected = active_apical_feedback_topology(1.0)
    assert np.array_equal(np.asarray(projection.i), expected[0])
    assert np.array_equal(np.asarray(projection.j), expected[1])
    np.testing.assert_allclose(np.asarray(projection.w), expected[2])
    sector.network.run(0.01 * brian.ms)
