from __future__ import annotations

import numpy as np
import pytest

from smart_robustness.analysis.lfp import (
    current_source_density_uV_per_um2,
    extracellular_potential_uV,
    figure16_electrode_geometry,
)
from smart_robustness.models.table3 import get_cell_spec


def test_equation31_single_source_has_expected_si_conversion() -> None:
    # 1 pA at 1 um with sigma=15 mS/cm=1.5 S/m gives
    # 1/(4*pi*1.5) microvolts.
    result = extracellular_potential_uV(np.asarray([[1.0]]), np.asarray([[1.0]]))
    assert result.shape == (1, 1)
    assert result[0, 0] == pytest.approx(1.0 / (4 * np.pi * 1.5))


def test_equation31_sums_signed_sources_and_sinks_by_inverse_distance() -> None:
    currents = np.asarray([[2.0, 4.0], [-1.0, -2.0]])
    distances = np.asarray([[10.0, 20.0], [20.0, 10.0]])
    result = extracellular_potential_uV(currents, distances)
    scale = 1.0 / (4 * np.pi * 1.5)
    assert result[0] == pytest.approx(scale * np.asarray([0.15, 0.30]))
    assert result[1] == pytest.approx(scale * np.asarray([0.0, 0.0]))


def test_equation33_recovers_constant_second_difference() -> None:
    positions = np.arange(5, dtype=float)
    potential = (positions**2)[:, None] * np.asarray([[1.0, 2.0]])
    csd = current_source_density_uV_per_um2(potential, tip_spacing_um=1.0)
    assert csd.shape == (3, 2)
    assert csd == pytest.approx(np.tile([2.0, 4.0], (3, 1)))


@pytest.mark.parametrize(
    "currents,distances",
    (
        (np.arange(3), np.ones((1, 3))),
        (np.ones((2, 3)), np.ones((1, 3))),
        (np.ones((2, 3)), np.asarray([[1.0, 0.0]])),
    ),
)
def test_lfp_rejects_invalid_shapes_and_distances(currents, distances) -> None:
    with pytest.raises(ValueError):
        extracellular_potential_uV(currents, distances)


def test_csd_rejects_too_few_tips_and_invalid_spacing() -> None:
    with pytest.raises(ValueError, match="three"):
        current_source_density_uV_per_um2(np.ones((2, 10)), 1.0)
    with pytest.raises(ValueError, match="spacing"):
        current_source_density_uV_per_um2(np.ones((3, 10)), 0.0)


def test_figure16_geometry_reconstructs_reported_54_tip_protocol() -> None:
    cell = get_cell_spec("layer23_excitatory")
    geometry = figure16_electrode_geometry(
        cell, 9, selected_cell_index=4, seed=2008
    )

    assert geometry.tip_depth_um.shape == (54,)
    assert geometry.tip_depth_um[[0, -1]] == pytest.approx([0.0, 275.0])
    assert np.diff(geometry.tip_depth_um) == pytest.approx(geometry.tip_spacing_um)
    assert geometry.compartment_depth_um[:2] == pytest.approx([25.0, 162.5])
    assert geometry.distance_um.shape == (54, 18)
    assert geometry.compartment_labels[:3] == (
        (0, "soma"),
        (0, "proximal_dendrite"),
        (1, "soma"),
    )
    assert 10.0 <= geometry.cell_lateral_distance_um[4] <= 200.0
    assert np.all(geometry.cell_lateral_distance_um >= 10.0)
    assert np.all(geometry.cell_lateral_distance_um <= 1000.0)
    assert np.all(geometry.distance_um > 0.0)


def test_figure16_geometry_is_seed_deterministic_and_read_only() -> None:
    cell = get_cell_spec("layer5_excitatory")
    first = figure16_electrode_geometry(cell, 3, selected_cell_index=1, seed=7)
    repeated = figure16_electrode_geometry(cell, 3, selected_cell_index=1, seed=7)
    changed = figure16_electrode_geometry(cell, 3, selected_cell_index=1, seed=8)

    assert first.fingerprint == repeated.fingerprint
    assert first.distance_um == pytest.approx(repeated.distance_um)
    assert first.fingerprint != changed.fingerprint
    assert not first.distance_um.flags.writeable


@pytest.mark.parametrize(
    "kwargs,error",
    (
        ({"population_size": 0, "selected_cell_index": 0, "seed": 1}, ValueError),
        ({"population_size": 2, "selected_cell_index": 2, "seed": 1}, ValueError),
        ({"population_size": 2, "selected_cell_index": 0, "seed": True}, TypeError),
        (
            {"population_size": 2, "selected_cell_index": 0, "seed": 1, "tip_count": 1},
            ValueError,
        ),
    ),
)
def test_figure16_geometry_rejects_invalid_protocol(kwargs, error) -> None:
    with pytest.raises(error):
        figure16_electrode_geometry(get_cell_spec("layer4_excitatory"), **kwargs)
