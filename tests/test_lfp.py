from __future__ import annotations

import numpy as np
import pytest

from smart_robustness.analysis.lfp import (
    current_source_density_uV_per_um,
    extracellular_potential_uV,
    figure16_cortical_field,
    figure16_electrode_geometry,
    figure16_population_field,
    standard_current_source_density_uV_per_um2,
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


def test_equation33_uses_printed_single_power_of_tip_spacing() -> None:
    positions = np.arange(5, dtype=float)
    potential = (positions**2)[:, None] * np.asarray([[1.0, 2.0]])
    csd = current_source_density_uV_per_um(potential, tip_spacing_um=2.0)
    assert csd.shape == (3, 2)
    assert csd == pytest.approx(np.tile([1.0, 2.0], (3, 1)))


def test_standard_second_derivative_is_explicit_alternate() -> None:
    potential = (np.arange(5, dtype=float) ** 2)[:, None]
    result = standard_current_source_density_uV_per_um2(potential, tip_spacing_um=2.0)
    assert result == pytest.approx(np.full((3, 1), 0.5))


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
        current_source_density_uV_per_um(np.ones((2, 10)), 1.0)
    with pytest.raises(ValueError, match="spacing"):
        current_source_density_uV_per_um(np.ones((3, 10)), 0.0)


def test_figure16_geometry_reconstructs_reported_54_tip_protocol() -> None:
    cell = get_cell_spec("layer23_excitatory")
    geometry = figure16_electrode_geometry(
        cell, 9, selected_cell_index=4, seed=2008
    )

    assert geometry.tip_depth_um.shape == (54,)
    assert geometry.tip_depth_um[[0, -1]] == pytest.approx([0.0, 1200.0])
    assert np.diff(geometry.tip_depth_um) == pytest.approx(geometry.tip_spacing_um)
    assert geometry.compartment_depth_um[:2] == pytest.approx([850.0, 1087.5])
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
    "cell_name,expected_depths_um",
    (
        ("layer23_inhibitory", [1125.0, 1175.0]),
        ("layer4_excitatory", [575.0, 702.5]),
        ("layer4_inhibitory", [475.0, 525.0]),
        ("layer5_excitatory", [150.0, 425.0, 850.0]),
        ("layer6ii_excitatory", [50.0, 150.0, 300.0]),
        ("layer6i_excitatory", [50.0, 150.0]),
    ),
)
def test_figure16_geometry_uses_figure18_absolute_depths(
    cell_name, expected_depths_um
) -> None:
    geometry = figure16_electrode_geometry(
        get_cell_spec(cell_name), 1, selected_cell_index=0, seed=18
    )
    assert geometry.compartment_depth_um == pytest.approx(expected_depths_um)


def test_figure16_geometry_rejects_non_cortical_cells() -> None:
    with pytest.raises(ValueError, match="no cortical geometry"):
        figure16_electrode_geometry(
            get_cell_spec("thalamic_relay"), 1, selected_cell_index=0, seed=18
        )


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


def test_population_field_preserves_cell_major_compartment_order() -> None:
    cell = get_cell_spec("layer23_excitatory")
    field = figure16_population_field(
        cell,
        {
            "soma": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            "proximal_dendrite": np.asarray([[10.0, 20.0], [30.0, 40.0]]),
        },
        selected_cell_index=0,
        seed=16,
    )

    assert field.transmembrane_current_pA == pytest.approx(
        np.asarray([[1.0, 2.0], [10.0, 20.0], [3.0, 4.0], [30.0, 40.0]])
    )
    assert field.geometry.compartment_labels == (
        (0, "soma"),
        (0, "proximal_dendrite"),
        (1, "soma"),
        (1, "proximal_dendrite"),
    )
    assert field.potential_uV.shape == (54, 2)
    assert field.current_source_density_uV_per_um.shape == (52, 2)
    assert not field.potential_uV.flags.writeable


def test_population_field_rejects_missing_or_misaligned_currents() -> None:
    cell = get_cell_spec("layer23_excitatory")
    with pytest.raises(ValueError, match="missing"):
        figure16_population_field(
            cell, {"soma": np.ones((2, 3))}, selected_cell_index=0, seed=1
        )
    with pytest.raises(ValueError, match="identical"):
        figure16_population_field(
            cell,
            {"soma": np.ones((2, 3)), "proximal_dendrite": np.ones((1, 3))},
            selected_cell_index=0,
            seed=1,
        )


def test_cortical_field_sums_populations_and_exposes_caption_regions() -> None:
    zeros_23 = {
        "soma": np.zeros((2, 4)),
        "proximal_dendrite": np.zeros((2, 4)),
    }
    driven_4 = {
        "soma": np.ones((1, 4)),
        "proximal_dendrite": np.full((1, 4), 2.0),
    }
    combined = figure16_cortical_field(
        {
            "layer23_excitatory_v1": (
                get_cell_spec("layer23_excitatory"),
                zeros_23,
                0,
            ),
            "layer4_excitatory_v1": (get_cell_spec("layer4_excitatory"), driven_4, 0),
        },
        seed=16,
    )
    layer4 = dict(combined.population_fields)["layer4_excitatory_v1"]

    assert combined.potential_uV == pytest.approx(layer4.potential_uV)
    assert combined.potential_uV.shape == (54, 4)
    assert combined.current_source_density_uV_per_um.shape == (52, 4)
    assert combined.inferior_300um_potential_uV.shape == (14, 4)
    assert combined.superior_300um_potential_uV.shape == (14, 4)
    assert np.all(combined.inferior_300um_tip_depth_um <= 300.0)
    assert np.all(combined.superior_300um_tip_depth_um >= 900.0)
    assert not combined.potential_uV.flags.writeable


def test_cortical_field_rejects_misaligned_time_axes() -> None:
    cell = get_cell_spec("layer4_excitatory")
    with pytest.raises(ValueError, match="same time axis"):
        figure16_cortical_field(
            {
                "layer4_excitatory_v1": (
                    cell,
                    {name: np.ones((1, 2)) for name in ("soma", "proximal_dendrite")},
                    0,
                ),
                "layer4_excitatory_v2": (
                    cell,
                    {name: np.ones((1, 3)) for name in ("soma", "proximal_dendrite")},
                    0,
                ),
            },
            seed=1,
        )


def test_source_specific_cell_uses_explicit_cortical_class() -> None:
    source_cell = get_cell_spec("layer4_excitatory")
    source_cell = type(source_cell)("recovered_source_specific_name", source_cell.compartments)
    geometry = figure16_electrode_geometry(
        source_cell,
        1,
        cortical_class="layer4_excitatory",
        selected_cell_index=0,
        seed=18,
    )
    assert geometry.compartment_depth_um == pytest.approx([575.0, 702.5])
