import inspect
import math
from dataclasses import replace

import pytest

from smart_robustness.models.axial import (
    AxialConvention,
    build_axial_edges,
    build_table3_axial_edges,
)
from smart_robustness.models.table3 import TABLE3_CELLS, CellSpec, CompartmentSpec, get_cell_spec


def _edge(cell_name: str, convention: AxialConvention, near_name: str):
    return next(
        edge
        for edge in build_axial_edges(get_cell_spec(cell_name), convention)
        if edge.near.compartment_name == near_name
    )


def test_all_axial_conventions_are_named_and_none_is_default() -> None:
    assert {convention.value for convention in AxialConvention} == {
        "paper_literal",
        "symmetric_cable",
        "kinness_2008",
        "kinness_serialized_edge",
    }
    assert (
        inspect.signature(build_axial_edges).parameters["convention"].default
        is inspect.Parameter.empty
    )
    assert (
        inspect.signature(build_table3_axial_edges).parameters["convention"].default
        is inspect.Parameter.empty
    )


@pytest.mark.parametrize("convention", tuple(AxialConvention))
def test_all_table3_cells_compartments_and_adjacent_edges_are_covered(
    convention: AxialConvention,
) -> None:
    edges = build_table3_axial_edges(convention)
    expected_edge_count = sum(len(cell.compartments) - 1 for cell in TABLE3_CELLS.values())
    assert len(TABLE3_CELLS) == 12
    assert sum(len(cell.compartments) for cell in TABLE3_CELLS.values()) == 30
    assert len(edges) == expected_edge_count == 18

    covered: set[tuple[str, str]] = set()
    for edge in edges:
        covered.add((edge.cell_name, edge.near.compartment_name))
        covered.add((edge.cell_name, edge.far.compartment_name))
    expected = {
        (cell.name, compartment.name)
        for cell in TABLE3_CELLS.values()
        for compartment in cell.compartments
    }
    assert covered == expected


@pytest.mark.parametrize("convention", tuple(AxialConvention))
def test_all_derived_axial_values_are_finite_and_positive(convention: AxialConvention) -> None:
    for edge in build_table3_axial_edges(convention):
        values = (
            edge.near.diameter_cm,
            edge.near.length_cm,
            edge.near.axial_resistivity_kohm_cm,
            edge.near.lateral_area_cm2,
            edge.near.half_resistance_kohm,
            edge.far.diameter_cm,
            edge.far.length_cm,
            edge.far.axial_resistivity_kohm_cm,
            edge.far.lateral_area_cm2,
            edge.far.half_resistance_kohm,
            edge.conductance_into_near_nS,
            edge.conductance_into_far_nS,
            edge.conductance_density_into_near_mS_cm2,
            edge.conductance_density_into_far_mS_cm2,
        )
        assert all(math.isfinite(value) and value > 0 for value in values)


def test_paper_literal_trn_proximal_hand_calculation() -> None:
    edge = _edge("trn", AxialConvention.PAPER_LITERAL, "soma")

    # D=0.01 mm=0.001 cm, L=0.05 mm=0.005 cm, rho=10 kOhm*cm.
    expected_density_mS_cm2 = 0.001 / (4 * 10 * 0.005**2)
    expected_area_cm2 = math.pi * 0.001 * 0.005
    expected_total_nS = expected_density_mS_cm2 * expected_area_cm2 * 1e6
    assert expected_density_mS_cm2 == pytest.approx(1.0)
    assert edge.conductance_density_into_far_mS_cm2 == pytest.approx(expected_density_mS_cm2)
    assert edge.conductance_into_far_nS == pytest.approx(expected_total_nS)
    assert edge.conductance_into_far_nS == pytest.approx(15.7079632679)


def test_paper_literal_preserves_directional_asymmetry() -> None:
    edge = _edge("trn", AxialConvention.PAPER_LITERAL, "soma")
    assert edge.conductance_into_near_nS == pytest.approx(392.699081699)
    assert edge.conductance_into_far_nS == pytest.approx(15.7079632679)
    assert edge.conductance_into_near_nS != pytest.approx(edge.conductance_into_far_nS)


def test_kinness_2008_trn_edge_matches_framework_equation_9() -> None:
    edge = _edge("trn", AxialConvention.KINNESS_2008, "soma")

    d_soma, l_soma = 0.005, 0.005
    d_dendrite, l_dendrite = 0.001, 0.005
    rho = 10.0
    expected_soma_density = d_soma / (
        2 * rho * l_soma**2 * (1 + l_dendrite * d_soma**2 / (l_soma * d_dendrite**2))
    )
    expected_dendrite_density = d_dendrite / (
        2 * rho * l_dendrite**2 * (1 + l_soma * d_dendrite**2 / (l_dendrite * d_soma**2))
    )

    assert edge.conductance_density_into_near_mS_cm2 == pytest.approx(expected_soma_density)
    assert edge.conductance_density_into_far_mS_cm2 == pytest.approx(expected_dendrite_density)
    assert edge.convention is AxialConvention.KINNESS_2008


def test_symmetric_trn_edge_matches_two_half_resistances_in_series() -> None:
    edge = _edge("trn", AxialConvention.SYMMETRIC_CABLE, "soma")

    soma_half_resistance_kohm = 10 * (0.005 / 2) / (math.pi * (0.005 / 2) ** 2)
    proximal_half_resistance_kohm = 10 * (0.005 / 2) / (math.pi * (0.001 / 2) ** 2)
    expected_conductance_nS = 1e6 / (soma_half_resistance_kohm + proximal_half_resistance_kohm)

    assert edge.near.half_resistance_kohm == pytest.approx(soma_half_resistance_kohm)
    assert edge.far.half_resistance_kohm == pytest.approx(proximal_half_resistance_kohm)
    assert edge.conductance_into_near_nS == pytest.approx(expected_conductance_nS)
    assert edge.conductance_into_far_nS == pytest.approx(expected_conductance_nS)
    assert expected_conductance_nS == pytest.approx(30.2076216691)


def test_symmetric_edges_conserve_current_and_convert_ns_mv_to_pa() -> None:
    for edge in build_table3_axial_edges(AxialConvention.SYMMETRIC_CABLE):
        assert edge.conductance_into_near_nS == pytest.approx(edge.conductance_into_far_nS)
        current_near_pA, current_far_pA = edge.currents_pA(-70.0, -60.0)
        assert current_near_pA > 0
        assert current_far_pA < 0
        assert current_near_pA == pytest.approx(-current_far_pA)
        assert current_near_pA == pytest.approx(edge.conductance_into_near_nS * 10.0)
        assert edge.currents_pA(-65.0, -65.0) == pytest.approx((0.0, 0.0))


def test_endpoint_conductance_densities_recover_total_conductance() -> None:
    for convention in AxialConvention:
        for edge in build_table3_axial_edges(convention):
            assert (
                edge.conductance_density_into_near_mS_cm2 * edge.near.lateral_area_cm2 * 1e6
                == pytest.approx(edge.conductance_into_near_nS)
            )
            assert (
                edge.conductance_density_into_far_mS_cm2 * edge.far.lateral_area_cm2 * 1e6
                == pytest.approx(edge.conductance_into_far_nS)
            )


def _valid_compartment(name: str = "soma") -> CompartmentSpec:
    return CompartmentSpec(
        name=name,
        diameter_mm=0.01,
        length_mm=0.02,
        axial_resistance_kohm_cm=10.0,
        e_leak_mV=-65.0,
        g_leak_mS_cm2=0.1,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("diameter_mm", 0.0, "diameter_mm must be positive"),
        ("diameter_mm", -1.0, "diameter_mm must be positive"),
        ("diameter_mm", math.nan, "diameter_mm must be a finite number"),
        ("length_mm", math.inf, "length_mm must be a finite number"),
        (
            "axial_resistance_kohm_cm",
            0.0,
            "axial_resistance_kohm_cm must be positive",
        ),
    ),
)
def test_invalid_geometry_and_resistivity_fail_loudly(
    field: str,
    value: float,
    message: str,
) -> None:
    malformed = replace(_valid_compartment(), **{field: value})
    cell = CellSpec("malformed", (malformed, _valid_compartment("dendrite")))
    with pytest.raises(ValueError, match=message):
        build_axial_edges(cell, AxialConvention.PAPER_LITERAL)


def test_invalid_cell_structure_and_convention_fail_loudly() -> None:
    with pytest.raises(ValueError, match="cell name must be non-empty"):
        build_axial_edges(
            CellSpec("", (_valid_compartment(), _valid_compartment("d"))), "paper_literal"
        )
    with pytest.raises(ValueError, match="at least two compartments"):
        build_axial_edges(CellSpec("single", (_valid_compartment(),)), "paper_literal")
    with pytest.raises(ValueError, match="compartment names must be unique"):
        build_axial_edges(
            CellSpec("duplicate", (_valid_compartment(), _valid_compartment())),
            "paper_literal",
        )
    with pytest.raises(ValueError, match="compartment name must be non-empty"):
        build_axial_edges(
            CellSpec("unnamed", (_valid_compartment(), _valid_compartment(""))),
            "paper_literal",
        )
    with pytest.raises(ValueError, match="unknown axial convention"):
        build_axial_edges(get_cell_spec("trn"), "unselected")


def test_current_calculation_rejects_non_finite_voltage() -> None:
    edge = _edge("trn", AxialConvention.SYMMETRIC_CABLE, "soma")
    with pytest.raises(ValueError, match="near voltage must be a finite number"):
        edge.currents_pA(math.nan, -60.0)


def test_kinness_serialized_edge_uses_child_connection_value_both_directions() -> None:
    edge = build_axial_edges(
        get_cell_spec("layer5_excitatory"),
        AxialConvention.KINNESS_SERIALIZED_EDGE,
    )[0]
    child_value = get_cell_spec("layer5_excitatory").compartments[1].axial_resistance_kohm_cm
    assert edge.near.axial_resistivity_kohm_cm == child_value
    assert edge.far.axial_resistivity_kohm_cm == child_value
    assert edge.conductance_into_near_nS == pytest.approx(edge.conductance_into_far_nS)
    near_current, far_current = edge.currents_pA(-70.0, -60.0)
    assert near_current == pytest.approx(-far_current)
