from __future__ import annotations

import pytest

from smart_robustness.models.axial import AxialConvention
from smart_robustness.models.currents import TTypeGateConvention
from smart_robustness.models.equation_builder import (
    CalciumDensityConvention,
    LeakConvention,
    VoltageCoordinate,
    compile_cell_equations,
)
from smart_robustness.models.table3 import TABLE3_CELLS, get_cell_spec


def _compile(name: str = "thalamic_relay"):
    return compile_cell_equations(
        get_cell_spec(name),
        axial_convention=AxialConvention.SYMMETRIC_CABLE,
        leak_convention=LeakConvention.TABLE3_REVERSAL,
        voltage_coordinate=VoltageCoordinate.RELATIVE_TO_TABLE3_LEAK,
        calcium_gate_convention=TTypeGateConvention.RECIPROCAL,
        calcium_density_convention=CalciumDensityConvention.TABLE3,
    )


def test_all_table3_compartments_compile() -> None:
    compiled = [_compile(name) for name in TABLE3_CELLS]
    assert sum(len(item.compartments) for item in compiled) == 30
    assert sum(len(item.axial_parameter_names) for item in compiled) == 36
    for item in compiled:
        for compartment in item.compartments:
            assert f"dv_{compartment}/dt" in item.equations


def test_relay_has_active_soma_and_calcium_dendrites() -> None:
    equations = _compile().equations
    assert "i_na_soma=" in equations
    assert "i_ca_proximal_dendrite=" in equations
    assert "i_ca_distal_dendrite=" in equations
    assert "m_ca_inf_proximal_dendrite=1/(" in equations


def test_conventions_must_be_explicit_enum_members() -> None:
    kwargs = {
        "axial_convention": AxialConvention.SYMMETRIC_CABLE,
        "leak_convention": LeakConvention.TABLE3_REVERSAL,
        "voltage_coordinate": VoltageCoordinate.RELATIVE_TO_TABLE3_LEAK,
        "calcium_gate_convention": TTypeGateConvention.RECIPROCAL,
        "calcium_density_convention": CalciumDensityConvention.TABLE3,
    }
    for key in tuple(kwargs):
        invalid = dict(kwargs)
        invalid[key] = str(kwargs[key])
        with pytest.raises(TypeError, match="explicit"):
            compile_cell_equations(get_cell_spec("trn"), **invalid)


def test_brian2_parses_every_cell_equation_set() -> None:
    brian = pytest.importorskip("brian2")
    for name in TABLE3_CELLS:
        brian.start_scope()
        compiled = _compile(name)
        group = brian.NeuronGroup(
            1,
            compiled.equations,
            threshold="armed > 0.5 and v_soma < 0*mV",
            reset="armed = 0",
            events={"arm_spike": "armed < 0.5 and v_soma > 30*mV"},
            method="exponential_euler",
        )
        group.run_on_event("arm_spike", "armed = 1")
        brian.Network(group).run(0 * brian.ms)
