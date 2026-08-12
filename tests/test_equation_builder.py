from __future__ import annotations

import pytest

from smart_robustness.models.axial import AxialConvention
from smart_robustness.models.currents import (
    AHPConvention,
    NaKRateConvention,
    TTypeGateConvention,
)
from smart_robustness.models.equation_builder import (
    CalciumDensityConvention,
    CalciumVoltageCoordinate,
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
        nak_rate_convention=NaKRateConvention.PRINTED_SMART,
        calcium_gate_convention=TTypeGateConvention.RECIPROCAL,
        calcium_voltage_coordinate=CalciumVoltageCoordinate.INTEGRATED_VOLTAGE,
        calcium_density_convention=CalciumDensityConvention.TABLE3,
        ahp_convention=AHPConvention.PAPER_TEXT,
        enable_ahp_ach=name == "layer5_excitatory",
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


def test_exact_voltage_clamp_replaces_only_selected_compartment_dynamics() -> None:
    compiled = compile_cell_equations(
        get_cell_spec("thalamic_relay"),
        axial_convention=AxialConvention.KINNESS_SERIALIZED_EDGE,
        leak_convention=LeakConvention.TABLE3_REVERSAL,
        voltage_coordinate=VoltageCoordinate.RELATIVE_TO_TABLE3_LEAK,
        nak_rate_convention=NaKRateConvention.STANDARD_TRAUB_MILES,
        calcium_gate_convention=TTypeGateConvention.MODELDB_112923,
        calcium_voltage_coordinate=CalciumVoltageCoordinate.INTEGRATED_VOLTAGE,
        calcium_density_convention=CalciumDensityConvention.TABLE3,
        ahp_convention=AHPConvention.MODELDB_112923,
        enable_ahp_ach=False,
        voltage_clamped_compartments=frozenset({"proximal_dendrite"}),
    )
    assert compiled.voltage_clamped_compartments == frozenset({"proximal_dendrite"})
    assert "dv_proximal_dendrite/dt=0*volt/second" in compiled.equations
    assert "dv_soma/dt=(" in compiled.equations


def test_global_67_mv_rate_coordinate_is_explicitly_compilable() -> None:
    compiled = compile_cell_equations(
        get_cell_spec("thalamic_relay"),
        axial_convention=AxialConvention.SYMMETRIC_CABLE,
        leak_convention=LeakConvention.TABLE3_REVERSAL,
        voltage_coordinate=VoltageCoordinate.SHIFTED_67_MV,
        nak_rate_convention=NaKRateConvention.PRINTED_SMART,
        calcium_gate_convention=TTypeGateConvention.RECIPROCAL,
        calcium_voltage_coordinate=CalciumVoltageCoordinate.INTEGRATED_VOLTAGE,
        calcium_density_convention=CalciumDensityConvention.TABLE3,
        ahp_convention=AHPConvention.PAPER_TEXT,
        enable_ahp_ach=False,
    )
    assert "v_soma+67*mV" in compiled.equations


def test_modeldb_calcium_equations_use_absolute_voltage_and_correct_roles() -> None:
    compiled = compile_cell_equations(
        get_cell_spec("thalamic_relay"),
        axial_convention=AxialConvention.SYMMETRIC_CABLE,
        leak_convention=LeakConvention.TABLE3_REVERSAL,
        voltage_coordinate=VoltageCoordinate.SHIFTED_67_MV,
        nak_rate_convention=NaKRateConvention.STANDARD_TRAUB_MILES,
        calcium_gate_convention=TTypeGateConvention.MODELDB_112923,
        calcium_voltage_coordinate=CalciumVoltageCoordinate.INTEGRATED_VOLTAGE,
        calcium_density_convention=CalciumDensityConvention.TABLE3,
        ahp_convention=AHPConvention.MODELDB_112923,
        enable_ahp_ach=False,
    )
    equations = compiled.equations
    assert "m_ca_inf_proximal_dendrite=1/(exp((-63*mV-v_proximal_dendrite)" in equations
    assert "tau_m_ca_proximal_dendrite=(2.44+" in equations
    assert "tau_h_ca_proximal_dendrite=(19.15+" in equations


def test_modeldb_reticular_calcium_uses_serialized_destexhe_gate_family() -> None:
    compiled = compile_cell_equations(
        get_cell_spec("trn"),
        axial_convention=AxialConvention.KINNESS_SERIALIZED_EDGE,
        leak_convention=LeakConvention.TABLE3_REVERSAL,
        voltage_coordinate=VoltageCoordinate.RELATIVE_TO_TABLE3_LEAK,
        nak_rate_convention=NaKRateConvention.STANDARD_TRAUB_MILES,
        calcium_gate_convention=TTypeGateConvention.MODELDB_RETICULAR_112923,
        calcium_voltage_coordinate=CalciumVoltageCoordinate.INTEGRATED_VOLTAGE,
        calcium_density_convention=CalciumDensityConvention.TABLE3,
        ahp_convention=AHPConvention.MODELDB_112923,
        enable_ahp_ach=False,
    )
    equations = compiled.equations
    assert (
        "m_ca_inf_proximal_dendrite=1/(exp((-52*mV-v_proximal_dendrite)/(7.4*mV))+1)"
        in equations
    )
    assert "tau_m_ca_proximal_dendrite=(1+0.33/(exp((v_proximal_dendrite+27*mV)" in equations
    assert (
        "h_ca_inf_proximal_dendrite=1/(exp((-80*mV-v_proximal_dendrite)/(-5*mV))+1)"
        in equations
    )
    assert "tau_h_ca_proximal_dendrite=(28.3+0.33/(exp((v_proximal_dendrite+48*mV)" in equations
    assert "g_ca_proximal_dendrite*m_ca_proximal_dendrite**2*h_ca_proximal_dendrite" in equations


def test_internal_zero_calcium_coordinate_adds_serialized_physical_leak() -> None:
    compiled = compile_cell_equations(
        get_cell_spec("thalamic_relay"),
        axial_convention=AxialConvention.KINNESS_SERIALIZED_EDGE,
        leak_convention=LeakConvention.PRINTED_ZERO,
        voltage_coordinate=VoltageCoordinate.ABSOLUTE,
        nak_rate_convention=NaKRateConvention.STANDARD_TRAUB_MILES,
        calcium_gate_convention=TTypeGateConvention.MODELDB_112923,
        calcium_voltage_coordinate=(
            CalciumVoltageCoordinate.INTERNAL_ZERO_PLUS_SERIALIZED_LEAK
        ),
        calcium_density_convention=CalciumDensityConvention.TABLE3,
        ahp_convention=AHPConvention.MODELDB_112923,
        enable_ahp_ach=False,
    )
    assert "(v_proximal_dendrite+(-60)*mV)" in compiled.equations
    assert "(e_ca-v_proximal_dendrite)" in compiled.equations
    brian = pytest.importorskip("brian2")
    brian.start_scope()
    group = brian.NeuronGroup(1, compiled.equations, method="rk4")
    brian.Network(group).run(0 * brian.ms)


def test_conventions_must_be_explicit_enum_members() -> None:
    kwargs = {
        "axial_convention": AxialConvention.SYMMETRIC_CABLE,
        "leak_convention": LeakConvention.TABLE3_REVERSAL,
        "voltage_coordinate": VoltageCoordinate.RELATIVE_TO_TABLE3_LEAK,
        "nak_rate_convention": NaKRateConvention.PRINTED_SMART,
        "calcium_gate_convention": TTypeGateConvention.RECIPROCAL,
        "calcium_voltage_coordinate": CalciumVoltageCoordinate.INTEGRATED_VOLTAGE,
        "calcium_density_convention": CalciumDensityConvention.TABLE3,
        "ahp_convention": AHPConvention.PAPER_TEXT,
        "enable_ahp_ach": False,
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
        group.armed = 0
        brian.Network(group).run(0 * brian.ms)


def test_modeldb_ahp_profile_uses_executable_tau() -> None:
    compiled = compile_cell_equations(
        get_cell_spec("layer5_excitatory"),
        axial_convention=AxialConvention.KINNESS_2008,
        leak_convention=LeakConvention.TABLE3_REVERSAL,
        voltage_coordinate=VoltageCoordinate.SHIFTED_67_MV,
        nak_rate_convention=NaKRateConvention.STANDARD_TRAUB_MILES,
        calcium_gate_convention=TTypeGateConvention.MODELDB_112923,
        calcium_voltage_coordinate=CalciumVoltageCoordinate.INTEGRATED_VOLTAGE,
        calcium_density_convention=CalciumDensityConvention.TABLE3,
        ahp_convention=AHPConvention.MODELDB_112923,
        enable_ahp_ach=True,
    )
    assert "dahp_fall/dt=-ahp_fall/(150.0*ms)" in compiled.equations
    assert "ahp_event_weight*ahp_gate" in compiled.equations
    assert compiled.ahp_convention is AHPConvention.MODELDB_112923


def test_full_network_ahp_profile_uses_smart_nml_kinetics() -> None:
    compiled = compile_cell_equations(
        get_cell_spec("layer6ii_excitatory"),
        axial_convention=AxialConvention.KINNESS_SERIALIZED_EDGE,
        leak_convention=LeakConvention.TABLE3_REVERSAL,
        voltage_coordinate=VoltageCoordinate.SHIFTED_67_MV,
        nak_rate_convention=NaKRateConvention.STANDARD_TRAUB_MILES,
        calcium_gate_convention=TTypeGateConvention.MODELDB_112923,
        calcium_voltage_coordinate=CalciumVoltageCoordinate.INTEGRATED_VOLTAGE,
        calcium_density_convention=CalciumDensityConvention.TABLE3,
        ahp_convention=AHPConvention.SMART_NETWORK_112923,
        enable_ahp_ach=True,
    )
    assert "dahp_rise/dt=-ahp_rise/(5.0*ms)" in compiled.equations
    assert "dahp_fall/dt=-ahp_fall/(20.0*ms)" in compiled.equations
    assert "(e_ahp-v_soma)" in compiled.equations
