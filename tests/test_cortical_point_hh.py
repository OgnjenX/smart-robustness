from __future__ import annotations

import math

import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.classic_sector import first_order_population_parameters
from smart_robustness.models.compartmental_hh import create_compartmental_hh_population
from smart_robustness.models.modeldb112923 import first_order_population_facts
from smart_robustness.models.point_hh import (
    collapse_cell_spec_to_conserved_point,
    conserved_cortical_point_hh_parameters,
    create_conserved_cortical_point_hh_population,
)
from smart_robustness.models.ports import GapJunctionPortSpec
from smart_robustness.models.selective import (
    CORTICAL_CELL_CLASSES,
    make_selective_population_factory,
)
from smart_robustness.synapses import kinness_gap_total_conductance_nS


def _all_params() -> dict[str, dict[str, object]]:
    baseline = load_frozen_classic_baseline(
        "configs/baselines/classic_smart_calibrated_v1.yaml"
    )
    return {
        facts.canonical_name: first_order_population_parameters(
            facts,
            conventions=baseline.runtime_conventions(),
        )
        for facts in first_order_population_facts()
    }


def _channel_total(cell, attribute: str) -> float:
    return sum(
        float(getattr(compartment, attribute) or 0.0)
        * compartment.lateral_area_cm2
        for compartment in cell.compartments
    )


@pytest.mark.parametrize("cell_class", sorted(CORTICAL_CELL_CLASSES))
def test_point_hh_conserves_area_leak_and_active_conductances(cell_class) -> None:
    source = _all_params()[cell_class]["cell_spec"]
    point = collapse_cell_spec_to_conserved_point(source)

    assert tuple(compartment.name for compartment in point.compartments) == ("soma",)
    source_area = sum(c.lateral_area_cm2 for c in source.compartments)
    assert point.soma.lateral_area_cm2 == pytest.approx(source_area, rel=1e-12)
    for attribute in (
        "g_leak_mS_cm2",
        "g_na_mS_cm2",
        "g_k_mS_cm2",
        "g_ca_mS_cm2",
    ):
        assert _channel_total(point, attribute) == pytest.approx(
            _channel_total(source, attribute), rel=1e-12, abs=1e-18
        )
    source_leak_equilibrium = sum(
        c.g_leak_mS_cm2 * c.lateral_area_cm2 * c.e_leak_mV
        for c in source.compartments
    )
    point_leak_equilibrium = (
        point.soma.g_leak_mS_cm2
        * point.soma.lateral_area_cm2
        * point.soma.e_leak_mV
    )
    assert point_leak_equilibrium == pytest.approx(
        source_leak_equilibrium, rel=1e-12
    )


@pytest.mark.parametrize("cell_class", sorted(CORTICAL_CELL_CLASSES))
def test_point_hh_conserves_every_registered_port_total(cell_class) -> None:
    params = _all_params()[cell_class]
    transformed = conserved_cortical_point_hh_parameters(params)
    source = params["cell_spec"]
    point = transformed["cell_spec"]

    for key in ("synaptic_ports", "external_input_ports"):
        for old, new in zip(params[key], transformed[key], strict=True):
            old_total = (
                old.conductance_density_mS_cm2
                * source.compartment(old.compartment).lateral_area_cm2
            )
            new_total = (
                new.conductance_density_mS_cm2
                * point.compartment(new.compartment).lateral_area_cm2
            )
            assert new.compartment == "soma"
            assert new_total == pytest.approx(old_total, rel=1e-12, abs=1e-18)

    for old, new in zip(
        params["gap_junction_ports"],
        transformed["gap_junction_ports"],
        strict=True,
    ):
        old_compartment = source.compartment(old.compartment)
        new_compartment = point.soma
        old_total = kinness_gap_total_conductance_nS(
            old.conductance_density_mS_cm2,
            diameter_mm=old_compartment.diameter_mm,
            length_mm=old_compartment.length_mm,
        )
        new_total = kinness_gap_total_conductance_nS(
            new.conductance_density_mS_cm2,
            diameter_mm=new_compartment.diameter_mm,
            length_mm=new_compartment.length_mm,
        )
        assert new.compartment == "soma"
        assert new_total == pytest.approx(old_total, rel=1e-12, abs=1e-18)

    for old, new in zip(
        params["injection_ports"],
        transformed["injection_ports"],
        strict=True,
    ):
        old_area = source.compartment(old.compartment).lateral_area_cm2
        new_area = point.soma.lateral_area_cm2
        assert new.compartment == "soma"
        for old_value, new_value in zip(
            old.sensitivities_pA_cm2,
            new.sensitivities_pA_cm2,
            strict=True,
        ):
            assert new_value * new_area == pytest.approx(
                old_value * old_area, rel=1e-12, abs=1e-18
            )


def test_point_hh_factory_has_one_voltage_and_no_axial_edge() -> None:
    brian.start_scope()
    params = _all_params()["layer5_excitatory_v1"]
    point = create_conserved_cortical_point_hh_population(
        name="layer5_point_hh",
        size=1,
        params=params,
        brian=brian,
    )

    assert point.compartments == ("soma",)
    assert point.compiled.axial_parameter_names == ()
    assert "v_soma" in point.group.variables
    assert "v_proximal_dendrite" not in point.group.variables
    assert "v_distal_dendrite" not in point.group.variables
    assert "i_ahp" in point.group.variables


def test_point_hh_conserves_synthetic_kinness_gap_port() -> None:
    params = _all_params()["layer4_excitatory_v1"]
    old_port = GapJunctionPortSpec(
        name="gap_test",
        record_id="synthetic.precheck.gap",
        compartment="proximal_dendrite",
        conductance_density_mS_cm2=0.25,
    )
    transformed = conserved_cortical_point_hh_parameters(
        params | {"gap_junction_ports": (old_port,)}
    )
    new_port = transformed["gap_junction_ports"][0]
    source = params["cell_spec"].compartment(old_port.compartment)
    point = transformed["cell_spec"].soma

    old_total = kinness_gap_total_conductance_nS(
        old_port.conductance_density_mS_cm2,
        diameter_mm=source.diameter_mm,
        length_mm=source.length_mm,
    )
    new_total = kinness_gap_total_conductance_nS(
        new_port.conductance_density_mS_cm2,
        diameter_mm=point.diameter_mm,
        length_mm=point.length_mm,
    )
    assert new_port.compartment == "soma"
    assert new_total == pytest.approx(old_total, rel=1e-12)


def test_selective_point_hh_factory_leaves_thalamic_adapter_unchanged() -> None:
    params = _all_params()["thalamic_relay"]
    selective = make_selective_population_factory(
        create_conserved_cortical_point_hh_population,
        CORTICAL_CELL_CLASSES,
    )
    brian.start_scope()
    control = create_compartmental_hh_population(
        name="relay_control_manifest",
        size=1,
        params=params,
        brian=brian,
    )
    selected = selective(
        name="relay_selected_manifest",
        size=1,
        params=params,
        brian=brian,
    )

    assert selected.cell_spec == control.cell_spec
    assert selected.compiled == control.compiled
    assert set(selected.group.variables) == set(control.group.variables)


def test_point_hh_rejects_thalamic_alternative_and_nested_controls() -> None:
    params = _all_params()
    with pytest.raises(ValueError, match="registered cortical"):
        conserved_cortical_point_hh_parameters(params["thalamic_relay"])
    with pytest.raises(ValueError, match="classic HH"):
        conserved_cortical_point_hh_parameters(
            params["layer4_excitatory_v1"] | {"somatic_spike_model": "adex"}
        )
    with pytest.raises(ValueError, match="controlled by"):
        conserved_cortical_point_hh_parameters(
            params["layer4_excitatory_v1"]
            | {"disabled_nak_compartments": frozenset()}
        )


def test_kinness_gap_total_is_linear_in_registered_coefficient() -> None:
    unit = kinness_gap_total_conductance_nS(
        1.0,
        diameter_mm=0.05,
        length_mm=0.2,
    )
    scaled = kinness_gap_total_conductance_nS(
        math.pi,
        diameter_mm=0.05,
        length_mm=0.2,
    )
    assert scaled == pytest.approx(math.pi * unit)
