from __future__ import annotations

import pytest

from smart_robustness.models.ports import (
    chemical_port,
    external_ports_for_target,
    gap_ports_for_target,
    ligand_conductance_density_mS_cm2,
    ports_for_target,
)
from smart_robustness.projections import SUPPLEMENTARY_TABLE3


def test_all_chemical_records_compile_to_unique_target_ports() -> None:
    chemical = tuple(
        record for record in SUPPLEMENTARY_TABLE3.records if record.kind.value == "chemical"
    )
    ports = tuple(chemical_port(record, index) for index, record in enumerate(chemical))
    assert len(ports) == 49
    assert len({port.record_id for port in ports}) == 49
    assert all(port.rise_ms > 0 and port.fall_ms > 0 for port in ports)


def test_ligand_channel_ps_and_million_per_cm2_weight_convert_to_density() -> None:
    assert ligand_conductance_density_mS_cm2(1.0) == pytest.approx(0.001)
    record = SUPPLEMENTARY_TABLE3.by_id("relay.distal_dendrite.from_l6ii.nmda")
    port = chemical_port(record, 0)
    assert port.conductance_density_mS_cm2 == pytest.approx(
        ligand_conductance_density_mS_cm2(float(record.parsed.conductance_pS))
    )


def test_nmda_port_retains_voltage_block_and_equal_tau_is_supported() -> None:
    nmda = chemical_port(SUPPLEMENTARY_TABLE3.by_id("relay.distal_dendrite.from_l6ii.nmda"), 0)
    assert nmda.voltage_block
    equal_tau = chemical_port(
        SUPPLEMENTARY_TABLE3.by_id("l23_i.proximal_dendrite.from_l23_e.ampa"), 1
    )
    assert equal_tau.rise_ms == equal_tau.fall_ms == 2.0


def test_layer4_target_ports_include_all_source_records() -> None:
    ports = ports_for_target(SUPPLEMENTARY_TABLE3.records, "layer4_excitatory_v1")
    assert len(ports) == 4
    assert {port.compartment for port in ports} == {"proximal_dendrite"}


def test_nonchemical_record_is_rejected_as_port() -> None:
    with pytest.raises(ValueError, match="only chemical"):
        chemical_port(SUPPLEMENTARY_TABLE3.by_id("l4_i.proximal_dendrite.from_l4_i.gap"), 0)


def test_all_four_gap_junction_records_compile_to_ports() -> None:
    populations = ("layer4_inhibitory_v1", "thalamic_interneuron", "trn")
    ports = tuple(
        port
        for population in populations
        for port in gap_ports_for_target(SUPPLEMENTARY_TABLE3.records, population)
    )
    assert len(ports) == 4


def test_both_bottom_up_external_inputs_compile_to_conductance_ports() -> None:
    relay = external_ports_for_target(SUPPLEMENTARY_TABLE3.records, "thalamic_relay")
    matrix = external_ports_for_target(SUPPLEMENTARY_TABLE3.records, "thalamic_matrix")
    assert len(relay) == len(matrix) == 1
    assert relay[0].compartment == matrix[0].compartment == "proximal_dendrite"
    assert relay[0].conductance_density_mS_cm2 == pytest.approx(1.1)
    assert matrix[0].conductance_density_mS_cm2 == pytest.approx(1.1)
