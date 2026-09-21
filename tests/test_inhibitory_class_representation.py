from __future__ import annotations

from pathlib import Path

import yaml

from smart_robustness.modeldb_projections import load_modeldb_first_order_catalog
from smart_robustness.models.modeldb112923 import first_order_population_facts

CORTICAL_INHIBITORY_POPULATIONS = {
    "layer23_inhibitory_v1",
    "layer4_inhibitory_v1",
}


def test_no_existing_cortical_inhibitory_output_has_a_distal_cortical_target() -> None:
    catalog = load_modeldb_first_order_catalog()
    cells = {
        fact.canonical_name: {compartment.name for compartment in fact.cell.compartments}
        for fact in first_order_population_facts()
    }
    cortical_gaba_outputs = [
        record
        for record in catalog.projections
        if record.source_population in CORTICAL_INHIBITORY_POPULATIONS
        and record.reversal_mV == -70.0
        and "excitatory" in record.target_population
    ]
    assert [record.id for record in cortical_gaba_outputs] == [
        "modeldb112923.projection.031",
        "modeldb112923.projection.036",
    ]
    assert all(record.target_compartment == "proximal_dendrite" for record in cortical_gaba_outputs)
    assert all(
        "distal_dendrite" not in cells[record.target_population]
        for record in cortical_gaba_outputs
    )


def test_distal_cortical_pyramidal_cells_have_no_cortical_gaba_route() -> None:
    catalog = load_modeldb_first_order_catalog()
    cells = {
        fact.canonical_name: {compartment.name for compartment in fact.cell.compartments}
        for fact in first_order_population_facts()
    }
    distal_cortical_targets = {
        name
        for name, compartments in cells.items()
        if "excitatory" in name and "distal_dendrite" in compartments
    }
    assert distal_cortical_targets == {
        "layer5_excitatory_v1",
        "layer6ii_excitatory_v1",
    }
    represented = {
        record.target_population
        for record in catalog.projections
        if record.source_population in CORTICAL_INHIBITORY_POPULATIONS
        and record.reversal_mV == -70.0
    }
    assert distal_cortical_targets.isdisjoint(represented)


def test_sst_representation_audit_requires_an_explicit_extension() -> None:
    path = Path(
        "docs/validation-results/"
        "post2008-sst-like-routing-representation-audit-983.yaml"
    )
    audit = yaml.safe_load(path.read_text())
    assert audit["network_execution_authorized"] is False
    assert audit["feasibility_result"] == {
        "existing_output_with_existing_distal_target": False,
        "one_field_target_reassignment_possible": False,
        "exact_resource_conserving_routing_only_arm_possible": False,
        "reason": audit["feasibility_result"]["reason"],
    }
    assert audit["decision"]["explicit_functional_class_extension_design_authorized"] is True
    assert audit["decision"]["vip_disinhibitory_network_execution_authorized"] is False
