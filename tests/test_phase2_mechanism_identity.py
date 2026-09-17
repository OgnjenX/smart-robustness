from pathlib import Path

import yaml

from smart_robustness.models.modeldb112923 import first_order_population_facts
from smart_robustness.models.selective import CORTICAL_CELL_CLASSES

ROOT = Path(__file__).parents[1]


def test_only_layer5_distal_dendrite_has_active_nak_in_cortex() -> None:
    active = [
        (facts.canonical_name, compartment.name)
        for facts in first_order_population_facts()
        if facts.canonical_name in CORTICAL_CELL_CLASSES
        for compartment in facts.cell.compartments
        if compartment.name != "soma"
        and (
            compartment.g_na_mS_cm2 is not None
            or compartment.g_k_mS_cm2 is not None
        )
    ]
    assert active == [("layer5_excitatory_v1", "distal_dendrite")]


def test_passivization_alias_audit_prohibits_duplicate_network_run() -> None:
    audit = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/"
            "mechanism-cortical-dendrite-passivization-alias-audit-926.yaml"
        ).read_text()
    )
    assert audit["identity_result"]["manipulated_variables_identical"] is True
    assert audit["identity_result"]["added_or_removed_variables_beyond_artifacts_923_925"] == []
    assert audit["decision"]["duplicate_network_run_authorized"] is False
    assert audit["decision"]["reuse_completed_evidence"] is True
