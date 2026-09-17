from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def _yaml(path: str):
    return yaml.safe_load((ROOT / path).read_text())


def test_expanded_layer5_registration_freezes_branched_conserved_mapping() -> None:
    study = _yaml("configs/robustness/expanded_layer5_branched_v1.yaml")
    assert study["arms"]["expanded_layer5_branched"]["axial_edges"] == [
        ["soma", "basal_dendrite"],
        ["soma", "proximal_apical_dendrite"],
        ["proximal_apical_dendrite", "distal_tuft"],
    ]
    proximal = study["transformation"]["original_proximal_dendrite"]
    assert proximal["area_fraction_basal"] == 0.5
    assert proximal["area_fraction_proximal_apical"] == 0.5
    assert study["transformation"]["axial_coupling"]["conserved"] is False
    assert study["execution"]["exact_reruns_per_arm"] == 2
    assert "fitting the 0.5 split" in study["prohibited"][0]


def test_expanded_layer5_source_audit_keeps_modern_anatomy_out() -> None:
    audit = _yaml(
        "docs/validation-results/mechanism-expanded-layer5-source-audit-931.yaml"
    )
    assert audit["decision"]["fitting_to_network_outcomes"] == "prohibited"
    assert audit["decision"]["proximal_area_split"].startswith("equal 0.5")
    excluded = audit["phase_boundary"]["excluded_until_modern_anatomy_phases"]
    assert "direct visual thalamus-to-layer-5 input" in excluded
    assert "new apical calcium or HCN mechanisms" in excluded


def test_expanded_layer5_is_registered_before_implementation_or_outcomes() -> None:
    registration = _yaml(
        "docs/validation-results/mechanism-expanded-layer5-registration-932.yaml"
    )
    state = registration["implementation_state_at_registration"]
    assert state["explicit_axial_tree_support"] == "not-implemented"
    assert state["expanded_layer5_factory"] == "not-implemented"
    assert state["conservation_checks"] == "not-run"
    assert state["connected_network_outcomes_observed"] is False
    assert registration["frozen_decisions"]["new_active_intrinsic_mechanisms"] == []
