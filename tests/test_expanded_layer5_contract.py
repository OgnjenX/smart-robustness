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
    assert study["transformation"]["ports"]["invariant"] == (
        "each port's total maximum conductance is conserved independently"
    )
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


def test_expanded_layer5_port_rule_was_amended_before_outcomes() -> None:
    amendment = _yaml(
        "docs/validation-results/mechanism-expanded-layer5-port-amendment-932a.yaml"
    )
    assert amendment["status"] == "amended-before-implementation-or-outcomes"
    assert amendment["reason"]["connected_network_outcomes_observed"] is False
    assert amendment["amended_rule"]["proximal_to_basal_scale"] == 2.0
    assert "conserved" in amendment["amended_rule"]["invariant"]


def test_expanded_layer5_prechecks_authorize_only_figure6() -> None:
    result = _yaml(
        "docs/validation-results/mechanism-expanded-layer5-precheck-933.yaml"
    )
    assert result["raw_result"] == {
        "path": "results/expanded-layer5-prechecks-933.yaml",
        "sha256": (
            "e8730f2bb6b18d5bc9c7a4d014b0b08f"
            "c342316b0339d1fd215842f8c317bea5"
        ),
        "bytes": 7436,
        "committed": "false-generated-result",
    }
    assert result["execution"]["connected_network_outcomes_observed"] is False
    assert result["structural_assessment"]["changed_adapter_classes"] == [
        "layer5_excitatory_v1"
    ]
    assert result["structural_assessment"][
        "maximum_membrane_conservation_relative_error"
    ] == 0.0
    assert result["structural_assessment"][
        "maximum_port_conservation_relative_error"
    ] == 0.0
    assert result["decision"] == {
        "stage_1_pass": True,
        "figure6_authorized": True,
        "later_network_stages_authorized": False,
    }


def test_expanded_layer5_figure6_passes_exactly_before_stage3() -> None:
    result = _yaml(
        "docs/validation-results/mechanism-expanded-layer5-figure6-assessment-934.yaml"
    )
    assert result["raw_result"] == {
        "path": "results/expanded-layer5-figure6-934.yaml",
        "sha256": (
            "5fdf735abbc958d8d44b869250422f7a"
            "2fd406f1ca186aaabb8f8c4553543775"
        ),
        "bytes": 971804,
        "committed": "false-generated-result",
    }
    for arm in ("classic_control", "expanded_layer5_branched"):
        assert result["arms"][arm]["exact_repeat"] is True
        assert result["arms"][arm]["all_trials_pass"] is True
        assert all(result["arms"][arm]["six_gates"].values())
    assert result["decision"] == {
        "classic_sentinel_pass": True,
        "expanded_layer5_figure6_pass": True,
        "stage3_match_mismatch_and_reset_authorized": True,
        "spectral_and_higher_order_stages_authorized": False,
    }


def test_expanded_layer5_stage3_closes_at_exact_figure7_failure() -> None:
    result = _yaml(
        "docs/validation-results/mechanism-expanded-layer5-stage3-assessment-935.yaml"
    )
    assert result["raw_result"] == {
        "path": "results/expanded-layer5-stage3-935.yaml",
        "sha256": (
            "e9904b4384d6e378b31560ed9fb91834"
            "11e2f83e452bca099e2a36185d1b5b2d"
        ),
        "bytes": 12704,
        "committed": "false-generated-result",
    }
    assert result["classic_control"]["all_stage3_gates_pass"] is True
    expanded = result["expanded_layer5_branched"]
    assert expanded["exact_repeat"] is True
    assert expanded["passed_figure7_gate_count"] == 7
    assert expanded["failed_gates"] == ["mismatch_nonspecific_events"]
    assert expanded["match_mismatch_nonspecific_events"] == [4, 6]
    assert expanded["figure10_executed"] is False
    assert result["decision"] == {
        "arm_closed_at_first_failed_stage": True,
        "figure10_expanded_layer5_authorized": False,
        "spectral_and_higher_order_stages_authorized": False,
        "neuron_model_and_mechanism_phase_complete": True,
    }
