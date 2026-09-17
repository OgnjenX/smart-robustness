from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def _yaml(path: str):
    return yaml.safe_load((ROOT / path).read_text())


def test_layer5_distal_nak_ablation_has_one_exact_intervention() -> None:
    study = _yaml("configs/robustness/layer5_distal_nak_ablation_v1.yaml")
    arm = study["arms"]["layer5_distal_nak_disabled"]
    assert arm["target_cell_classes"] == ["layer5_excitatory_v1"]
    assert arm["target_compartments"] == ["distal_dendrite"]
    assert arm["disabled_intrinsic_currents"] == [
        "fast_na",
        "delayed_rectifier_k",
    ]
    assert study["intervention_identity"]["compensation_or_refitting"] == "prohibited"


def test_layer5_distal_nak_ablation_is_registered_before_outcomes() -> None:
    registration = _yaml(
        "docs/validation-results/mechanism-layer5-distal-nak-registration-922.yaml"
    )
    state = registration["implementation_state_at_registration"]
    assert state["ablation_factory"] == "not-implemented"
    assert state["equation_option"] == "not-implemented"
    assert state["isolated_checks"] == "not-run"
    assert state["connected_network_outcomes_observed"] is False
    assert registration["frozen_decisions"]["fitting_permitted"] is False


def test_layer5_distal_nak_progression_stops_at_first_failure() -> None:
    study = _yaml("configs/robustness/layer5_distal_nak_ablation_v1.yaml")
    progression = study["progression"]
    assert progression["stage_1"]["endpoint"] == "structural and isolated finite checks"
    assert "six-gate" in progression["stage_2"]["endpoint"]
    assert "stage 2" in progression["stage_3"]["rule"]
    assert "stage-3" in progression["stage_4"]["rule"]
    assert "failed prerequisite" in study["stopping_rules"][0]


def test_layer5_distal_nak_prechecks_authorize_figure6_without_network_data() -> None:
    assessment = _yaml(
        "docs/validation-results/mechanism-layer5-distal-nak-precheck-923.yaml"
    )
    assert assessment["structural_assessment"]["all_checks_pass"] is True
    assert assessment["isolated_assessment"]["classic_control_finite"] is True
    assert assessment["isolated_assessment"]["ablation_finite"] is True
    assert assessment["decision"]["stage_1_pass"] is True
    assert assessment["decision"]["figure6_authorized"] is True
    assert assessment["decision"]["compensation_or_refitting_used"] is False
    assert assessment["execution"]["network_outcomes_observed"] is False


def test_layer5_distal_nak_figure6_pass_authorizes_only_stage3() -> None:
    assessment = _yaml(
        "docs/validation-results/"
        "mechanism-layer5-distal-nak-figure6-assessment-924.yaml"
    )
    assert assessment["arms"]["classic_control"]["exact_repeat"] is True
    assert assessment["arms"]["classic_control"]["all_trials_pass"] is True
    assert assessment["arms"]["layer5_distal_nak_disabled"]["exact_repeat"] is True
    assert assessment["arms"]["layer5_distal_nak_disabled"]["all_trials_pass"] is True
    decision = assessment["decision"]
    assert decision["stage3_match_mismatch_and_reset_authorized"] is True
    assert decision["spectral_and_higher_order_stages_authorized"] is False
    assert decision["compensation_or_refitting_used"] is False
