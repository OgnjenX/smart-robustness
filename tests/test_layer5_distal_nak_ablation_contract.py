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
