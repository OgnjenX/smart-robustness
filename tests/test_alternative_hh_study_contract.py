from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def _yaml(path: str):
    return yaml.safe_load((ROOT / path).read_text())


def test_alternative_hh_registration_precedes_all_results() -> None:
    registration = _yaml(
        "docs/validation-results/neuron-model-alternative-hh-registration-920.yaml"
    )
    state = registration["implementation_state"]
    assert state["equations"] == "not-implemented"
    assert state["isolated_fit"] == "not-run"
    assert state["network_outcomes_observed"] is False
    assert registration["frozen_decisions"]["fitting_uses_network_outcomes"] is False


def test_alternative_hh_fit_has_sealed_holdout_and_exact_bounds() -> None:
    protocol = _yaml(
        "configs/models/alternative_hh_isolated_match_protocol_v1.yaml"
    )
    selection = protocol["sealed_level_selection"]
    assert set(selection["training_positions"]).isdisjoint(
        selection["sealed_holdout_positions"]
    )
    assert selection["alternative_outputs_used_to_select_levels"] is False
    assert protocol["candidate_generation"]["candidates_per_cell_class"] == 256
    assert set(protocol["candidate_generation"]["dimensions"]) == {
        "threshold_mV",
        "sodium_density_mS_cm2",
        "potassium_density_mS_cm2",
        "m_current_density_mS_cm2",
        "m_current_tau_max_ms",
    }
    assert protocol["selection"]["holdout_not_used_for_selection"] is True
    assert "all full-network outcomes" in protocol["forbidden_fit_inputs"]


def test_alternative_hh_network_progression_is_strictly_staged() -> None:
    study = _yaml("configs/robustness/alternative_hh_phase1_v1.yaml")
    progression = study["progression"]
    assert progression["stage_1"]["endpoint"] == "isolated-cell promotion gates"
    assert "Figure 6" in progression["stage_2"]["endpoint"]
    assert "Figure-6-passing" in progression["stage_3"]["rule"]
    assert "stage-3" in progression["stage_4"]["rule"]
    assert study["structural_identity"]["any_other_difference_invalidates_arm"] is True
    assert "modern-anatomy changes" in study["prohibited"]


def test_failed_isolated_gate_prohibits_alternative_hh_network_run() -> None:
    assessment = _yaml(
        "docs/validation-results/neuron-model-alternative-hh-isolated-fit-921.yaml"
    )
    assert assessment["execution"]["network_outcomes_used"] is False
    assert assessment["execution"]["holdout_used_for_selection"] is False
    assert assessment["assessment"]["every_cortical_class_promoted"] is False
    assert assessment["assessment"]["promoted_class_count"] == 2
    assert assessment["decision"]["matched_parameter_map_frozen"] is False
    assert assessment["decision"]["figure6_network_arms_authorized"] is False
    assert assessment["alternative_network_outcomes_run_before_assessment"] is False
