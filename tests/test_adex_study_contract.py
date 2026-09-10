from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def _yaml(path: str):
    return yaml.safe_load((ROOT / path).read_text())


def test_phase1_changes_only_neuron_model_before_modern_anatomy() -> None:
    phase = _yaml("configs/robustness/adex_phase1_v1.yaml")
    assert phase["control_manifest"] == "configs/baselines/classic_smart_calibrated_v1.yaml"
    assert phase["modern_anatomy_changes"] == "prohibited-in-phase-1"
    assert phase["network_level_parameter_rescue"] == "prohibited-in-phase-1"
    arm = phase["arms"]["adex_current_step_matched"]
    assert arm["matching_protocol"].endswith(
        "adex_isolated_match_protocol_generation2_v1.yaml"
    )
    assert arm["parameters"].endswith("adex_current_step_matched_v1.yaml")


def test_v3_fit_separates_training_holdout_and_forbids_network_outputs() -> None:
    protocol = _yaml("configs/models/adex_isolated_match_protocol_v3.yaml")
    selection = protocol["sealed_level_selection"]
    assert set(selection["training_positions"]).isdisjoint(
        selection["sealed_holdout_positions"]
    )
    assert selection["adex_outputs_used_to_select_levels"] is False
    assert "all full-network results" in protocol["forbidden_fit_inputs"]
    assert protocol["selection"]["holdout_not_used_for_selection"] is True


def test_failed_literature_arm_cannot_feed_downstream_protocols() -> None:
    result = _yaml(
        "docs/validation-results/neuron-model-adex-literature-figure6-891.yaml"
    )
    assert result["arm"]["parameters_registered_before_network_result"] is True
    assert result["figure6"]["all_gates_pass"] is False
    assert result["decision"]["downstream_learned_state_protocols_authorized"] is False
    assert result["decision"]["art_core_endpoint_pass"] is False


def test_current_step_matched_arm_discloses_failed_isolated_gates() -> None:
    parameters = _yaml("configs/models/adex_current_step_matched_v1.yaml")
    registration = _yaml(
        "docs/validation-results/neuron-model-adex-current-step-arm-registration-896.yaml"
    )
    assert parameters["claims"]["current_step_training_and_holdout_promoted"] is True
    assert parameters["claims"]["complete_isolated_phenotype_matched"] is False
    assert parameters["claims"]["network_behavior_preserved"] == "not-yet-observed"
    assert registration["interpretation_contract"]["no_parameter_changes_after_network_result"]
