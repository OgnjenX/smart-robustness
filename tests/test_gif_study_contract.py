from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def _yaml(path: str):
    return yaml.safe_load((ROOT / path).read_text())


def test_gif_study_is_registered_before_network_outcomes() -> None:
    registration = _yaml(
        "docs/validation-results/neuron-model-gif-study-registration-908.yaml"
    )
    phase = _yaml("configs/robustness/gif_phase1_v1.yaml")
    assert registration["status"] == "registered-before-any-gif-network-outcome"
    assert registration["execution_state"]["network_outcomes_observed"] is False
    assert phase["control_manifest"] == "configs/baselines/classic_smart_calibrated_v1.yaml"
    assert phase["modern_anatomy_changes"] == "prohibited-in-phase-1"


def test_stochastic_inference_and_progression_are_fixed() -> None:
    phase = _yaml("configs/robustness/gif_phase1_v1.yaml")
    seeds = phase["network_seed_ensemble"]["seeds"]
    rule = phase["figure6_progression_rule"]
    assert len(seeds) == len(set(seeds)) == 20
    assert rule["minimum_successful_trials"] == 16
    assert rule["total_trials"] == 20
    assert rule["no_seed_replacement"] is True


def test_gif_fit_has_disjoint_current_and_seed_holdouts() -> None:
    protocol = _yaml("configs/models/gif_isolated_match_protocol_v1.yaml")
    levels = protocol["sealed_level_selection"]
    stochastic = protocol["stochastic_repetitions"]
    assert set(levels["training_positions"]).isdisjoint(
        levels["sealed_current_holdout_positions"]
    )
    assert set(stochastic["training_seeds"]).isdisjoint(
        stochastic["sealed_seed_holdout"]
    )
    assert "all full-network results" in protocol["forbidden_fit_inputs"]


def test_literature_arm_declares_thalamic_transfer_limitation() -> None:
    manifest = _yaml("configs/models/gif_parameter_invariant_v1.yaml")
    assert manifest["passive_parameters_from_smart"] is True
    assert manifest["network_outcome_fitting"] == "prohibited"
    assert "not a claim" in manifest["provenance"]["qualification"]


def test_isolated_fit_is_sealed_before_network_holdout() -> None:
    assessment = _yaml(
        "docs/validation-results/neuron-model-gif-isolated-fit-assessment-909.yaml"
    )
    registration = _yaml(
        "docs/validation-results/neuron-model-gif-figure6-registration-910.yaml"
    )
    parameters = _yaml("configs/models/gif_current_step_matched_v1.yaml")
    assert assessment["summary"]["all_classes_promoted"] is True
    assert parameters["claims"]["network_behavior_preserved"] == "not-yet-observed"
    assert registration["knowledge_boundary"]["gif_network_outcomes_observed_at_registration"] is False
    assert registration["knowledge_boundary"]["parameter_maps_frozen"] is True


def test_completed_ensemble_blocks_downstream_protocols() -> None:
    result = _yaml(
        "docs/validation-results/neuron-model-gif-figure6-assessment-911.yaml"
    )
    assert result["registered_seeds_completed_per_arm"] == 20
    assert result["arms"]["literature"]["full_figure6_successes"] == 0
    assert result["arms"]["current_step_matched"]["full_figure6_successes"] == 0
    assert result["arms"]["current_step_matched"][
        "individual_gate_success_counts"
    ]["top_down_horizontal_contrast"] == 18
    assert result["decision"]["downstream_figures_7_10_14_15_16_authorized"] is False
