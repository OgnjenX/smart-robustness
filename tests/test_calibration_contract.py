import hashlib
import json
from pathlib import Path

import pytest
import yaml

from smart_robustness.validation.calibration import (
    TrnStageAResult,
    load_calibration_contract,
    runtime_conventions_for_candidate,
)

ROOT = Path(__file__).parents[1]
CONTRACT_PATH = ROOT / "configs/calibration/classic_uncertainty_space.yaml"
TRN_SURVIVOR_PATH = ROOT / "configs/calibration/trn_stage_a_survivor_v1.yaml"
NETWORK_CALIBRATION_PATH = (
    ROOT / "docs/validation-results/calibration-network-trn-survivor-121.yaml"
)
GAUSSIAN_VARIANCE_PROFILE_PATH = (
    ROOT / "configs/calibration/trn_stage_a_survivor_gaussian_variance_v1.yaml"
)
FIGURE8_VOLTAGE_AUDIT_PATH = (
    ROOT / "docs/validation-results/figure8-voltage-observable-audit-124.yaml"
)
FIGURE6_RELAY_BALANCE_PATH = (
    ROOT / "docs/validation-results/figure6-relay-current-balance-125.yaml"
)
FIGURE6_RELAY_SCREEN_PATH = (
    ROOT / "docs/validation-results/figure6-relay-survivor-screen-126.yaml"
)
FIGURE6_RELAY_EQUILIBRATION_PATH = (
    ROOT / "docs/validation-results/figure6-relay-equilibration-127.yaml"
)
FIGURE6_SOURCE_HYBRID_PROFILE_PATH = (
    ROOT / "configs/calibration/figure6_relay_source_hybrid_v1.yaml"
)
FIGURE6_SOURCE_HYBRID_RESULT_PATH = (
    ROOT / "docs/validation-results/figure6-relay-source-hybrid-128.yaml"
)
FIGURE6_RELAY_AXIAL_HYBRID_PROFILE_PATH = (
    ROOT / "configs/calibration/figure6_relay_axial_source_hybrid_v1.yaml"
)
FIGURE6_RELAY_AXIAL_HYBRID_RESULT_PATH = (
    ROOT / "docs/validation-results/figure6-relay-axial-source-hybrid-129.yaml"
)
FIGURE6_RELAY_AXIAL_DECOMPOSITION_PATH = (
    ROOT / "docs/validation-results/figure6-relay-axial-map-decomposition-130.yaml"
)
FIGURE6_LEADING_ALTERNATIVES_PATH = (
    ROOT / "docs/validation-results/figure6-leading-source-alternatives-131.yaml"
)
FIGURE6_LEARNING_PHASE_PATH = (
    ROOT / "docs/validation-results/figure6-top-down-learning-phase-132.yaml"
)
FIGURE6_PROJECTION_BOUNDS_PROFILE_PATH = (
    ROOT / "configs/calibration/figure6_projection_level_learning_bounds_v1.yaml"
)
FIGURE6_PROJECTION_BOUNDS_RESULT_PATH = (
    ROOT / "docs/validation-results/figure6-projection-level-learning-bounds-133.yaml"
)
FIGURE6_PROJECTION_D_PROFILE_PATH = (
    ROOT / "configs/calibration/figure6_projection_depression_scale_v1.yaml"
)
FIGURE6_PROJECTION_D_RESULT_PATH = (
    ROOT / "docs/validation-results/figure6-projection-depression-scale-134.yaml"
)
FIGURE6_VOLLEY_DECOMPOSITION_PATH = (
    ROOT / "docs/validation-results/figure6-teaching-volley-decomposition-135.yaml"
)
FIGURE6_POPULATION_AXIAL_PROFILE_PATH = (
    ROOT / "configs/calibration/figure6_population_resolved_axial_v1.yaml"
)
FIGURE6_POPULATION_AXIAL_RESULT_PATH = (
    ROOT / "docs/validation-results/figure6-population-resolved-axial-136.yaml"
)
FIGURE6_POPULATION_AXIAL_AMPLITUDE_AUDIT_PATH = (
    ROOT / "docs/validation-results/figure6-population-axial-amplitude-audit-140.yaml"
)
FIGURE6_POPULATION_AXIAL_LEARNING_PHASE_PATH = (
    ROOT / "docs/validation-results/figure6-population-axial-learning-phase-141.yaml"
)
FIGURE6_LEARNING_THRESHOLD_ASSESSMENT_PATH = (
    ROOT / "docs/validation-results/figure6-learning-threshold-coordinate-assessment-145.yaml"
)
FIGURE6_LEARNING_RULE_ASSESSMENT_PATH = (
    ROOT / "docs/validation-results/figure6-learning-rule-coordinate-assessment-148.yaml"
)
FIGURE6_DUAL_AND_LEAK_PHASE_PATH = (
    ROOT / "docs/validation-results/figure6-dual-and-leak-learning-phase-149.yaml"
)
FIGURE6_RELAY_WAVEFORM_PATH = (
    ROOT / "docs/validation-results/figure6-relay-waveform-nak-audit-150.yaml"
)
FIGURE6_UPWARD_TIMESTAMP_PATH = (
    ROOT / "docs/validation-results/figure6-learning-timestamp-upward-151.yaml"
)
FIGURE6_LEAK_PLUS30_PATH = (
    ROOT / "docs/validation-results/figure6-learning-coordinate-leak-plus30-152.yaml"
)
FIGURE7_POPULATION_AXIAL_RESULT_PATH = (
    ROOT / "docs/validation-results/calibration-figure6-figure7-population-axial-137.yaml"
)
FIGURE7_THALAMOCORTICAL_AXIAL_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_thalamocortical_axial_v1.yaml"
)
FIGURE6_THALAMOCORTICAL_AXIAL_RESULT_PATH = (
    ROOT / "docs/validation-results/figure6-thalamocortical-axial-138.yaml"
)
FIGURE7_POPULATION_TRN_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_population_resolved_trn_v1.yaml"
)
FIGURE6_POPULATION_TRN_RESULT_PATH = (
    ROOT / "docs/validation-results/figure6-population-resolved-trn-139.yaml"
)
FIGURE7_TRN_POTASSIUM_SCREEN_PATH = (
    ROOT / "docs/validation-results/figure7-trn-potassium-source-screen-166.yaml"
)
FIGURE7_TRN_CALCIUM_SCREEN_PATH = (
    ROOT / "docs/validation-results/figure7-trn-calcium-source-screen-167.yaml"
)
FIGURE7_TRN_CALCIUM_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_trn_calcium_reversal_v1.yaml"
)
FIGURE7_TRN_CALCIUM_PAIR_PATH = (
    ROOT
    / "docs/validation-results/figure7-trn-calcium-reversal-simultaneous-pair-168.yaml"
)
FIGURE7_TRN_DENDRITIC_CALCIUM_SCREEN_PATH = (
    ROOT
    / "docs/validation-results/figure7-trn-dendritic-calcium-source-screen-169.yaml"
)
FIGURE7_TRN_DENSITY_GRID_PROFILE_PATH = (
    ROOT / "configs/calibration/trn_dendritic_density_behavior_grid_v1.yaml"
)
FIGURE7_TRN_DENSITY_CUE_GRID_PATH = (
    ROOT / "docs/validation-results/figure7-trn-density-cue-grid-170.yaml"
)
FIGURE7_TRN_DENSITY_MATCH_GRID_PATH = (
    ROOT / "docs/validation-results/figure7-trn-density-match-grid-171.yaml"
)
FIGURE7_TRN_SOURCE_TOPOLOGY_AUDIT_PATH = (
    ROOT / "docs/validation-results/figure7-trn-source-topology-audit-172.yaml"
)
FIGURE7_TRN_AXIAL_GRID_PROFILE_PATH = (
    ROOT / "configs/calibration/trn_soma_proximal_axial_behavior_grid_v1.yaml"
)
FIGURE7_TRN_AXIAL_CUE_GRID_PATH = (
    ROOT / "docs/validation-results/figure7-trn-axial-cue-grid-173.yaml"
)
FIGURE7_TRN_AXIAL_MATCH_GRID_PATH = (
    ROOT / "docs/validation-results/figure7-trn-axial-match-grid-174.yaml"
)
FIGURE7_TRN_EVENT_OFFSET_PROFILE_PATH = (
    ROOT / "configs/calibration/trn_event_offset_behavior_grid_v1.yaml"
)
FIGURE7_TRN_EVENT_OFFSET_CUE_PATH = (
    ROOT / "docs/validation-results/figure7-trn-event-offset-cue-grid-175.yaml"
)
FIGURE7_TRN_EVENT_OFFSET_MATCH_PATH = (
    ROOT / "docs/validation-results/figure7-trn-event-offset-match-grid-176.yaml"
)
FIGURE7_TRN_DENSITY_EVENT_OFFSET_PROFILE_PATH = (
    ROOT / "configs/calibration/trn_density_event_offset_cross_v1.yaml"
)
FIGURE7_TRN_DENSITY_EVENT_OFFSET_CUE_PATH = (
    ROOT
    / "docs/validation-results/figure7-trn-density-event-offset-cue-grid-177.yaml"
)
FIGURE7_TRN_EVENT_BLEND_PROFILE_PATH = (
    ROOT / "configs/calibration/trn_soma_proximal_event_blend_v1.yaml"
)
FIGURE7_TRN_EVENT_BLEND_CUE_PATH = (
    ROOT / "docs/validation-results/figure7-trn-event-blend-cue-grid-178.yaml"
)
FIGURE7_TRN_EVENT_BLEND_MATCH_PATH = (
    ROOT / "docs/validation-results/figure7-trn-event-blend-match-grid-179.yaml"
)
FIGURE7_TRN_EVENT_BLEND_PAIR_PATH = (
    ROOT / "docs/validation-results/figure7-trn-event-blend-pair-180.yaml"
)
FIGURE7_PROTOCOL_GATE_AUDIT_PATH = (
    ROOT / "docs/validation-results/figure7-protocol-gate-audit-181.yaml"
)
FIGURE7_EVENT_BLEND_CURRENT_PROFILE_PATH = (
    ROOT / "configs/calibration/figure7_event_blend_top_down_grid_v1.yaml"
)
FIGURE7_EVENT_BLEND_CURRENT_CUE_PATH = (
    ROOT / "docs/validation-results/figure7-event-blend-current-cue-grid-182.yaml"
)
FIGURE7_EVENT_BLEND_CURRENT_MATCH_PATH = (
    ROOT / "docs/validation-results/figure7-event-blend-current-match-grid-183.yaml"
)
FIGURE7_EVENT_BLEND_CURRENT_PAIR_PATH = (
    ROOT / "docs/validation-results/figure7-event-blend-current-pair-300ms-184.yaml"
)
FIGURE7_NONSPECIFIC_EVENT_BLEND_PROFILE_PATH = (
    ROOT / "configs/calibration/nonspecific_soma_proximal_event_blend_v1.yaml"
)
FIGURE7_NONSPECIFIC_EVENT_BLEND_CUE_PATH = (
    ROOT / "docs/validation-results/figure7-nonspecific-blend-cue-grid-185.yaml"
)
FIGURE7_NONSPECIFIC_EVENT_BLEND_MISMATCH_PATH = (
    ROOT / "docs/validation-results/figure7-nonspecific-blend-mismatch-grid-186.yaml"
)
FIGURE7_NONSPECIFIC_EVENT_BLEND_COMPARISON_PATH = (
    ROOT
    / "docs/validation-results/figure7-nonspecific-blend-match-comparison-187.yaml"
)
FIGURE7_FEEDBACK_ARRIVAL_AUDIT_PATH = (
    ROOT / "docs/validation-results/figure7-feedback-arrival-order-audit-188.yaml"
)
FIGURE7_FEEDBACK_ARRIVAL_GRID_PATH = (
    ROOT / "docs/validation-results/figure7-feedback-arrival-alignment-grid-189.yaml"
)
FIGURE7_TRN_BLEND_ARRIVAL_MATCH_PATH = (
    ROOT
    / "docs/validation-results/figure7-trn-event-blend-arrival-match-grid-190.yaml"
)
FIGURE7_TRN_BLEND_ARRIVAL_MISMATCH_PATH = (
    ROOT
    / "docs/validation-results/figure7-trn-event-blend-arrival-mismatch-191.yaml"
)
FIGURE7_INHIBITORY_ARRIVAL_PATH = (
    ROOT / "docs/validation-results/figure7-inhibitory-arrival-alignment-192.yaml"
)
FIGURE6_SOURCE_STRENGTH_REASSESSMENT_PATH = (
    ROOT / "docs/validation-results/figure6-source-strength-reassessment-193.yaml"
)
FIGURE7_TRN_DETECTOR_CYCLE_PATH = (
    ROOT / "docs/validation-results/figure7-trn-detector-cycle-diagnostic-194.yaml"
)
FIGURE7_SAME_NETWORK_DETECTOR_PATH = (
    ROOT / "docs/validation-results/figure7-same-network-detector-diagnostic-195.yaml"
)
FIGURE6_RELAY_DETECTOR_CYCLE_PATH = (
    ROOT / "docs/validation-results/figure6-relay-detector-cycle-audit-196.yaml"
)
FIGURE7_TRN_SOMA_POTASSIUM_PROFILE_PATH = (
    ROOT / "configs/calibration/trn_soma_potassium_behavior_grid_v1.yaml"
)
FIGURE7_TRN_SOMA_POTASSIUM_STAGE1_PATH = (
    ROOT / "docs/validation-results/figure7-trn-soma-potassium-stage1-197.yaml"
)
LEGACY_SOURCE_RECOVERY_FOLLOWUP_PATH = (
    ROOT / "docs/validation-results/legacy-source-recovery-followup-198.yaml"
)
FIGURE7_TRN_SOMA_SODIUM_PROFILE_PATH = (
    ROOT / "configs/calibration/trn_soma_sodium_behavior_grid_v1.yaml"
)
FIGURE7_TRN_SOMA_SODIUM_STAGE1_PATH = (
    ROOT / "docs/validation-results/figure7-trn-soma-sodium-stage1-199.yaml"
)
FIGURE7_TRN_DETECTOR_HYSTERESIS_PROFILE_PATH = (
    ROOT / "configs/calibration/trn_detector_hysteresis_behavior_grid_v1.yaml"
)
FIGURE7_TRN_DETECTOR_HYSTERESIS_STAGE1_PATH = (
    ROOT / "docs/validation-results/figure7-trn-detector-hysteresis-stage1-200.yaml"
)
FIGURE6_TRN_DETECTOR_HYSTERESIS_PREREQUISITE_PATH = (
    ROOT
    / "docs/validation-results/figure6-trn-detector-hysteresis-prerequisite-201.yaml"
)
FIGURE6_TRN_GABA_TRANSFER_GRID_PATH = (
    ROOT / "docs/validation-results/figure6-trn-gaba-transfer-grid-202.yaml"
)
FIGURE7_TRN_GABA_TRANSFER_MATCH_PATH = (
    ROOT / "docs/validation-results/figure7-trn-gaba-transfer-match-203.yaml"
)
FIGURE7_TRN_GABA_TRANSFER_PAIR_PATH = (
    ROOT / "docs/validation-results/figure7-trn-gaba-transfer-pair-204.yaml"
)
FIGURE6_TRN_GABA_COMPARTMENT_SOURCE_PATH = (
    ROOT
    / "docs/validation-results/figure6-trn-gaba-compartment-source-endpoint-205.yaml"
)
FIGURE6_TRN_GABA_COMPARTMENT_INTERMEDIATE_PATH = (
    ROOT
    / "docs/validation-results/figure6-trn-gaba-compartment-intermediate-grid-206.yaml"
)
FIGURE7_TRN_GABA_COMPARTMENT_MATCH_PATH = (
    ROOT
    / "docs/validation-results/figure7-trn-gaba-compartment-match-207.yaml"
)
FIGURE7_TRN_GABA_COMPARTMENT_PAIR_PATH = (
    ROOT
    / "docs/validation-results/figure7-trn-gaba-compartment-pair-208.yaml"
)
FIGURE6_PROJECTION022_SOURCE_RESOLUTION_PATH = (
    ROOT
    / "docs/validation-results/figure6-projection022-source-resolution-209.yaml"
)
FIGURE7_PROJECTION022_SOURCE_RESOLUTION_MATCH_PATH = (
    ROOT
    / "docs/validation-results/figure7-projection022-source-resolution-match-210.yaml"
)
FIGURE7_PROJECTION022_SOURCE_RESOLUTION_PAIR_PATH = (
    ROOT
    / "docs/validation-results/figure7-projection022-source-resolution-pair-211.yaml"
)
FIGURE6_PROJECTION022_DISTAL002_PATH = (
    ROOT / "docs/validation-results/figure6-projection022-distal002-212.yaml"
)


def test_classic_calibration_contract_separates_training_and_holdout() -> None:
    contract = load_calibration_contract(CONTRACT_PATH)
    assert contract.base_tag == "classic-smart-source-constrained-v0.1.0"
    assert set(contract.training_targets).isdisjoint(contract.holdout_targets)
    assert len(contract.dimensions) == 10
    assert contract.fingerprint == load_calibration_contract(CONTRACT_PATH).fingerprint


def test_candidate_fingerprint_requires_complete_admissible_values() -> None:
    contract = load_calibration_contract(CONTRACT_PATH)
    candidate = {
        dimension.name: (
            dimension.values[0] if dimension.kind == "categorical" else dimension.grid[0]
        )
        for dimension in contract.dimensions
    }
    assert contract.candidate_fingerprint(candidate) == contract.candidate_fingerprint(candidate)
    with pytest.raises(ValueError, match="missing"):
        contract.candidate_fingerprint({})
    candidate["top_down_current_pA"] = 2000
    with pytest.raises(ValueError, match="outside declared bounds"):
        contract.candidate_fingerprint(candidate)


def test_contract_enumerates_complete_and_projected_spaces_deterministically() -> None:
    contract = load_calibration_contract(CONTRACT_PATH)
    complete = list(contract.iter_candidates())
    assert len(complete) == 9216
    assert complete == list(contract.iter_candidates())
    assert len({contract.candidate_fingerprint(item) for item in complete}) == 9216

    stage_a_dimensions = (
        "intrinsic_cell_convention",
        "calcium_kinetics_convention",
        "calcium_density_convention",
        "nak_rate_convention",
        "axial_convention",
        "membrane_initialization_convention",
        "spike_event_rule",
    )
    projected = list(contract.iter_candidates(stage_a_dimensions))
    assert len(projected) == 768
    assert {item["gaussian_spread_convention"] for item in projected} == {
        "standard_deviation"
    }
    assert {item["relay_input_interpretation"] for item in projected} == {
        "archived_finite_conductance"
    }
    assert {item["top_down_current_pA"] for item in projected} == {600.0}


def test_candidate_maps_source_coupled_calcium_conventions() -> None:
    contract = load_calibration_contract(CONTRACT_PATH)
    modeldb = next(contract.iter_candidates(()))
    modeldb_runtime = runtime_conventions_for_candidate(modeldb)
    assert modeldb_runtime.calcium_kinetics_convention == "modeldb_112923"
    assert modeldb_runtime.calcium_gate_convention == "modeldb_112923"
    assert modeldb_runtime.calcium_density_convention == "table3"
    assert modeldb_runtime.nak_rate_convention == "standard_traub_miles"

    paper = dict(modeldb, calcium_kinetics_convention="paper_2008")
    paper_runtime = runtime_conventions_for_candidate(paper)
    assert paper_runtime.calcium_kinetics_convention == "paper_2008"
    assert paper_runtime.calcium_gate_convention == "reciprocal"
    assert paper_runtime.fingerprint != modeldb_runtime.fingerprint

    printed_nak = dict(modeldb, nak_rate_convention="printed_smart")
    printed_nak_runtime = runtime_conventions_for_candidate(printed_nak)
    assert printed_nak_runtime.nak_rate_convention == "printed_smart"
    assert printed_nak_runtime.fingerprint != modeldb_runtime.fingerprint

    global_calcium = dict(modeldb, calcium_density_convention="methods_global_250")
    global_calcium_runtime = runtime_conventions_for_candidate(global_calcium)
    assert global_calcium_runtime.calcium_density_convention == "methods_global_250"
    assert global_calcium_runtime.fingerprint != modeldb_runtime.fingerprint


def test_projected_enumeration_rejects_unknown_or_duplicate_dimensions() -> None:
    contract = load_calibration_contract(CONTRACT_PATH)
    with pytest.raises(ValueError, match="unknown active dimensions"):
        list(contract.iter_candidates(("unknown",)))
    with pytest.raises(ValueError, match="must be unique"):
        list(contract.iter_candidates(("axial_convention", "axial_convention")))


def test_trn_stage_a_requires_quiescence_and_recruitment() -> None:
    base = {
        "candidate_fingerprint": "a" * 64,
        "runtime_fingerprint": "b" * 64,
        "control_finite": True,
        "driven_finite": True,
        "control_post_drive_spikes": 0,
        "driven_post_drive_spikes": 1,
        "control_soma_range_mV": (-80.0, -60.0),
        "driven_soma_range_mV": (-80.0, 35.0),
    }
    assert TrnStageAResult(**base).promoted
    assert not TrnStageAResult(**dict(base, control_post_drive_spikes=1)).promoted
    assert not TrnStageAResult(**dict(base, driven_post_drive_spikes=0)).promoted
    assert not TrnStageAResult(**dict(base, driven_finite=False)).promoted


def test_frozen_trn_survivor_matches_current_contract_and_runtime() -> None:
    contract = load_calibration_contract(CONTRACT_PATH)
    survivor = yaml.safe_load(TRN_SURVIVOR_PATH.read_text())
    candidate = survivor["candidate"]
    assert survivor["contract_fingerprint"] == contract.fingerprint
    assert survivor["candidate_fingerprint"] == contract.candidate_fingerprint(candidate)
    assert survivor["runtime_fingerprint"] == runtime_conventions_for_candidate(
        candidate
    ).fingerprint
    assert survivor["result"]["control_post_drive_spikes"] == 0
    assert survivor["result"]["driven_post_drive_spikes"] > 0


def test_network_calibration_artifact_is_bound_to_frozen_trn_survivor() -> None:
    survivor = yaml.safe_load(TRN_SURVIVOR_PATH.read_text())
    artifact = yaml.safe_load(NETWORK_CALIBRATION_PATH.read_text())
    assert artifact["candidate_fingerprint"] == survivor["candidate_fingerprint"]
    assert artifact["runtime_fingerprint"] == survivor["runtime_fingerprint"]
    assert artifact["holdouts_consulted"] is False
    assert artifact["status"] == "failed-figure6-training-target"
    assert artifact["figure6"]["feedforward_chain_complete"] is False
    assert artifact["assessment"]["figure6_reproduced"] is False
    assert artifact["assessment"]["figure7_eligible"] is False


def test_gaussian_variance_discriminator_is_a_registered_complete_candidate() -> None:
    contract = load_calibration_contract(CONTRACT_PATH)
    profile = yaml.safe_load(GAUSSIAN_VARIANCE_PROFILE_PATH.read_text())
    candidate = profile["candidate"]
    assert candidate["gaussian_spread_convention"] == "variance"
    assert profile["contract_fingerprint"] == contract.fingerprint
    assert profile["candidate_fingerprint"] == contract.candidate_fingerprint(candidate)
    assert profile["runtime_fingerprint"] == runtime_conventions_for_candidate(
        candidate
    ).fingerprint


def test_figure8_voltage_audit_rejects_release_events_as_the_observable() -> None:
    artifact = yaml.safe_load(FIGURE8_VOLTAGE_AUDIT_PATH.read_text())
    assert artifact["observable_correction"]["old"] == (
        "SMART Equation-8 axonal release events"
    )
    assert artifact["literal_250_result"]["tonic_voltage_peaks"] > artifact[
        "literal_250_result"
    ]["tonic_release_events"]
    assert artifact["calcium_unit_grid_mS_cm2"]["best_tonic_peak_times_ms"] == [
        48.13,
        112.96,
        195.31,
        288.55,
    ]
    assert artifact["assessment"]["calcium_unit_rescue"] is False
    assert artifact["assessment"]["holdouts_consulted"] is False


def test_figure6_relay_balance_causally_localizes_trn_feedback() -> None:
    survivor = yaml.safe_load(TRN_SURVIVOR_PATH.read_text())
    artifact = yaml.safe_load(FIGURE6_RELAY_BALANCE_PATH.read_text())
    assert artifact["candidate_fingerprint"] == survivor["candidate_fingerprint"]
    assert artifact["holdouts_consulted"] is False
    assert len(artifact["connected_result"]["relay_event_times_ms"]) == 1
    assert len(artifact["intrinsic_only_result"]["relay_event_times_ms"]) == 19
    trn_control = artifact["without_trn_to_relay_result"]
    assert trn_control["disabled_projection_ids"] == [
        "modeldb112923.projection.000",
        "modeldb112923.projection.001",
        "modeldb112923.projection.004",
    ]
    assert len(trn_control["relay_event_times_ms"]) == 19
    assert artifact["assessment"]["trn_inhibition_explains_repetition_failure"]


def test_all_registered_trn_survivors_fail_intact_relay_repetition() -> None:
    artifact = yaml.safe_load(FIGURE6_RELAY_SCREEN_PATH.read_text())
    assert artifact["holdouts_consulted"] is False
    assert artifact["candidate_count"] == 6
    assert artifact["relay_repetition_survivor_count"] == 0
    assert {len(item["result"]["relay_event_times_ms"]) for item in artifact["outcomes"]} == {
        1
    }
    assert {item["result"]["relay_event_times_ms"][0] for item in artifact["outcomes"]} == {
        1.8900000000000001
    }


def test_registered_trn_predrive_does_not_rescue_figure6_relay() -> None:
    artifact = yaml.safe_load(FIGURE6_RELAY_EQUILIBRATION_PATH.read_text())
    assert artifact["holdouts_consulted"] is False
    assert artifact["protocol"]["warmup_ms"] == 5.0
    assert artifact["result"]["relay_event_times_ms"] == []
    assert artifact["assessment"]["equilibration_rejected"]


def test_figure6_source_hybrid_is_reproducible_but_not_promoted() -> None:
    profile = yaml.safe_load(FIGURE6_SOURCE_HYBRID_PROFILE_PATH.read_text())
    candidate = profile["candidate"]
    fingerprint = hashlib.sha256(
        json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    artifact = yaml.safe_load(FIGURE6_SOURCE_HYBRID_RESULT_PATH.read_text())

    assert profile["candidate_fingerprint"] == fingerprint
    assert profile["runtime_fingerprint"] == runtime_conventions_for_candidate(
        candidate
    ).fingerprint
    assert artifact["candidate_fingerprint"] == fingerprint
    assert artifact["population_spikes"]["thalamic_relay"] == 5
    assert artifact["population_spikes"]["layer4_excitatory_v1"] == 5
    assert artifact["maps"]["bottom_up_oriented"]
    assert not artifact["maps"]["top_down_oriented"]
    assert not artifact["assessment"]["figure6_reproduced"]
    assert not artifact["assessment"]["promoted"]


def test_leading_figure6_hybrid_keeps_the_top_down_gate_fixed() -> None:
    profile = yaml.safe_load(FIGURE6_RELAY_AXIAL_HYBRID_PROFILE_PATH.read_text())
    candidate = profile["candidate"]
    fingerprint = hashlib.sha256(
        json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    artifact = yaml.safe_load(FIGURE6_RELAY_AXIAL_HYBRID_RESULT_PATH.read_text())

    assert profile["candidate_fingerprint"] == fingerprint
    assert profile["runtime_fingerprint"] == runtime_conventions_for_candidate(
        candidate
    ).fingerprint
    assert artifact["recruitment"]["feedforward_chain_complete"]
    assert artifact["top_down_timing"]["causal_pair_in_learning_window"]
    assert artifact["maps"]["bottom_up_oriented"]
    assert artifact["maps"]["minimum_top_down_horizontal_contrast"] == 0.01
    assert artifact["maps"]["top_down_horizontal_orientation_contrast"] < 0.01
    assert artifact["assessment"]["only_failed_gate"] == (
        "top_down_horizontal_orientation_contrast"
    )
    assert not artifact["assessment"]["promoted"]


def test_leading_figure6_map_shortfall_is_wide_field_phase_specific() -> None:
    artifact = yaml.safe_load(FIGURE6_RELAY_AXIAL_DECOMPOSITION_PATH.read_text())
    maps = artifact["maps"]
    assert maps["top_down_narrow"]["horizontal_orientation_contrast"] > maps[
        "top_down_wide"
    ]["horizontal_orientation_contrast"]
    assert maps["top_down_wide"]["horizontal_arm_delta"][1] < 0
    assert maps["top_down_wide"]["horizontal_arm_delta"][0] > 0
    assert maps["top_down_wide"]["vertical_arm_delta"] == [0.0] * 4
    assert artifact["active_cell_times_ms"]["category_40"] == [
        7.88,
        14.93,
        39.27,
        62.02,
        82.92,
    ]


def test_registered_intrinsic_alternatives_do_not_clear_figure6c() -> None:
    artifact = yaml.safe_load(FIGURE6_LEADING_ALTERNATIVES_PATH.read_text())
    assert artifact["holdouts_consulted"] is False
    assert artifact["assessment"]["base_profile_remains_leading"]
    assert all(
        artifact["assessment"][name] is False
        for name in (
            "event_rule_rescue",
            "initialization_rescue",
            "calcium_kinetics_rescue",
            "nak_rate_rescue",
            "archived_category_cell_rescue",
            "serialized_weight_initialization_rescue",
        )
    )
    assert not artifact["assessment"]["figure6_reproduced"]


def test_figure6_learning_phase_exactly_localizes_wide_field_depression() -> None:
    profile = yaml.safe_load(FIGURE6_RELAY_AXIAL_HYBRID_PROFILE_PATH.read_text())
    artifact = yaml.safe_load(FIGURE6_LEARNING_PHASE_PATH.read_text())
    assert artifact["candidate_fingerprint"] == profile["candidate_fingerprint"]
    assert artifact["runtime_fingerprint"] == profile["runtime_fingerprint"]
    assert artifact["holdouts_consulted"] is False
    assert artifact["assessment"]["integration_consistent"]
    assert artifact["assessment"]["maximum_delta_reconstruction_error"] < 1e-12

    by_projection_target = {
        (connection["projection_id"], connection["target_index"]): connection
        for connection in artifact["result"]["connections"]
    }
    wide_near = by_projection_target[("modeldb112923.projection.005", 39)]
    wide_far = by_projection_target[("modeldb112923.projection.005", 38)]
    narrow_near = by_projection_target[("modeldb112923.projection.007", 39)]
    assert wide_near["measured_delta"] < 0 < wide_far["measured_delta"]
    assert wide_near["negative_correlation_delta"] < -wide_near[
        "positive_correlation_delta"
    ]
    assert narrow_near["measured_delta"] > 0
    assert artifact["result"]["relay_event_times_ms"][31] == []


def test_projection_level_learning_bounds_are_source_literal_but_fail_figure6c() -> None:
    profile = yaml.safe_load(FIGURE6_PROJECTION_BOUNDS_PROFILE_PATH.read_text())
    candidate = profile["candidate"]
    fingerprint = hashlib.sha256(
        json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    conventions = runtime_conventions_for_candidate(candidate)
    artifact = yaml.safe_load(FIGURE6_PROJECTION_BOUNDS_RESULT_PATH.read_text())
    assert profile["candidate_fingerprint"] == fingerprint
    assert profile["runtime_fingerprint"] == conventions.fingerprint
    assert conventions.gaussian_learning_bounds_convention == "projection_level"
    assert artifact["candidate_fingerprint"] == fingerprint
    assert artifact["recruitment"]["feedforward_chain_complete"]
    assert artifact["top_down_timing"]["causal_pair_in_learning_window"]
    assert artifact["maps"]["bottom_up_oriented"]
    assert artifact["maps"]["top_down_horizontal_orientation_contrast"] < 0
    assert any(
        delta > 0
        for delta in artifact["maps"]["top_down_wide"]["vertical_arm_delta"]
    )
    assert not artifact["assessment"]["figure6_reproduced"]
    assert not artifact["assessment"]["promoted"]


def test_projection_level_depression_scale_is_incompatible_with_local_baseline() -> None:
    profile = yaml.safe_load(FIGURE6_PROJECTION_D_PROFILE_PATH.read_text())
    candidate = profile["candidate"]
    fingerprint = hashlib.sha256(
        json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    conventions = runtime_conventions_for_candidate(candidate)
    artifact = yaml.safe_load(FIGURE6_PROJECTION_D_RESULT_PATH.read_text())
    assert profile["candidate_fingerprint"] == fingerprint
    assert profile["runtime_fingerprint"] == conventions.fingerprint
    assert conventions.postsynaptic_depression_scale_convention == (
        "serialized_projection_bounds"
    )
    assert artifact["recruitment"]["feedforward_chain_complete"]
    assert artifact["top_down_timing"]["causal_pair_in_learning_window"]
    assert artifact["maps"]["bottom_up_oriented"]
    assert artifact["maps"]["top_down_horizontal_orientation_contrast"] < 0
    assert min(artifact["maps"]["top_down_narrow"]["horizontal_arm_after"]) < 0
    assert not artifact["assessment"]["figure6_reproduced"]
    assert not artifact["assessment"]["promoted"]


def test_figure6_teaching_volley_decomposition_closes_each_connection() -> None:
    artifact = yaml.safe_load(FIGURE6_VOLLEY_DECOMPOSITION_PATH.read_text())
    assert artifact["assessment"]["integration_consistent"]
    assert artifact["assessment"]["maximum_delta_reconstruction_error"] < 1e-12
    by_projection_target = {
        (connection["projection_id"], connection["target_index"]): connection
        for connection in artifact["result"]["connections"]
    }
    wide_near = by_projection_target[("modeldb112923.projection.005", 39)]
    assert sum(window["measured_delta"] for window in wide_near["windows"]) == (
        pytest.approx(wide_near["measured_delta"])
    )
    onset_window = wide_near["windows"][0]
    assert onset_window["start_ms"] == pytest.approx(9.88)
    assert onset_window["end_ms"] == pytest.approx(16.93)
    assert onset_window["positive_correlation_delta"] == 0
    assert onset_window["negative_correlation_delta"] < 0


def test_population_resolved_axial_profile_is_retracted_by_amplitude_audit() -> None:
    profile = yaml.safe_load(FIGURE6_POPULATION_AXIAL_PROFILE_PATH.read_text())
    candidate = profile["candidate"]
    fingerprint = hashlib.sha256(
        json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    conventions = runtime_conventions_for_candidate(candidate)
    historical = yaml.safe_load(FIGURE6_POPULATION_AXIAL_RESULT_PATH.read_text())
    artifact = yaml.safe_load(
        FIGURE6_POPULATION_AXIAL_AMPLITUDE_AUDIT_PATH.read_text()
    )
    assert profile["candidate_fingerprint"] == fingerprint
    assert profile["runtime_fingerprint"] == conventions.fingerprint
    assert profile["status"] == "retracted-shape-pass-amplitude-fail"
    assert artifact["candidate_fingerprint"] == fingerprint
    assert artifact["status"] == "partial-figure6b-pass-figure6c-fail"
    assert artifact["population_spikes"]["thalamic_relay"] == 20
    assert all(
        len(times) == 4
        for times in artifact["active_cell_times_ms"]["relay_horizontal"].values()
    )
    assert artifact["recruitment"]["feedforward_chain_complete"]
    assert artifact["top_down_timing"]["causal_pair_in_learning_window"]
    assert artifact["maps"]["bottom_up_oriented"]
    assert artifact["maps"]["top_down_horizontal_orientation_contrast"] >= 0.01
    assert artifact["maps"]["top_down_combined"]["maximum_after"] < 0.5
    assert not artifact["maps"]["top_down_oriented"]
    assert not artifact["assessment"]["figure6_reproduced"]
    assert not artifact["assessment"]["promoted"]
    # Artifact 136 remains the immutable record of the superseded shape-only gate.
    assert historical["status"] == "complete-figure6-pass"
    assert historical["maps"]["top_down_oriented"]


def test_shape_only_figure6_profile_fails_first_genuine_figure7_holdout() -> None:
    artifact = yaml.safe_load(FIGURE7_POPULATION_AXIAL_RESULT_PATH.read_text())
    assert artifact["holdouts_consulted"] is True
    assert artifact["status"] == "passing-figure6-failed-figure7-holdout"
    assert artifact["assessment"]["figure6_reproduced"]
    profile = yaml.safe_load(FIGURE6_POPULATION_AXIAL_PROFILE_PATH.read_text())
    assert profile["status"] == "retracted-shape-pass-amplitude-fail"
    assert artifact["figure7"]["conditions"]["match"][
        "nonspecific_spike_times_ms"
    ] == [pytest.approx(0.74)]
    assert artifact["figure7"]["conditions"]["mismatch"][
        "nonspecific_spike_times_ms"
    ] == [pytest.approx(0.74)]
    assessment = artifact["figure7"]["assessment"]
    assert assessment["arousal"]["match_rate_hz"] == 10.0
    assert assessment["arousal"]["mismatch_rate_hz"] == 10.0
    assert assessment["pathway"]["match_trn_spikes"] == 81
    assert assessment["pathway"]["mismatch_trn_spikes"] == 81
    assert not artifact["figure7"]["reproduced"]


def test_population_axial_learning_phase_exactly_closes_amplitude_deficit() -> None:
    artifact = yaml.safe_load(FIGURE6_POPULATION_AXIAL_LEARNING_PHASE_PATH.read_text())
    assert artifact["assessment"]["integration_consistent"]
    assert artifact["assessment"]["maximum_delta_reconstruction_error"] < 1e-12
    result = artifact["result"]
    assert len(result["category_event_times_ms"]) == 5
    active_relays = {38, 39, 41, 42}
    assert all(len(result["relay_event_times_ms"][index]) == 4 for index in active_relays)
    selected = {
        (connection["projection_id"], connection["target_index"]): connection
        for connection in result["connections"]
    }
    wide = selected[("modeldb112923.projection.005", 39)]
    narrow = selected[("modeldb112923.projection.007", 39)]
    assert wide["measured_delta"] == pytest.approx(0.013663, abs=1e-6)
    assert narrow["measured_delta"] == pytest.approx(0.026395, abs=1e-6)
    assert wide["postsynaptic_positive_overlap_ms"] < 0.5
    assert narrow["postsynaptic_positive_overlap_ms"] < 0.5


def test_learning_threshold_and_coordinate_candidates_are_all_rejected() -> None:
    artifact = yaml.safe_load(FIGURE6_LEARNING_THRESHOLD_ASSESSMENT_PATH.read_text())
    assert artifact["registered_gates"]["minimum_combined_peak"] == 2.0
    assert artifact["registered_gates"]["events_per_relay_in_100_ms"] == 4
    assert len(artifact["candidates"]) == 4
    assert not any(candidate["figure6_reproduced"] for candidate in artifact["candidates"])
    leak_relative = artifact["candidates"][-1]
    assert leak_relative["combined_peak"] == pytest.approx(0.527413146)
    assert leak_relative["relay_spikes"] == 58
    assert not leak_relative["relay_recruitment_confined"]
    assert artifact["assessment"]["promoted_profile"] is None


def test_methods_dual_and_candidates_fail_amplitude_or_spatial_gate() -> None:
    artifact = yaml.safe_load(FIGURE6_LEARNING_RULE_ASSESSMENT_PATH.read_text())
    assert len(artifact["candidates"]) == 6
    absolute = artifact["candidates"][-2]
    interaction = artifact["candidates"][-1]
    assert absolute["combined_peak"] == pytest.approx(0.193188217)
    assert absolute["relay_recruitment_confined"]
    assert interaction["combined_peak"] == pytest.approx(0.968761508)
    assert interaction["relay_spikes"] == 58
    assert not interaction["relay_recruitment_confined"]
    assert not artifact["assessment"]["learning_rule_explanation_sufficient"]
    assert artifact["assessment"]["promoted_profile"] is None


def test_dual_and_leak_phase_exposes_subthreshold_surround_potentiation() -> None:
    artifact = yaml.safe_load(FIGURE6_DUAL_AND_LEAK_PHASE_PATH.read_text())
    assert artifact["assessment"]["integration_consistent"]
    assert artifact["assessment"]["maximum_delta_reconstruction_error"] < 1e-12
    selected = {
        (connection["projection_id"], connection["target_index"]): connection
        for connection in artifact["result"]["connections"]
    }
    horizontal = selected[("modeldb112923.projection.005", 39)]
    vertical = selected[("modeldb112923.projection.005", 31)]
    assert horizontal["postsynaptic_positive_overlap_ms"] > 5.0
    assert vertical["postsynaptic_positive_overlap_ms"] > 2.0
    assert artifact["result"]["relay_event_times_ms"][31] == []
    assert vertical["final_weight"] > 0.2


def test_registered_nak_families_do_not_extend_relay_positive_phase() -> None:
    artifact = yaml.safe_load(FIGURE6_RELAY_WAVEFORM_PATH.read_text())
    results = {item["nak_rate_convention"]: item["result"] for item in artifact["results"]}
    standard = results["standard_traub_miles"]
    hybrid = results["archived_activation_printed_inactivation"]
    assert len(standard["relay_event_times_ms"]) == 1
    assert len(hybrid["relay_event_times_ms"]) == 1
    assert standard["soma_time_above_30_mV"] == pytest.approx(0.18)
    assert hybrid["soma_time_above_30_mV"] == pytest.approx(0.17)
    for family in ("printed_activation_archived_inactivation", "printed_smart"):
        assert results[family]["relay_event_times_ms"] == []
        assert results[family]["target_layer4_event_times_ms"] == []
        assert results[family]["soma_voltage_peak_mV"] < 0
    assert not artifact["assessment"]["nak_family_explanation_sufficient"]
    assert artifact["assessment"]["waveform_survivor"] is None


def test_upward_learning_timestamp_preserves_spikes_but_reduces_peak() -> None:
    artifact = yaml.safe_load(FIGURE6_UPWARD_TIMESTAMP_PATH.read_text())
    assert artifact["population_spikes"]["thalamic_relay"] == 20
    assert artifact["relay_recruitment"]["confined_to_horizontal_bar_at_40_hz"]
    assert artifact["maps"]["top_down_combined"]["maximum_after"] == pytest.approx(
        0.1074981615
    )
    assert not artifact["assessment"]["figure6_reproduced"]
    assert not artifact["assessment"]["promoted"]


def test_leak_relative_plus30_preserves_confinement_but_misses_peak() -> None:
    artifact = yaml.safe_load(FIGURE6_LEAK_PLUS30_PATH.read_text())
    assert artifact["population_spikes"]["thalamic_relay"] == 20
    assert artifact["relay_recruitment"]["confined_to_horizontal_bar_at_40_hz"]
    combined = artifact["maps"]["top_down_combined"]
    assert combined["maximum_after"] == pytest.approx(0.1860603038)
    assert combined["horizontal_orientation_contrast"] > 0.01
    assert not artifact["assessment"]["figure6_reproduced"]


@pytest.mark.parametrize(
    ("profile_path", "result_path", "relay_spikes", "trn_spikes"),
    (
        (
            FIGURE7_THALAMOCORTICAL_AXIAL_PROFILE_PATH,
            FIGURE6_THALAMOCORTICAL_AXIAL_RESULT_PATH,
            5,
            319,
        ),
        (
            FIGURE7_POPULATION_TRN_PROFILE_PATH,
            FIGURE6_POPULATION_TRN_RESULT_PATH,
            0,
            810,
        ),
    ),
)
def test_post_holdout_trn_source_candidates_fail_figure6_prerequisite(
    profile_path: Path,
    result_path: Path,
    relay_spikes: int,
    trn_spikes: int,
) -> None:
    profile = yaml.safe_load(profile_path.read_text())
    candidate = profile["candidate"]
    artifact = yaml.safe_load(result_path.read_text())
    assert profile["runtime_fingerprint"] == runtime_conventions_for_candidate(
        candidate
    ).fingerprint
    assert artifact["population_spikes"]["thalamic_relay"] == relay_spikes
    assert artifact["population_spikes"]["trn"] == trn_spikes
    assert not artifact["assessment"]["figure6_reproduced"]
    assert not artifact["assessment"]["promoted"]


def test_trn_potassium_source_matrix_has_no_connected_survivor() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_POTASSIUM_SCREEN_PATH.read_text())
    assert artifact["status"] == "no-connected-causal-survivor"
    assert artifact["assessment"]["connected_causal_survivors"] == 0
    assert all(item["post_bottom_up_trn_events"] == 0 for item in artifact["outcomes"])
    assert artifact["assessment"]["best_sampled_soma_peak_mV"] == pytest.approx(
        -12.1097056947
    )
    assert artifact["assessment"]["gap_to_published_arm_threshold_mV"] > 42.0


def test_trn_soma_potassium_behavior_grid_is_registered_outside_source_range() -> None:
    profile = yaml.safe_load(FIGURE7_TRN_SOMA_POTASSIUM_PROFILE_PATH.read_text())
    assert profile["status"] == "registered-before-execution"
    assert profile["dimension"]["source_status"] == (
        "behavior_calibration_outside_published_80_to_100_range"
    )
    assert profile["dimension"]["grid"] == [
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
    ]
    assert profile["fixed_choices"]["trn_spike_event_proximal_blend_fraction"] is None


def test_lower_trn_soma_potassium_has_no_isolated_recruitment_survivor() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_SOMA_POTASSIUM_STAGE1_PATH.read_text())
    assert artifact["status"] == "no-stage-1-survivor"
    assert artifact["stage_1_survivor_densities_mS_cm2"] == []
    assert artifact["assessment"]["all_controls_post_drive_quiet"]
    assert artifact["assessment"]["all_driven_trials_finite"]
    assert not artifact["assessment"]["lower_potassium_sufficient_for_recruitment"]
    assert not artifact["assessment"]["advance_to_connected_match"]
    assert artifact["assessment"]["best_driven_soma_peak_mV"] < -22.0
    assert all(not item["stage_1_pass"] for item in artifact["outcomes"])
    assert all(item["driven"]["post_drive_spike_times_ms"] == [] for item in artifact["outcomes"])


def test_source_recovery_followup_does_not_promote_compiler_log_to_source() -> None:
    artifact = yaml.safe_load(LEGACY_SOURCE_RECOVERY_FOLLOWUP_PATH.read_text())
    assert artifact["status"] == "source-body-not-recovered"
    assert artifact["new_public_trace"]["evidence_class"] == "third-party-build-log"
    assert not artifact["search_outcome"]["package_archive_recovered"]
    assert not artifact["search_outcome"]["detector_implementation_recovered"]
    assert not artifact["search_outcome"]["source_correction_authorized"]


def test_trn_soma_sodium_grid_registers_nonmonotonic_shunting_drive() -> None:
    profile = yaml.safe_load(FIGURE7_TRN_SOMA_SODIUM_PROFILE_PATH.read_text())
    assert profile["status"] == "registered-before-execution"
    assert profile["dimension"]["source_value_mS_cm2"] == 100.0
    assert profile["dimension"]["grid"] == [
        100.0,
        125.0,
        150.0,
        200.0,
        300.0,
        400.0,
    ]
    assert profile["stage_1_protocol"]["drive_multipliers"] == [
        0.05,
        0.1,
        0.2,
        0.4,
        0.7,
        1.0,
    ]


def test_increased_trn_soma_sodium_has_no_detector_cycle_survivor() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_SOMA_SODIUM_STAGE1_PATH.read_text())
    assert artifact["status"] == "no-stage-1-survivor"
    assert artifact["stage_1_survivor_densities_mS_cm2"] == []
    assessment = artifact["assessment"]
    assert assessment["all_controls_post_drive_quiet"]
    assert assessment["finite_driven_trial_count"] == 36
    assert assessment["total_driven_trial_count"] == 36
    assert assessment["best_finite_driven_soma_peak_mV"] == pytest.approx(
        14.6526694610
    )
    assert not assessment["sodium_density_sufficient_in_registered_assay"]
    assert not assessment["advance_to_connected_match"]
    assert all(not item["stage_1_pass"] for item in artifact["outcomes"])


def test_trn_detector_hysteresis_grid_separates_source_and_calibration_pairs() -> None:
    profile = yaml.safe_load(FIGURE7_TRN_DETECTOR_HYSTERESIS_PROFILE_PATH.read_text())
    assert profile["status"] == "registered-before-execution"
    assert profile["dimension"]["source_pair"] == {
        "label": "paper_30_to_0",
        "arm_mV": 30.0,
        "release_mV": 0.0,
        "status": "published",
    }
    assert profile["dimension"]["source_status"] == (
        "behavior_calibration_except_paper_30_to_0"
    )
    assert profile["stage_1_protocol"]["drive_multipliers"] == [
        0.05,
        0.1,
        0.2,
        0.4,
        0.7,
        1.0,
    ]


def test_trn_detector_hysteresis_has_four_isolated_fresh_cycle_survivors() -> None:
    artifact = yaml.safe_load(
        FIGURE7_TRN_DETECTOR_HYSTERESIS_STAGE1_PATH.read_text()
    )
    assert artifact["status"] == "stage-1-survivors-found"
    assert artifact["stage_1_survivor_labels"] == [
        "arm_n35_release_n45",
        "arm_n30_release_n40",
        "arm_n25_release_n35",
        "arm_n20_release_n30",
    ]
    assessment = artifact["assessment"]
    assert assessment["stage_1_survivor_count"] == 4
    assert not assessment["source_pair_survives"]
    assert assessment["all_controls_pass"]
    assert not assessment["startup_latched_release_counts_as_pass"]
    assert assessment["advance_to_connected_prerequisite"]

    outcomes = {item["pair"]["label"]: item for item in artifact["outcomes"]}
    assert not outcomes["paper_30_to_0"]["stage_1_pass"]
    assert all(
        not item["fresh_detector_cycle_pass"]
        for item in outcomes["paper_30_to_0"]["driven_outcomes"]
    )
    sparse = outcomes["arm_n20_release_n30"]
    passing = [
        item for item in sparse["driven_outcomes"] if item["fresh_detector_cycle_pass"]
    ]
    assert len(passing) == 1
    assert passing[0]["drive_multiplier"] == 0.4
    result = passing[0]["result"]
    assert len(result["post_stimulus_spike_times_ms"]) == 1
    assert result["threshold_upcrossings"] == 1
    assert result["arm_transitions"] == 1
    assert result["release_transitions"] == 1
    assert all(
        not outcomes[label]["stage_1_pass"]
        for label in (
            "arm_n10_release_n20",
            "arm_0_release_n10",
            "arm_10_release_0",
            "arm_20_release_10",
            "arm_30_release_20",
        )
    )


def test_trn_detector_hysteresis_has_no_figure6_survivor() -> None:
    artifact = yaml.safe_load(
        FIGURE6_TRN_DETECTOR_HYSTERESIS_PREREQUISITE_PATH.read_text()
    )
    assert artifact["status"] == "figure6-prerequisite-complete"
    assert not artifact["holdouts_consulted"]
    assert artifact["figure6_survivor_labels"] == []
    assessment = artifact["assessment"]
    assert assessment["completed_candidate_count"] == 4
    assert assessment["registered_candidate_count"] == 4
    assert assessment["figure6_survivor_count"] == 0
    assert not assessment["advance_to_same_network_match"]
    assert [
        item["population_spikes"]["trn"] for item in artifact["outcomes"]
    ] == [506, 525, 489, 511]
    for item in artifact["outcomes"]:
        assert item["population_spikes"]["thalamic_relay"] == 5
        assert item["relay_spike_indices"] == [38, 39, 40, 41, 42]
        assert set(item["relay_event_counts_by_index"].values()) == {1}
        assert item["gates"]["relay_active_indices"]
        assert item["gates"]["feedforward_chain_complete"]
        assert item["gates"]["bottom_up_horizontal_orientation"]
        assert item["gates"]["top_down_horizontal_contrast"]
        assert not item["gates"]["relay_event_count"]
        assert not item["gates"]["relay_events_per_active_index"]
        assert not item["gates"]["causal_pair_in_learning_window"]
        assert not item["figure6_pass"]


def test_trn_gaba_transfer_grid_has_one_calibrated_figure6_survivor() -> None:
    artifact = yaml.safe_load(FIGURE6_TRN_GABA_TRANSFER_GRID_PATH.read_text())
    assert artifact["status"] == "complete"
    assert not artifact["holdouts_consulted"]
    assert artifact["stage_1_survivor_scales"] == [0.001, 0.003, 0.01]
    assert artifact["stage_2_survivor_scales"] == [0.01]
    assessment = artifact["assessment"]
    assert assessment["stage_1_completed_count"] == 8
    assert assessment["registered_scale_count"] == 8
    assert assessment["stage_2_completed_count"] == 3
    assert assessment["figure6_survivor_count"] == 1
    assert assessment["advance_to_same_network_match"]
    outcomes = {item["scale"]: item for item in artifact["stage_2_outcomes"]}
    assert outcomes[0.001]["population_spikes"]["thalamic_relay"] == 35
    assert outcomes[0.003]["population_spikes"]["thalamic_relay"] == 30
    survivor = outcomes[0.01]
    assert survivor["population_spikes"]["thalamic_relay"] == 20
    assert survivor["population_spikes"]["trn"] == 728
    assert set(survivor["relay_event_counts_by_index"].values()) == {4}
    assert set(
        survivor["relay_detector_threshold_upcrossings_by_index"].values()
    ) == {4}
    assert set(survivor["relay_detector_arm_transitions_by_index"].values()) == {
        4
    }
    assert set(
        survivor["relay_detector_release_transitions_by_index"].values()
    ) == {4}
    assert all(survivor["gates"].values())
    assert survivor["pass"]


def test_calibrated_trn_transfer_passes_same_network_match_gate() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_GABA_TRANSFER_MATCH_PATH.read_text())
    assert artifact["status"] == "match-pass"
    assert artifact["holdouts_consulted"] == ["figure7_match"]
    result = artifact["result"]
    assert result["learned_state_provenance"] == "same-network-figure6-episode"
    assert len(result["relay_spike_indices"]) == 25
    assert set(result["relay_spike_indices"]) == {38, 39, 40, 41, 42}
    assert len(result["trn_spike_indices"]) == 538
    assert len(result["nonspecific_spike_times_ms"]) == 6
    assert set(artifact["relay_event_counts_by_index"].values()) == {5}
    assert artifact["sampled_trn_event_counts_by_index"] == artifact[
        "sampled_trn_threshold_upcrossings_by_index"
    ]
    assert artifact["sampled_trn_event_counts_by_index"] == artifact[
        "sampled_trn_arm_transitions_by_index"
    ]
    assert artifact["sampled_trn_event_counts_by_index"] == artifact[
        "sampled_trn_release_transitions_by_index"
    ]
    assert all(artifact["gates"].values())
    assert artifact["assessment"] == {
        "same_network_match_pass": True,
        "advance_to_mismatch": True,
    }


def test_calibrated_trn_transfer_pair_fails_official_directional_pathway() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_GABA_TRANSFER_PAIR_PATH.read_text())
    assert artifact["status"] == "figure7-failed"
    assert artifact["holdouts_consulted"] == ["figure7_match", "figure7_mismatch"]
    assert not artifact["reproduced"]
    assessment = artifact["official_assessment"]
    assert assessment["arousal"] == {
        "match_rate_hz": 60.0,
        "mismatch_rate_hz": 70.0,
    }
    assert assessment["pathway"]["match_active_relay_cells"] == 5
    assert assessment["pathway"]["mismatch_active_relay_cells"] == 5
    assert assessment["pathway"]["match_trn_spikes"] == 538
    assert assessment["pathway"]["mismatch_trn_spikes"] == 589
    assert set(assessment["pathway"]["mismatch_active_relay_indices"]) == {
        22,
        31,
        40,
        49,
        58,
    }
    assert artifact["gates"] == {
        "match_relay_spatial_pattern": True,
        "mismatch_relay_overlap_only": False,
        "match_more_active_relay_cells": False,
        "match_more_trn_events": False,
        "mismatch_more_nonspecific_events": True,
        "sampled_mismatch_trn_events_have_fresh_cycles": True,
    }


def test_each_compartment_source_endpoint_fails_figure6_repeat_gate() -> None:
    artifact = yaml.safe_load(FIGURE6_TRN_GABA_COMPARTMENT_SOURCE_PATH.read_text())
    assert artifact["status"] == "complete"
    assert not artifact["holdouts_consulted"]
    assert artifact["stage_1_survivor_labels"] == []
    assert artifact["stage_2_outcomes"] == []
    assert artifact["stage_2_survivor_labels"] == []
    assert artifact["assessment"] == {
        "registered_profile_count": 3,
        "stage_1_completed_count": 3,
        "stage_2_completed_count": 0,
        "figure6_survivor_count": 0,
        "advance_to_figure7": False,
    }
    assert [item["label"] for item in artifact["stage_1_outcomes"]] == [
        "restore_soma_source",
        "restore_proximal_source",
        "restore_distal_source",
    ]
    for item in artifact["stage_1_outcomes"]:
        assert item["population_spikes"]["thalamic_relay"] == 5
        assert item["population_spikes"]["trn"] == 392
        assert set(item["relay_event_counts_by_index"].values()) == {1}
        assert not item["pass"]


def test_distal_intermediate_grid_has_one_figure6_survivor() -> None:
    artifact = yaml.safe_load(
        FIGURE6_TRN_GABA_COMPARTMENT_INTERMEDIATE_PATH.read_text()
    )
    assert artifact["status"] == "complete"
    assert not artifact["holdouts_consulted"]
    assert artifact["stage_1_survivor_labels"] == [
        "distal_0_015",
        "distal_0_02",
    ]
    assert artifact["stage_2_survivor_labels"] == ["distal_0_015"]
    assert artifact["assessment"] == {
        "registered_profile_count": 9,
        "stage_1_completed_count": 9,
        "stage_2_completed_count": 2,
        "figure6_survivor_count": 1,
        "advance_to_figure7": True,
    }

    outcomes = {
        item["label"]: item for item in artifact["stage_2_outcomes"]
    }
    survivor = outcomes["distal_0_015"]
    assert survivor["scales"] == {
        "modeldb112923.projection.000": 0.01,
        "modeldb112923.projection.001": 0.01,
        "modeldb112923.projection.004": 0.015,
    }
    assert survivor["population_spikes"]["thalamic_relay"] == 20
    assert survivor["population_spikes"]["trn"] == 724
    assert set(survivor["relay_event_counts_by_index"].values()) == {4}
    assert all(survivor["gates"].values())
    assert survivor["pass"]

    rejected = outcomes["distal_0_02"]
    assert rejected["population_spikes"]["thalamic_relay"] == 19
    assert rejected["population_spikes"]["trn"] == 714
    assert not rejected["gates"]["relay_event_count"]
    assert not rejected["gates"]["relay_events_per_active_index"]
    assert not rejected["pass"]


def test_distal_transfer_survivor_passes_same_network_match() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_GABA_COMPARTMENT_MATCH_PATH.read_text())
    assert artifact["status"] == "match-pass"
    assert artifact["holdouts_consulted"] == ["figure7_match"]
    assert artifact["projection_weight_scales"] == {
        "modeldb112923.projection.000": 0.01,
        "modeldb112923.projection.001": 0.01,
        "modeldb112923.projection.004": 0.015,
    }
    result = artifact["result"]
    assert len(result["relay_spike_indices"]) == 22
    assert set(result["relay_spike_indices"]) == {38, 39, 40, 41, 42}
    assert len(result["trn_spike_indices"]) == 558
    assert len(result["nonspecific_spike_times_ms"]) == 9
    assert artifact["relay_event_counts_by_index"] == {
        "38": 5,
        "39": 4,
        "40": 4,
        "41": 4,
        "42": 5,
    }
    assert artifact["sampled_trn_event_counts_by_index"] == artifact[
        "sampled_trn_threshold_upcrossings_by_index"
    ]
    assert artifact["sampled_trn_event_counts_by_index"] == artifact[
        "sampled_trn_arm_transitions_by_index"
    ]
    assert artifact["sampled_trn_event_counts_by_index"] == artifact[
        "sampled_trn_release_transitions_by_index"
    ]
    assert all(artifact["gates"].values())
    assert artifact["assessment"] == {
        "same_network_match_pass": True,
        "advance_to_mismatch": True,
    }


def test_distal_transfer_pair_fails_official_figure7_directions() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_GABA_COMPARTMENT_PAIR_PATH.read_text())
    assert artifact["status"] == "figure7-failed"
    assert artifact["holdouts_consulted"] == [
        "figure7_match",
        "figure7_mismatch",
    ]
    assert not artifact["reproduced"]
    assessment = artifact["official_assessment"]
    assert assessment["arousal"] == {
        "match_rate_hz": 90.0,
        "mismatch_rate_hz": 80.0,
    }
    assert assessment["pathway"]["match_active_relay_cells"] == 5
    assert assessment["pathway"]["mismatch_active_relay_cells"] == 5
    assert assessment["pathway"]["match_trn_spikes"] == 558
    assert assessment["pathway"]["mismatch_trn_spikes"] == 566
    assert set(assessment["pathway"]["mismatch_active_relay_indices"]) == {
        22,
        31,
        40,
        49,
        58,
    }
    assert artifact["gates"] == {
        "match_relay_spatial_pattern": True,
        "mismatch_relay_overlap_only": False,
        "match_more_active_relay_cells": False,
        "match_more_trn_events": False,
        "mismatch_more_nonspecific_events": False,
        "sampled_mismatch_trn_events_have_fresh_cycles": True,
    }


def test_projection022_source_resolution_preserves_complete_figure6() -> None:
    artifact = yaml.safe_load(FIGURE6_PROJECTION022_SOURCE_RESOLUTION_PATH.read_text())
    assert artifact["status"] == "complete"
    assert not artifact["holdouts_consulted"]
    assert artifact["stage_1_survivor_labels"] == [
        "paper_supplement_projection022"
    ]
    assert artifact["stage_2_survivor_labels"] == [
        "paper_supplement_projection022"
    ]
    assert artifact["assessment"] == {
        "registered_profile_count": 1,
        "stage_1_completed_count": 1,
        "stage_2_completed_count": 1,
        "figure6_survivor_count": 1,
        "advance_to_figure7": True,
    }
    outcome = artifact["stage_2_outcomes"][0]
    assert outcome["population_spikes"]["thalamic_relay"] == 20
    assert outcome["population_spikes"]["trn"] == 541
    assert set(outcome["relay_event_counts_by_index"].values()) == {4}
    assert outcome["top_down_combined_horizontal_orientation_contrast"] == pytest.approx(
        0.48046899168012513
    )
    assert all(outcome["gates"].values())
    assert outcome["pass"]


def test_projection022_source_resolution_passes_same_network_match() -> None:
    artifact = yaml.safe_load(
        FIGURE7_PROJECTION022_SOURCE_RESOLUTION_MATCH_PATH.read_text()
    )
    assert artifact["status"] == "match-pass"
    assert artifact["holdouts_consulted"] == ["figure7_match"]
    result = artifact["result"]
    assert len(result["relay_spike_indices"]) == 30
    assert set(result["relay_spike_indices"]) == {38, 39, 40, 41, 42}
    assert len(result["trn_spike_indices"]) == 547
    assert len(result["nonspecific_spike_times_ms"]) == 5
    assert set(artifact["relay_event_counts_by_index"].values()) == {6}
    assert artifact["sampled_trn_event_counts_by_index"] == artifact[
        "sampled_trn_threshold_upcrossings_by_index"
    ]
    assert artifact["sampled_trn_event_counts_by_index"] == artifact[
        "sampled_trn_arm_transitions_by_index"
    ]
    assert artifact["sampled_trn_event_counts_by_index"] == artifact[
        "sampled_trn_release_transitions_by_index"
    ]
    assert all(artifact["gates"].values())
    assert artifact["assessment"]["advance_to_mismatch"]


def test_projection022_source_resolution_fixes_only_trn_direction() -> None:
    artifact = yaml.safe_load(
        FIGURE7_PROJECTION022_SOURCE_RESOLUTION_PAIR_PATH.read_text()
    )
    assert artifact["status"] == "figure7-failed"
    assert not artifact["reproduced"]
    assessment = artifact["official_assessment"]
    assert assessment["arousal"] == {
        "match_rate_hz": 50.0,
        "mismatch_rate_hz": 50.0,
    }
    pathway = assessment["pathway"]
    assert pathway["match_active_relay_cells"] == 5
    assert pathway["mismatch_active_relay_cells"] == 9
    assert pathway["match_trn_spikes"] == 547
    assert pathway["mismatch_trn_spikes"] == 514
    assert set(pathway["mismatch_active_relay_indices"]) == {
        22,
        31,
        38,
        39,
        40,
        41,
        42,
        49,
        58,
    }
    assert artifact["gates"] == {
        "match_relay_spatial_pattern": True,
        "mismatch_relay_overlap_only": False,
        "match_more_active_relay_cells": False,
        "match_more_trn_events": True,
        "mismatch_more_nonspecific_events": False,
        "sampled_mismatch_trn_events_have_fresh_cycles": True,
    }


def test_source_resolved_distal002_preserves_complete_figure6() -> None:
    artifact = yaml.safe_load(FIGURE6_PROJECTION022_DISTAL002_PATH.read_text())
    assert artifact["status"] == "complete"
    assert not artifact["holdouts_consulted"]
    assert artifact["stage_2_survivor_labels"] == [
        "paper_supplement_projection022_distal_0_02"
    ]
    assert artifact["assessment"]["advance_to_figure7"]
    outcome = artifact["stage_2_outcomes"][0]
    assert outcome["scales"]["modeldb112923.projection.004"] == 0.02
    assert outcome["population_spikes"]["thalamic_relay"] == 20
    assert outcome["population_spikes"]["trn"] == 544
    assert set(outcome["relay_event_counts_by_index"].values()) == {4}
    assert outcome["top_down_combined_horizontal_orientation_contrast"] == pytest.approx(
        0.5687421760725476
    )
    assert all(outcome["gates"].values())
    assert outcome["pass"]


def test_trn_calcium_reversal_restores_wrong_cue_lead_mechanism() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_CALCIUM_SCREEN_PATH.read_text())
    outcomes = {
        item["trn_calcium_source_convention"]: item for item in artifact["outcomes"]
    }
    reversal = outcomes["modeldb_reversal"]
    assert reversal["post_bottom_up_trn_events"] == 112
    assert len(reversal["connected_match"]["cue_lead_trn_spike_times_ms"]) == 81
    assert reversal["post_bottom_up_relay_events"] == 0
    assert not reversal["connected_causal_recruitment_pass"]
    assert artifact["assessment"]["connected_causal_survivors"] == 0


def test_trn_calcium_reversal_simultaneous_pair_is_condition_invariant() -> None:
    profile = yaml.safe_load(FIGURE7_TRN_CALCIUM_PROFILE_PATH.read_text())
    assert profile["runtime_fingerprint"] == runtime_conventions_for_candidate(
        profile["candidate"]
    ).fingerprint
    artifact = yaml.safe_load(FIGURE7_TRN_CALCIUM_PAIR_PATH.read_text())
    assert artifact["protocol"]["top_down_cue_lead_ms"] == 0.0
    assert not artifact["reproduced"]
    match = artifact["conditions"]["match"]
    mismatch = artifact["conditions"]["mismatch"]
    for condition in (match, mismatch):
        assert condition["relay_spike_indices"] == []
        assert len(condition["trn_spike_times_ms"]) == 229
        assert len(set(condition["trn_spike_indices"])) == 81
        assert len(condition["nonspecific_spike_times_ms"]) == 3
    assert match["trn_spike_times_ms"] == mismatch["trn_spike_times_ms"]
    assert artifact["assessment"]["arousal"]["match_rate_hz"] == 30.0
    assert artifact["assessment"]["arousal"]["mismatch_rate_hz"] == 30.0


def test_archived_trn_dendritic_calcium_density_closes_source_cube() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_DENDRITIC_CALCIUM_SCREEN_PATH.read_text())
    assert artifact["protocol"]["trn_dendritic_calcium_density_convention"] == (
        "modeldb_100"
    )
    assert artifact["assessment"]["connected_causal_survivors"] == 0
    assert all(item["post_bottom_up_trn_events"] == 0 for item in artifact["outcomes"])
    assert all(item["sampled_trn_proximal_peak_mV"] > 85.0 for item in artifact["outcomes"])
    assert max(item["sampled_trn_soma_peak_mV"] for item in artifact["outcomes"]) < -23.0


def test_behavior_density_grid_rejects_low_endpoint_and_promotes_cue_safe_values() -> None:
    profile = yaml.safe_load(FIGURE7_TRN_DENSITY_GRID_PROFILE_PATH.read_text())
    cue = yaml.safe_load(FIGURE7_TRN_DENSITY_CUE_GRID_PATH.read_text())
    assert profile["dimension"]["grid"] == [
        10.0,
        15.0,
        20.0,
        30.0,
        40.0,
        60.0,
        80.0,
        100.0,
    ]
    outcomes = {
        item["trn_dendritic_calcium_density_mS_cm2"]: item
        for item in cue["outcomes"]
    }
    assert outcomes[10.0]["cue_lead_trn_events"] == 81
    assert not outcomes[10.0]["stage_1_pass"]
    assert cue["stage_1_survivor_densities_mS_cm2"] == profile["dimension"][
        "grid"
    ][1:]


def test_behavior_density_grid_has_no_simultaneous_match_survivor() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_DENSITY_MATCH_GRID_PATH.read_text())
    assert artifact["status"] == "no-stage-2a-survivor"
    assert artifact["stage_2a_survivor_densities_mS_cm2"] == []
    for outcome in artifact["outcomes"]:
        assert outcome["active_relay_indices"] == [38, 39, 40, 41, 42]
        assert outcome["trn_events"] == 0
        assert outcome["nonspecific_events"] == 0
        assert not outcome["stage_2a_pass"]


def test_trn_source_topology_is_linear_somatic_output_and_not_a_search_dimension() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_SOURCE_TOPOLOGY_AUDIT_PATH.read_text())
    source = artifact["sources"]["smart_nml"]
    assert source["structure_class"] == "linear"
    assert source["serialized_compartment_order"] == ["Soma", "Dendrite 0", "Dendrite 1"]
    assert source["spike_monitoring"] == {
        "Soma": True,
        "Dendrite 0": False,
        "Dendrite 1": False,
    }
    assert artifact["implementation"]["compiled_edges"] == [
        ["soma", "proximal_dendrite"],
        ["proximal_dendrite", "distal_dendrite"],
    ]
    assert artifact["implementation"]["chemical_output_compartment"] == "soma"
    assert not artifact["assessment"]["star_topology_admissible"]
    assert not artifact["assessment"]["dendritic_event_output_admissible"]
    assert not artifact["assessment"]["topology_search_authorized"]


def test_local_trn_axial_grid_is_registered_and_entirely_cue_safe() -> None:
    profile = yaml.safe_load(FIGURE7_TRN_AXIAL_GRID_PROFILE_PATH.read_text())
    cue = yaml.safe_load(FIGURE7_TRN_AXIAL_CUE_GRID_PATH.read_text())
    assert profile["dimension"]["grid"] == [
        1.0,
        1.25,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        8.0,
        12.0,
        16.0,
    ]
    assert cue["stage_1_survivor_scales"] == profile["dimension"]["grid"]
    assert all(item["stage_1_pass"] for item in cue["outcomes"])
    assert all(item["cue_lead_trn_events"] == 0 for item in cue["outcomes"])
    assert all(item["cue_lead_relay_events"] == 0 for item in cue["outcomes"])


def test_local_trn_axial_grid_loses_relay_selectivity_before_recruiting_trn() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_AXIAL_MATCH_GRID_PATH.read_text())
    assert artifact["status"] == "no-stage-2a-survivor"
    assert artifact["stage_2a_survivor_scales"] == []
    for outcome in artifact["outcomes"]:
        scale = outcome["trn_soma_proximal_axial_conductance_scale"]
        expected_relay = list(range(81)) if scale >= 4.0 else [38, 39, 40, 41, 42]
        assert outcome["active_relay_indices"] == expected_relay
        assert outcome["trn_events"] == 0
        assert not outcome["stage_2a_pass"]
    assert max(
        item["sampled_trn_soma_peak_mV"] for item in artifact["outcomes"]
    ) < -23.7


def test_trn_event_offset_grid_has_a_nonmonotonic_cue_safety_boundary() -> None:
    profile = yaml.safe_load(FIGURE7_TRN_EVENT_OFFSET_PROFILE_PATH.read_text())
    cue = yaml.safe_load(FIGURE7_TRN_EVENT_OFFSET_CUE_PATH.read_text())
    assert profile["dimension"]["grid"] == [
        0.0,
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        67.0,
        69.0,
    ]
    assert cue["stage_1_survivor_offsets_mV"] == [
        0.0,
        10.0,
        20.0,
        30.0,
        40.0,
        60.0,
        67.0,
        69.0,
    ]
    outcomes = {
        item["trn_spike_event_voltage_offset_mV"]: item for item in cue["outcomes"]
    }
    for offset in (0.0, 10.0, 20.0, 30.0, 40.0):
        assert len(outcomes[offset]["result"]["equilibration_trn_spike_times_ms"]) == 81
        assert outcomes[offset]["equilibration_tail_output_events"] == 0
    assert len(outcomes[50.0]["result"]["equilibration_trn_spike_times_ms"]) == 405
    assert outcomes[50.0]["equilibration_tail_output_events"] == 162
    assert outcomes[50.0]["cue_lead_trn_events"] == 162
    for offset in (60.0, 67.0, 69.0):
        assert outcomes[offset]["result"]["equilibration_trn_spike_times_ms"] == []


def test_trn_event_offset_grid_has_no_selective_match_survivor() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_EVENT_OFFSET_MATCH_PATH.read_text())
    assert artifact["status"] == "no-stage-2a-survivor"
    assert artifact["stage_2a_survivor_offsets_mV"] == []
    for outcome in artifact["outcomes"]:
        offset = outcome["trn_spike_event_voltage_offset_mV"]
        expected_relay = list(range(81)) if offset >= 60.0 else [38, 39, 40, 41, 42]
        assert outcome["active_relay_indices"] == expected_relay
        assert outcome["trn_events"] == 0
        assert not outcome["stage_2a_pass"]


def test_trn_density_event_offset_cross_has_no_cue_safe_survivor() -> None:
    profile = yaml.safe_load(FIGURE7_TRN_DENSITY_EVENT_OFFSET_PROFILE_PATH.read_text())
    artifact = yaml.safe_load(FIGURE7_TRN_DENSITY_EVENT_OFFSET_CUE_PATH.read_text())
    assert profile["fixed_choices"]["trn_spike_event_voltage_offset_mV"] == 50.0
    assert profile["dimension"]["grid"] == [
        10.0,
        15.0,
        20.0,
        30.0,
        40.0,
        60.0,
        80.0,
        100.0,
    ]
    assert artifact["status"] == "no-stage-1-survivor"
    assert artifact["stage_1_survivor_densities_mS_cm2"] == []
    assert [item["equilibration_trn_events"] for item in artifact["outcomes"]] == [
        162,
        162,
        162,
        243,
        243,
        324,
        324,
        405,
    ]
    assert all(item["cue_lead_trn_events"] >= 81 for item in artifact["outcomes"])
    assert all(not item["stage_1_pass"] for item in artifact["outcomes"])


def test_trn_event_blend_grid_is_cue_safe_and_has_one_match_survivor() -> None:
    profile = yaml.safe_load(FIGURE7_TRN_EVENT_BLEND_PROFILE_PATH.read_text())
    cue = yaml.safe_load(FIGURE7_TRN_EVENT_BLEND_CUE_PATH.read_text())
    match = yaml.safe_load(FIGURE7_TRN_EVENT_BLEND_MATCH_PATH.read_text())
    expected_grid = [index / 10 for index in range(11)]
    assert profile["dimension"]["grid"] == expected_grid
    assert cue["stage_1_survivor_blend_fractions"] == expected_grid
    assert match["stage_2a_survivor_blend_fractions"] == [0.5]
    survivor = next(
        item
        for item in match["outcomes"]
        if item["trn_spike_event_proximal_blend_fraction"] == 0.5
    )
    assert survivor["active_relay_indices"] == [38, 39, 40, 41, 42]
    assert survivor["relay_events"] == 5
    assert survivor["trn_events"] == 81
    assert survivor["stage_2a_pass"]


def test_trn_event_blend_short_pair_fails_mismatch_gates() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_EVENT_BLEND_PAIR_PATH.read_text())
    assert artifact["status"] == "no-stage-2b-survivor"
    assert artifact["stage_2b_survivor_blend_fractions"] == []
    outcome = artifact["outcomes"][0]
    assert outcome["trn_spike_event_proximal_blend_fraction"] == 0.5
    assert outcome["gates"] == {
        "match_relay_subset": True,
        "mismatch_relay_suppressed": False,
        "trn_match_greater_than_mismatch": False,
        "nonspecific_mismatch_greater_than_match": False,
    }
    match = outcome["conditions"]["match"]
    mismatch = outcome["conditions"]["mismatch"]
    assert match["relay_spike_indices"] == [38, 39, 40, 41, 42]
    assert mismatch["relay_spike_indices"] == [22, 31, 40, 49, 58]
    assert match["trn_spike_times_ms"] == mismatch["trn_spike_times_ms"]


def test_figure7_protocol_audit_reopens_cue_trn_and_extends_final_pair() -> None:
    audit = yaml.safe_load(FIGURE7_PROTOCOL_GATE_AUDIT_PATH.read_text())
    assert audit["status"] == "validation-gate-correction-required"
    assert audit["reopened_registered_dimension"]["grid"] == [600.0, 800.0, 1000.0]
    replacements = audit["scoring_correction"]["replace"]
    assert any("TRN events are permitted" in item["new"] for item in replacements)
    assert any("300 ms" in item["new"] for item in replacements)


def test_event_blend_top_down_current_grid_is_cue_safe_and_matches_early() -> None:
    profile = yaml.safe_load(FIGURE7_EVENT_BLEND_CURRENT_PROFILE_PATH.read_text())
    cue = yaml.safe_load(FIGURE7_EVENT_BLEND_CURRENT_CUE_PATH.read_text())
    match = yaml.safe_load(FIGURE7_EVENT_BLEND_CURRENT_MATCH_PATH.read_text())
    assert profile["dimension"]["grid"] == [600.0, 800.0, 1000.0]
    assert cue["stage_1_survivor_currents_pA"] == profile["dimension"]["grid"]
    assert match["stage_2a_survivor_currents_pA"] == profile["dimension"]["grid"]
    assert [
        item["result"]["cue_lead_category_spike_times_ms"][0]
        for item in cue["outcomes"]
    ] == pytest.approx([8.89, 5.83, 4.47])
    assert all(item["cue_lead_trn_events"] == 0 for item in cue["outcomes"])
    assert all(item["cue_lead_relay_events"] == 0 for item in cue["outcomes"])
    for item in match["outcomes"]:
        assert item["active_relay_indices"] == [38, 39, 40, 41, 42]
        assert item["relay_events"] == 5
        assert item["trn_events"] == 81


def test_event_blend_top_down_current_grid_has_no_300ms_pair_survivor() -> None:
    artifact = yaml.safe_load(FIGURE7_EVENT_BLEND_CURRENT_PAIR_PATH.read_text())
    assert artifact["protocol"]["duration_ms"] == 300.0
    assert artifact["protocol"]["scoring_window"] == "whole_epoch"
    assert artifact["status"] == "no-stage-2b-survivor"
    assert artifact["stage_2b_survivor_currents_pA"] == []
    counts = []
    for item in artifact["outcomes"]:
        match = item["conditions"]["match"]
        mismatch = item["conditions"]["mismatch"]
        counts.append(
            (
                item["top_down_current_pA"],
                len(match["relay_spike_times_ms"]),
                len(mismatch["relay_spike_times_ms"]),
                len(match["trn_spike_times_ms"]),
                len(mismatch["trn_spike_times_ms"]),
            )
        )
        assert match["nonspecific_spike_times_ms"] == []
        assert mismatch["nonspecific_spike_times_ms"] == []
        assert not item["stage_2b_pass"]
    assert counts == [
        (600.0, 389, 432, 81, 81),
        (800.0, 441, 419, 81, 81),
        (1000.0, 418, 426, 81, 81),
    ]


def test_nonspecific_event_blend_grid_is_cue_safe_and_restores_output() -> None:
    profile = yaml.safe_load(FIGURE7_NONSPECIFIC_EVENT_BLEND_PROFILE_PATH.read_text())
    cue = yaml.safe_load(FIGURE7_NONSPECIFIC_EVENT_BLEND_CUE_PATH.read_text())
    mismatch = yaml.safe_load(
        FIGURE7_NONSPECIFIC_EVENT_BLEND_MISMATCH_PATH.read_text()
    )
    assert profile["dimension"]["grid"] == [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    assert cue["stage_1_survivor_blend_fractions"] == profile["dimension"]["grid"]
    assert mismatch["stage_2a_survivor_blend_fractions"] == [0.3, 0.5, 0.7, 1.0]
    assert all(item["stage_1_pass"] for item in cue["outcomes"])
    events = {
        item["nonspecific_spike_event_proximal_blend_fraction"]: len(
            item["result"]["nonspecific_spike_times_ms"]
        )
        for item in mismatch["outcomes"]
    }
    assert events == {0.0: 0, 0.1: 0, 0.2: 0, 0.3: 1, 0.5: 1, 0.7: 2, 1.0: 2}


def test_nonspecific_event_blend_has_no_match_mismatch_survivor() -> None:
    artifact = yaml.safe_load(
        FIGURE7_NONSPECIFIC_EVENT_BLEND_COMPARISON_PATH.read_text()
    )
    assert artifact["status"] == "no-stage-2b-survivor"
    assert artifact["stage_2b_survivor_blend_fractions"] == []
    expected_nonspecific_events = {0.3: 1, 0.5: 1, 0.7: 2, 1.0: 2}
    for item in artifact["outcomes"]:
        blend = item["nonspecific_spike_event_proximal_blend_fraction"]
        match = item["match"]
        mismatch = item["mismatch"]
        assert len(match["relay_spike_times_ms"]) == 20
        assert len(mismatch["relay_spike_times_ms"]) == 20
        assert len(match["trn_spike_times_ms"]) == 81
        assert len(mismatch["trn_spike_times_ms"]) == 81
        assert len(match["nonspecific_spike_times_ms"]) == expected_nonspecific_events[
            blend
        ]
        assert len(mismatch["nonspecific_spike_times_ms"]) == (
            expected_nonspecific_events[blend]
        )
        assert not item["stage_2b_pass"]


def test_feedback_arrival_audit_registers_only_archived_delay_landmarks() -> None:
    audit = yaml.safe_load(FIGURE7_FEEDBACK_ARRIVAL_AUDIT_PATH.read_text())
    assert audit["status"] == "protocol-timing-discrepancy-localized"
    assert audit["fixed_operating_point"] == {
        "top_down_current_pA": 800.0,
        "first_category_event_ms_after_current_onset": 5.83,
        "first_bottom_up_relay_event_ms_after_stimulus_onset": 4.1,
    }
    assert [
        item["first_receptor_arrival_ms_after_current_onset"]
        for item in audit["archived_feedback_delays"]
    ] == [7.83, 8.83, 9.83]
    assert audit["registered_diagnostic"]["top_down_cue_lead_ms"] == [
        0.0,
        7.83,
        8.83,
        9.83,
    ]


def test_feedback_arrival_alignment_has_no_early_pathway_survivor() -> None:
    artifact = yaml.safe_load(FIGURE7_FEEDBACK_ARRIVAL_GRID_PATH.read_text())
    assert artifact["status"] == "no-stage-2a-survivor"
    assert artifact["stage_2a_survivor_leads_ms"] == []
    assert [item["top_down_cue_lead_ms"] for item in artifact["outcomes"]] == [
        0.0,
        7.83,
        8.83,
        9.83,
    ]
    for item in artifact["outcomes"]:
        match = item["conditions"]["match"]
        mismatch = item["conditions"]["mismatch"]
        assert set(match["relay_spike_indices"]) == {38, 39, 40, 41, 42}
        assert set(mismatch["relay_spike_indices"]) == {22, 31, 40, 49, 58}
        assert len(match["trn_spike_times_ms"]) == 81
        assert len(mismatch["trn_spike_times_ms"]) == 81
        assert not item["stage_2a_pass"]


def test_local_trn_blend_arrival_interaction_has_no_mismatch_survivor() -> None:
    match = yaml.safe_load(FIGURE7_TRN_BLEND_ARRIVAL_MATCH_PATH.read_text())
    mismatch = yaml.safe_load(FIGURE7_TRN_BLEND_ARRIVAL_MISMATCH_PATH.read_text())
    assert match["stage_1_survivor_blend_fractions"] == [0.48, 0.49, 0.5]
    assert mismatch["status"] == "no-stage-2-survivor"
    assert mismatch["stage_2_survivor_blend_fractions"] == []
    for item in mismatch["outcomes"]:
        assert set(item["match"]["relay_spike_indices"]) == {38, 39, 40, 41, 42}
        assert set(item["mismatch"]["relay_spike_indices"]) == {
            22,
            31,
            40,
            49,
            58,
        }
        assert len(item["match"]["trn_spike_times_ms"]) == 81
        assert len(item["mismatch"]["trn_spike_times_ms"]) == 81
        assert not item["stage_2_pass"]


def test_inhibitory_arrival_alignment_never_generates_top_down_only_trn() -> None:
    artifact = yaml.safe_load(FIGURE7_INHIBITORY_ARRIVAL_PATH.read_text())
    assert artifact["status"] == "no-stage-2-survivor"
    assert artifact["stage_2_survivor_leads_ms"] == []
    assert [item["top_down_cue_lead_ms"] for item in artifact["outcomes"]] == [
        13.65,
        13.75,
    ]
    for item in artifact["outcomes"]:
        assert item["match"]["cue_lead_trn_spike_times_ms"] == []
        assert item["match"]["cue_lead_relay_spike_times_ms"] == []
        assert set(item["match"]["relay_spike_indices"]) == {
            31,
            38,
            39,
            40,
            41,
            42,
            49,
        }
        assert item["mismatch"] is None
        assert not item["stage_1_pass"]


def test_figure6_source_strength_reassessment_promotes_only_verifiable_claims() -> None:
    artifact = yaml.safe_load(FIGURE6_SOURCE_STRENGTH_REASSESSMENT_PATH.read_text())
    correction = artifact["source_strength_correction"]
    assessment = artifact["assessment"]
    assert artifact["status"] == "qualitative-figure6-reproduced"
    assert correction["absolute_map_amplitude"] == "not-identifiable"
    assert correction["historical_2_0_peak_gate"] == "retracted-unsupported"
    assert all(artifact["gates"].values())
    assert artifact["observed"]["combined_adaptive_final_peak"] == pytest.approx(
        0.893
    )
    assert assessment["qualitative_figure6_reproduced"] is True
    assert assessment["exact_absolute_amplitude_reproduced"] is None
    assert assessment["figure7_eligible_as_source_strength_prerequisite"] is True


def test_cold_network_trn_volley_releases_a_pre_stimulus_latch() -> None:
    artifact = yaml.safe_load(FIGURE7_TRN_DETECTOR_CYCLE_PATH.read_text())
    observed = artifact["observed"]
    assert artifact["interpretation"] == (
        "pre_stimulus_latched_arm_released_by_stimulus"
    )
    assert observed["active_relay_indices"] == [38, 39, 40, 41, 42]
    assert observed["trn_events"] == 81
    assert observed["pre_stimulus_latched_release_inferred"] is True
    assert set(observed["threshold_upcrossings_by_index"].values()) == {0}
    assert set(observed["release_transitions_by_index"].values()) == {1}
    assert set(observed["emitted_events_by_index"].values()) == {1}
    assert max(
        item[2]
        for item in observed[
            "post_first_event_detector_voltage_range_mV_by_index"
        ]
    ) < 8.0


def test_same_network_candidate_has_no_evoked_trn_and_global_relay_output() -> None:
    artifact = yaml.safe_load(FIGURE7_SAME_NETWORK_DETECTOR_PATH.read_text())
    observed = artifact["observed"]
    assert artifact["interpretation"] == "unified_candidate_has_no_evoked_trn_output"
    assert observed["relay_events"] == 181
    assert observed["active_relay_indices"] == list(range(81))
    assert observed["trn_events"] == 0
    assert observed["nonspecific_events"] == 0
    assert set(observed["threshold_upcrossings_by_index"].values()) == {0}
    assert set(observed["arm_transitions_by_index"].values()) == {0}
    assert set(observed["release_transitions_by_index"].values()) == {0}
    assert max(
        item[2] for item in observed["detector_voltage_range_mV_by_index"]
    ) < -9.0


def test_figure6_relay_train_has_four_genuine_detector_cycles_per_cell() -> None:
    artifact = yaml.safe_load(FIGURE6_RELAY_DETECTOR_CYCLE_PATH.read_text())
    control = artifact["control"]
    training = artifact["training"]
    assessment = artifact["assessment"]
    assert control["relay_events"] == 0
    assert control["active_relay_indices"] == []
    assert set(control["threshold_upcrossings_by_index"].values()) == {0}
    assert set(control["arm_transitions_by_index"].values()) == {0}
    assert set(control["release_transitions_by_index"].values()) == {0}
    assert set(control["final_armed_by_index"].values()) == {0.0}
    for key in (
        "relay_events_by_index",
        "threshold_upcrossings_by_index",
        "arm_transitions_by_index",
        "release_transitions_by_index",
    ):
        assert set(training[key].values()) == {4}
    assert assessment == {
        "control_latched_without_release": False,
        "training_cycle_valid": True,
        "interpretation": "figure6_relay_event_train_is_detector-cycle-valid",
    }


def test_contract_rejects_holdout_leakage(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONTRACT_PATH.read_text())
    raw["training_targets"].append(raw["holdout_targets"][0])
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="targets overlap"):
        load_calibration_contract(path)


def test_contract_rejects_published_free_parameter(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONTRACT_PATH.read_text())
    raw["dimensions"]["top_down_current_pA"]["status"] = "published"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="not calibration-admissible"):
        load_calibration_contract(path)
