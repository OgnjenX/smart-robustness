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
