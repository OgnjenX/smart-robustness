from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def _yaml(path: str):
    return yaml.safe_load((ROOT / path).read_text())


def test_point_hh_registration_freezes_total_conductance_aggregation() -> None:
    study = _yaml("configs/robustness/cortical_point_hh_conserved_v1.yaml")
    aggregation = study["aggregation_rule"]
    assert aggregation["capacitance"]["invariant"] == (
        "total capacitance is exactly conserved"
    )
    assert aggregation["receptor_and_external_ports"]["invariant"] == (
        "each port's total maximum conductance is conserved independently"
    )
    assert aggregation["receptor_and_external_ports"]["target_compartment"] == "soma"
    assert aggregation["gap_junction_ports"]["invariant"] == (
        "each gap junction port's executable total conductance is conserved"
    )
    assert study["execution"]["exact_reruns_per_arm"] == 2
    assert "fitting or compensating" in study["prohibited"][0]


def test_point_hh_gap_rule_was_amended_before_outcomes() -> None:
    amendment = _yaml(
        "docs/validation-results/mechanism-cortical-point-hh-gap-amendment-927a.yaml"
    )
    assert amendment["status"] == "amended-before-implementation-or-outcomes"
    assert amendment["reason"]["network_outcomes_observed"] is False
    assert "KInNeSS Equation 8" in amendment["reason"]["issue"]
    assert "conserved exactly" in amendment["amended_rule"]["invariant"]


def test_point_hh_is_registered_before_implementation_and_outcomes() -> None:
    registration = _yaml(
        "docs/validation-results/mechanism-cortical-point-hh-registration-927.yaml"
    )
    state = registration["implementation_state_at_registration"]
    assert state["aggregation_function"] == "not-implemented"
    assert state["point_population_factory"] == "not-implemented"
    assert state["conservation_checks"] == "not-run"
    assert state["isolated_checks"] == "not-run"
    assert state["connected_network_outcomes_observed"] is False
    assert registration["frozen_decisions"]["fitting_permitted"] is False


def test_point_hh_progression_stops_at_first_failed_stage() -> None:
    study = _yaml("configs/robustness/cortical_point_hh_conserved_v1.yaml")
    progression = study["progression"]
    assert "conservation" in progression["stage_1"]["endpoint"]
    assert "six-gate" in progression["stage_2"]["endpoint"]
    assert "stage 2" in progression["stage_3"]["rule"]
    assert "stage-3" in progression["stage_4"]["rule"]
    assert "first failed stage" in study["stopping_rules"][0]


def test_point_hh_prechecks_authorize_only_figure6() -> None:
    result = _yaml(
        "docs/validation-results/mechanism-cortical-point-hh-precheck-928.yaml"
    )
    assert result["execution"]["network_outcomes_observed"] is False
    assert result["structural_assessment"]["cortical_cell_classes_checked"] == 7
    assert result["structural_assessment"][
        "all_thalamic_adapter_manifests_identical"
    ] is True
    assert result["decision"]["stage_1_pass"] is True
    assert result["decision"]["figure6_authorized"] is True
    assert result["decision"]["later_network_stages_authorized"] is False


def test_point_hh_figure6_passes_exactly_before_stage3() -> None:
    result = _yaml(
        "docs/validation-results/mechanism-cortical-point-hh-figure6-assessment-929.yaml"
    )
    assert result["arms"]["classic_control"]["exact_repeat"] is True
    assert result["arms"]["classic_control"]["all_trials_pass"] is True
    assert result["arms"]["cortical_point_hh_conserved"]["exact_repeat"] is True
    assert result["arms"]["cortical_point_hh_conserved"]["all_trials_pass"] is True
    assert result["recovery_audit"][
        "canonical_trial_field_differences_excluding_repetition"
    ] == 0
    assert result["decision"]["stage3_match_mismatch_and_reset_authorized"] is True
    assert result["decision"]["spectral_and_higher_order_stages_authorized"] is False


def test_point_hh_stage3_closes_at_exact_figure7_failure() -> None:
    result = _yaml(
        "docs/validation-results/mechanism-cortical-point-hh-stage3-assessment-930.yaml"
    )
    assert result["classic_control"]["all_stage3_gates_pass"] is True
    point = result["cortical_point_hh_conserved"]
    assert point["exact_repeat"] is True
    assert point["figure7_pass"] is False
    assert point["match_mismatch_nonspecific_events"] == [5, 7]
    assert point["failed_gates"] == [
        "match_more_trn_to_nonspecific_gaba",
        "match_nonspecific_events",
    ]
    assert point["figure10_executed"] is False
    assert result["decision"]["arm_closed_at_first_failed_stage"] is True
    assert result["decision"]["spectral_and_higher_order_stages_authorized"] is False
