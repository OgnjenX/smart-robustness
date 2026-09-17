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
    assert study["execution"]["exact_reruns_per_arm"] == 2
    assert "fitting or compensating" in study["prohibited"][0]


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
