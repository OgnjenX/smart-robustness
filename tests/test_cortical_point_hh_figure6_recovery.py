from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def test_point_hh_figure6_recovery_is_hash_locked_and_runs_only_missing_repeat() -> None:
    source = (ROOT / "scripts/recover_cortical_point_hh_figure6.py").read_text()
    assert "REGISTERED_PARTIAL_SHA256" in source
    assert 'len(classic["trials"]) != 2' in source
    assert 'len(point["trials"]) != 1' in source
    assert "repetition=1" in source
    assert 'population_factory=factory' in source
    assert 'point["trials"].append' in source


def test_point_hh_recovery_registration_preserves_known_outcomes() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/"
            "mechanism-cortical-point-hh-figure6-recovery-registration-929r.yaml"
        ).read_text()
    )
    assert registration["interruption_evidence"]["handle_status"] == "missing"
    assert registration["interruption_evidence"]["completed_trials"] == {
        "classic_control": 2,
        "cortical_point_hh_conserved": 1,
    }
    assert registration["known_outcomes_at_registration"][
        "first_point_hh_repeat_pass"
    ] is True
    assert registration["known_outcomes_at_registration"][
        "second_point_hh_repeat"
    ] == "not-run"
