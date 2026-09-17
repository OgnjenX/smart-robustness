from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def test_point_hh_exactness_correction_is_hash_locked_and_exact() -> None:
    source = (
        ROOT / "scripts/correct_cortical_point_hh_figure6_exactness.py"
    ).read_text()
    assert "REGISTERED_INPUT_SHA256" in source
    assert "yaml.safe_dump(value, sort_keys=True)" in source
    assert "first != second" in source
    assert "canonical point-HH repeats are not exactly equal" in source
    assert 'point["exact_repeat"] = True' in source
    assert '"trial_fields_changed": 0' in source
    assert "isclose" not in source
    assert "allclose" not in source


def test_point_hh_exactness_correction_registration_prohibits_reruns() -> None:
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/"
            "mechanism-cortical-point-hh-figure6-exactness-correction-929c.yaml"
        ).read_text()
    )
    assert registration["observed_state"][
        "canonical_yaml_point_hh_trials_equal_excluding_repetition"
    ] is True
    assert registration["observed_state"]["field_by_field_scientific_differences"] == 0
    assert "rerunning or discarding" in registration["prohibited"][0]
    assert "approximate numeric comparison" in registration["prohibited"]
