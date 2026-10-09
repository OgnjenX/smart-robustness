"""Source-only inclusion and fail-closed inventory contracts."""

from __future__ import annotations

import pytest

from smart_robustness.validation.sst_vip_source_inventory import classify_record, inventory


def row(**overrides):
    record = {
        "specimen__id": 1,
        "donor__species": "Mus musculus",
        "structure__acronym": "VISp5",
        "structure_parent__acronym": "VISp",
        "structure__layer": "5",
        "line_name": "Sst-IRES-Cre",
        "cell_reporter_status": "positive",
        "tag__dendrite_type": "aspiny",
        "erwkf__id": 12,
        "nrwkf__id": None,
        "m__biophys_all_active": 0,
        "m__biophys_perisomatic": 0,
        "m__glif": None,
    }
    return record | overrides


def test_inclusion_does_not_require_models_or_morphology():
    result = classify_record(row())
    assert result["included"]
    assert result["model_counts"]["glif"] is None
    assert result["model_ids"] is None
    assert "nrwkf__id" in result["unknown_metadata_fields"]


@pytest.mark.parametrize(
    "line,expected",
    [
        ("Vip-IRES-Cre", "Vip"),
        ("Nos1-CreERT2|Sst-IRES-FlpO", "Sst"),
        ("Htr3a-Cre", None),
        ("OtherSst-Cre", None),
        ("Sst-Cre|Vip-Cre", None),
    ],
)
def test_marker_identity_is_explicit(line, expected):
    result = classify_record(row(line_name=line))
    assert result["targeting_class"] == expected
    assert result["included"] == (expected is not None)


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"donor__species": "Homo Sapiens"}, "outside-mouse-species"),
        (
            {"structure__acronym": "VISl", "structure_parent__acronym": "VISl"},
            "outside-primary-visual-area",
        ),
        ({"structure__layer": "4"}, "outside-registered-layer"),
        ({"cell_reporter_status": None}, "reporter-not-positive"),
        ({"tag__dendrite_type": "spiny"}, "dendrite-not-aspiny"),
    ],
)
def test_exclusions_are_retained(changes, reason):
    assert reason in classify_record(row(**changes))["exclusion_reasons"]


def test_partial_failed_duplicate_and_empty_responses_reject():
    for payload in (
        {"success": False, "total_rows": 1, "msg": [row()]},
        {"success": True, "total_rows": 2, "msg": [row()]},
        {"success": True, "total_rows": 2, "msg": [row(), row()]},
        {"success": True, "total_rows": 0, "msg": []},
    ):
        with pytest.raises(ValueError):
            inventory(payload)


def test_schema_drift_and_invalid_counts_reject():
    record = row()
    del record["line_name"]
    with pytest.raises(ValueError, match="schema"):
        classify_record(record)
    with pytest.raises(ValueError, match="model counts"):
        classify_record(row(m__glif=-1))


def test_all_cells_sorted_and_strata_separate():
    result = inventory(
        {
            "success": True,
            "total_rows": 3,
            "msg": [
                row(specimen__id=3, line_name="Vip-IRES-Cre", structure__layer="2/3"),
                row(specimen__id=1),
                row(specimen__id=2, structure__layer="4"),
            ],
        }
    )
    assert [r["specimen_id"] for r in result["records"]] == [1, 2, 3]
    assert result["included_records"] == 2
    assert result["excluded_records"] == 1
    assert result["stratum_counts"] == {"Sst-L2/3": 0, "Sst-L5": 1, "Vip-L2/3": 1, "Vip-L5": 0}
    assert not result["network_execution"] and not result["parameter_fitting"]
