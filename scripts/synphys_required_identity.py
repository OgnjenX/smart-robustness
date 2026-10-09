"""Require selected primary IDs independently of release-value equality.

The coverage checks operate on projected records only. The adapter uses the
bounded selected-row reader; real execution requires a sealed registration.
"""

from __future__ import annotations

REQUIRED_TABLES = ("slice", "experiment", "cell", "pair", "synapse", "avg_response_fit")
CHILD_KEYS = {"cortical_cell_location": "cell_id", "resting_state_fit": "synapse_id",
              "conductance": "synapse_id"}


def primary_ids(values):
    ids = list(values)
    if any(type(value) is not int or value <= 0 for value in ids):
        raise ValueError("required primary IDs must be positive integers")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate required primary ID")
    return set(ids)


def required_id_coverage(tables, selection):
    expected_tables = set(REQUIRED_TABLES) | set(CHILD_KEYS)
    if set(tables) != expected_tables or set(selection) != expected_tables:
        raise ValueError("required identity projection scope differs")
    for table, expected_key in CHILD_KEYS.items():
        if selection[table][0] != expected_key:
            raise ValueError("child relationship key differs")
    coverage = {}
    for table in REQUIRED_TABLES:
        key, selected = selection[table]
        if key != "id":
            raise ValueError("required identity must use primary ID selection")
        expected = primary_ids(selected)
        if not expected:
            raise ValueError("required primary identity selection is empty")
        actual = primary_ids(row["id"] for row in tables[table])
        missing, unexpected = sorted(expected - actual), sorted(actual - expected)
        coverage[table] = {"expected_rows": len(expected), "present_rows": len(actual),
                           "missing_ids": missing, "unexpected_ids": unexpected,
                           "complete": not missing and not unexpected}
    return {"tables": coverage, "complete": all(row["complete"] for row in coverage.values())}


def paired_required_id_coverage(small_tables, medium_tables, selection):
    small = required_id_coverage(small_tables, selection)
    medium = required_id_coverage(medium_tables, selection)
    return {"small": small, "medium": medium, "complete": small["complete"] and medium["complete"]}


def reconcile_with_required_ids(small_path, medium_path, selection):
    """Add primary identity coverage to the bounded selected-value comparison.

    The CLI hash-pins this adapter. Tests supply synthetic databases only;
    real execution still needs a separately sealed source registration.
    """
    from synphys_release_reconciliation import reconcile_selected

    result = reconcile_selected(small_path, medium_path, selection)
    coverage = paired_required_id_coverage(result["small_tables"], result["medium_tables"], selection)
    result["required_primary_id_coverage"] = coverage
    result["projected_values_equal"] = result["equal"]
    result["equal"] = result["equal"] and coverage["complete"]
    return result
