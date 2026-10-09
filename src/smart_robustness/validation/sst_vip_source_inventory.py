"""Network-blind Allen cell metadata inventory; no fits or simulations."""

from __future__ import annotations

import re
from typing import Any

REQUIRED_FIELDS = (
    "specimen__id",
    "donor__species",
    "structure__acronym",
    "structure_parent__acronym",
    "structure__layer",
    "line_name",
    "cell_reporter_status",
    "tag__dendrite_type",
    "erwkf__id",
    "nrwkf__id",
    "m__biophys_all_active",
    "m__biophys_perisomatic",
    "m__glif",
)


def classify_record(row: dict[str, Any]) -> dict[str, Any]:
    """Keep exact source metadata and every exclusion, not a preferred cell."""
    missing = [key for key in REQUIRED_FIELDS if key not in row]
    if missing:
        raise ValueError(f"metadata schema missing fields: {missing}")
    specimen = row["specimen__id"]
    if type(specimen) is not int or specimen <= 0:
        raise ValueError("specimen id must be a positive integer")
    line = row["line_name"]
    classes = [
        marker
        for marker in ("Sst", "Vip")
        if isinstance(line, str) and re.search(rf"(?:^|\|){marker}(?:-|$)", line)
    ]
    reasons = []
    for passed, reason in (
        (row["donor__species"] == "Mus musculus", "outside-mouse-species"),
        (
            "VISp" in (row["structure__acronym"], row["structure_parent__acronym"]),
            "outside-primary-visual-area",
        ),
        (row["structure__layer"] in ("2/3", "5"), "outside-registered-layer"),
        (row["cell_reporter_status"] == "positive", "reporter-not-positive"),
        (row["tag__dendrite_type"] == "aspiny", "dendrite-not-aspiny"),
        (len(classes) == 1, "missing-or-ambiguous-sst-vip-driver"),
    ):
        if not passed:
            reasons.append(reason)
    counts = {}
    for label, field in (
        ("all_active", "m__biophys_all_active"),
        ("perisomatic", "m__biophys_perisomatic"),
        ("glif", "m__glif"),
    ):
        value = row[field]
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("model counts must be nonnegative integers or unknown")
        counts[label] = value
    return {
        "specimen_id": specimen,
        "included": not reasons,
        "exclusion_reasons": reasons,
        "targeting_class": classes[0] if len(classes) == 1 else None,
        "species": row["donor__species"],
        "area": row["structure__acronym"],
        "parent_area": row["structure_parent__acronym"],
        "layer": row["structure__layer"],
        "driver_line": line,
        "reporter_status": row["cell_reporter_status"],
        "dendrite_type": row["tag__dendrite_type"],
        "electrophysiology_file_id": row["erwkf__id"],
        "morphology_file_id": row["nrwkf__id"],
        "model_counts": counts,
        "model_ids": None,
        "model_id_resolution": "requires-separate-neuronal-model-query",
        "unknown_metadata_fields": [key for key in REQUIRED_FIELDS if row[key] is None],
    }


def inventory(response: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on partial responses, duplicate identities or schema drift."""
    rows = response.get("msg")
    if response.get("success") is not True or not isinstance(rows, list):
        raise ValueError("unsuccessful metadata response")
    total = response.get("total_rows")
    if type(total) is not int or total != len(rows) or not rows:
        raise ValueError("metadata response is empty or incomplete")
    records = sorted((classify_record(row) for row in rows), key=lambda x: x["specimen_id"])
    if len({r["specimen_id"] for r in records}) != len(records):
        raise ValueError("duplicate specimen ids")
    included = [r for r in records if r["included"]]
    return {
        "total_records": len(records),
        "included_records": len(included),
        "excluded_records": len(records) - len(included),
        "stratum_counts": {
            f"{marker}-L{layer}": sum(
                r["targeting_class"] == marker and r["layer"] == layer for r in included
            )
            for marker in ("Sst", "Vip")
            for layer in ("2/3", "5")
        },
        "records": records,
        "network_execution": False,
        "parameter_fitting": False,
    }
