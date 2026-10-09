"""Complete-cohort model identity checks, independent of network performance."""

from __future__ import annotations

from typing import Any

# Official Allen GLIF template identities, not aliases for our GIF backend.
GLIF_TEMPLATE_IDS = frozenset((395310498, 395310469, 395310475, 395310479, 471355161))
BIOPHYSICAL_TEMPLATE_NAMES = {
    "Biophysical - all active": "all_active",
    "Biophysical - perisomatic": "perisomatic",
}


def validate_batch(payload: dict[str, Any], requested: list[int]) -> list[dict[str, Any]]:
    """Check exact identities and preserve complete model/resource records."""
    rows = payload.get("msg")
    if payload.get("success") is not True or not isinstance(rows, list):
        raise ValueError("unsuccessful model identity response")
    if payload.get("total_rows") != len(rows) or len(rows) != len(requested):
        raise ValueError("incomplete model identity response")
    ids = [row.get("id") for row in rows]
    if any(type(value) is not int for value in ids):
        raise ValueError("invalid specimen identity")
    if len(set(ids)) != len(ids) or set(ids) != set(requested):
        raise ValueError("requested specimen identity set changed")
    model_ids = set()
    for row in rows:
        models = row.get("neuronal_models")
        if not isinstance(models, list):
            raise TypeError("missing model list is not an empty model list")
        for model in models:
            model_id = model.get("id")
            if type(model_id) is not int or model_id <= 0 or model_id in model_ids:
                raise ValueError("invalid or duplicate model identity")
            model_ids.add(model_id)
            if model.get("specimen_id") != row["id"]:
                raise ValueError("model attached to wrong specimen")
            if "neuronal_model_template" not in model or "well_known_files" not in model:
                raise ValueError("model resource schema changed")
            files = model["well_known_files"]
            if not isinstance(files, list):
                raise TypeError("missing resource list")
    return sorted(rows, key=lambda row: row["id"])


def reconcile(rows: list[dict[str, Any]], included: list[dict[str, Any]]) -> dict[str, Any]:
    """Retain unknown templates and report drift; never choose a better model."""
    expected = {row["specimen_id"]: row for row in included}
    if len(expected) != len(included) or len(rows) != len(included):
        raise ValueError("cohort inventory changed")
    if {row["id"] for row in rows} != set(expected):
        raise ValueError("cohort identities changed")
    seen = set()
    discrepancies = []
    unknown = []
    missing_resources = []
    family_totals = dict.fromkeys(("all_active", "perisomatic", "glif"), 0)
    for row in rows:
        counts = dict.fromkeys(family_totals, 0)
        for model in row["neuronal_models"]:
            if model["id"] in seen:
                raise ValueError("duplicate model identity across batches")
            seen.add(model["id"])
            template = model["neuronal_model_template"]
            family = None
            if isinstance(template, dict):
                family = (
                    "glif"
                    if template.get("id") in GLIF_TEMPLATE_IDS
                    else BIOPHYSICAL_TEMPLATE_NAMES.get(template.get("name"))
                )
            if family is None:
                unknown.append(
                    {"specimen_id": row["id"], "model_id": model["id"], "template": template}
                )
            else:
                counts[family] += 1
                family_totals[family] += 1
            for resource in model["well_known_files"]:
                if any(
                    resource.get(key) is None
                    for key in (
                        "id",
                        "download_link",
                        "well_known_file_type",
                    )
                ):
                    missing_resources.append({"model_id": model["id"], "resource": resource})
        original = expected[row["id"]]["model_counts"]
        for family, count in counts.items():
            if original[family] is None or count != original[family]:
                discrepancies.append(
                    {
                        "specimen_id": row["id"],
                        "family": family,
                        "inventory_count": original[family],
                        "resolved_count": count,
                    }
                )
    return {
        "specimens": len(rows),
        "models": len(seen),
        "family_model_totals": family_totals,
        "no_model_specimens": [row["id"] for row in rows if not row["neuronal_models"]],
        "count_discrepancies": discrepancies,
        "unknown_templates": unknown,
        "incomplete_resource_metadata": missing_resources,
        "candidate_selection_authorized": False,
    }
