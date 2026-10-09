"""Validate full-cohort physiology metadata without running source models."""

from __future__ import annotations

from typing import Any


def validate_physiology_batch(payload: dict[str, Any], requested: list[int]) -> list[dict]:
    rows = payload.get("msg")
    if payload.get("success") is not True or not isinstance(rows, list):
        raise ValueError("unsuccessful physiology response")
    if payload.get("total_rows") != len(rows) or len(rows) != len(requested):
        raise ValueError("incomplete physiology response")
    ids = [row.get("id") for row in rows]
    if (
        any(type(i) is not int for i in ids)
        or len(set(ids)) != len(ids)
        or set(ids) != set(requested)
    ):
        raise ValueError("physiology specimen identities changed")
    for row in rows:
        if "ephys_result" not in row:
            raise ValueError("ephys result schema missing")
        for relation in ("ephys_sweeps", "neuron_reconstructions"):
            if not isinstance(row.get(relation), list):
                raise TypeError(f"missing or invalid {relation}")
        if row["ephys_result"] is not None and not isinstance(row["ephys_result"], dict):
            raise TypeError("invalid ephys result")
    return sorted(rows, key=lambda row: row["id"])


def reconcile_physiology(rows: list[dict], included: list[dict]) -> dict:
    expected = {r["specimen_id"]: r for r in included}
    if len(rows) != len(expected) or {r["id"] for r in rows} != set(expected):
        raise ValueError("physiology cohort changed")
    discrepancies = []
    missing = []
    total_sweeps = 0
    counts = {"nwb_available_specimens": 0, "morphology_available_specimens": 0}
    for row in rows:
        ephys = row["ephys_result"]
        physiology_files = [] if ephys is None else ephys.get("well_known_files", [])
        morphology_files = [
            f
            for reconstruction in row["neuron_reconstructions"]
            for f in reconstruction.get("well_known_files", [])
        ]
        nwb = [
            f
            for f in physiology_files
            if f.get("well_known_file_type", {}).get("name") == "NWBDownload"
        ]
        swc = [
            f
            for f in morphology_files
            if f.get("well_known_file_type", {}).get("name") == "3DNeuronReconstruction"
        ]
        for resource, files, field in (
            ("electrophysiology", nwb, "electrophysiology_file_id"),
            ("morphology", swc, "morphology_file_id"),
        ):
            source_id = expected[row["id"]][field]
            if source_id is not None and source_id not in {f["id"] for f in files}:
                discrepancies.append(
                    {
                        "specimen_id": row["id"],
                        "resource": resource,
                        "inventory_file_id": source_id,
                        "resolved_file_ids": [f["id"] for f in files],
                    }
                )
            if not files:
                missing.append({"specimen_id": row["id"], "resource": resource})
        counts["nwb_available_specimens"] += bool(nwb)
        counts["morphology_available_specimens"] += bool(swc)
        total_sweeps += len(row["ephys_sweeps"])
    return {
        "specimens": len(rows),
        "sweep_records": total_sweeps,
        **counts,
        "identity_discrepancies": discrepancies,
        "missing_resources": missing,
        "response_guided_sweep_selection": False,
    }
