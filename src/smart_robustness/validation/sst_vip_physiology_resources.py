"""Full-cohort resource identity selection, independent of physiological outcomes."""

from __future__ import annotations


def resource_plan(records: list[dict]) -> dict:
    specimens = set()
    resources = {}
    missing_morphology = []
    for specimen in sorted(records, key=lambda s: s["id"]):
        sid = specimen["id"]
        if sid in specimens:
            raise ValueError("duplicate specimen")
        specimens.add(sid)
        ephys = specimen.get("ephys_result") or {}
        nwb = [
            f
            for f in ephys.get("well_known_files", [])
            if f["well_known_file_type"]["name"] == "NWBDownload"
        ]
        if not nwb:
            raise ValueError("registered specimen has no NWBDownload")
        morphology = [
            (r["id"], f)
            for r in specimen.get("neuron_reconstructions", [])
            for f in r.get("well_known_files", [])
            if f["well_known_file_type"]["name"] == "3DNeuronReconstruction"
        ]
        if not morphology:
            missing_morphology.append(sid)
        refs = [(ephys["id"], "EphysResult", f) for f in nwb] + [
            (rid, "NeuronReconstruction", f) for rid, f in morphology
        ]
        for attachment, attachment_type, file in refs:
            fid = file["id"]
            if (
                file["attachable_id"] != attachment
                or file["attachable_type"] != attachment_type
                or file["download_link"] != f"/api/v2/well_known_file_download/{fid}"
            ):
                raise ValueError("resource attachment or download identity mismatch")
            if fid not in resources:
                resources[fid] = {"file_id": fid, "resource": file, "specimen_ids": []}
            elif resources[fid]["resource"] != file:
                raise ValueError("conflicting resource identity")
            resources[fid]["specimen_ids"].append(sid)
    return {
        "specimen_ids": sorted(specimens),
        "resources": [resources[k] for k in sorted(resources)],
        "missing_morphology_specimen_ids": missing_morphology,
    }


def content_length(value: str | None) -> int:
    if value is None or not value.isascii() or not value.isdecimal() or int(value) <= 0:
        raise ValueError("missing or invalid positive content length")
    return int(value)
