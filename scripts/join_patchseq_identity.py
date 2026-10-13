"""Read whitelisted metadata, preserve ambiguities; no physiological promotion."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import tarfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

REGISTRATION = Path("docs/validation-results/post2008-patchseq-identity-join-registration-1175.yaml")
FIELDS = ["project", "cell_specimen_id", "donor_id", "ephys_session_id", "transcriptomics_sample_id",
          "structure", "full_genotype", "corresponding_AIT2.3.1_ID", "corresponding_AIT2.3.1_alias"]
IDS = ("cell_specimen_id", "donor_id", "ephys_session_id")
PATH = re.compile(r"sub-([1-9][0-9]*)/sub-\1_ses-([1-9][0-9]*)_icephys\.nwb")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def canonical_id(value):
    return int(value) if type(value) is str and re.fullmatch(r"[1-9][0-9]*", value) else None


def read_rows(path, schema):
    if schema["annotation_row_values_interpreted"] is not False:
        raise ValueError("header-only verified source required")
    previous_limit = csv.field_size_limit(4096)
    try:
        with tarfile.open(path, "r:") as archive:
            matches = [member for member in archive if member.name == schema["CSV_member"]]
            if len(matches) != 1 or not matches[0].isfile() or matches[0].size > 2097152:
                raise ValueError("exact bounded CSV member required")
            with io.TextIOWrapper(archive.extractfile(matches[0]), encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle, strict=True)
                header = next(reader)
                if header != schema["columns"] or any(field not in header for field in FIELDS):
                    raise ValueError("verified header or whitelist differs")
                positions = {field: header.index(field) for field in FIELDS}
                rows = []
                for number, values in enumerate(reader, start=2):
                    if len(rows) >= 10000 or len(values) != len(header):
                        raise ValueError("annotation row extent or shape differs")
                    raw = {field: values[index] for field, index in positions.items()}
                    if any(len(value) > 4096 for value in raw.values()):
                        raise ValueError("raw annotation field limit exceeded")
                    rows.append({"source_row_number": number, "raw_fields": raw,
                                 "canonical_ids": {field: canonical_id(raw[field]) for field in IDS}})
    finally:
        csv.field_size_limit(previous_limit)
    return rows


def join_rows(assets, rows, old_records):
    if (len(assets) != 4435 or len(rows) > 10000 or len({asset["asset_id"] for asset in assets}) != 4435
        or len({asset["path"] for asset in assets}) != 4435
        or any(type(row["source_row_number"]) is not int or row["source_row_number"] < 2 for row in rows)
        or len({row["source_row_number"] for row in rows}) != len(rows)):
        raise ValueError("bounded complete catalog and annotations required")
    sessions, cells = Counter(), Counter()
    by_session = defaultdict(list)
    for row in rows:
        if set(row["raw_fields"]) != set(FIELDS) or set(row["canonical_ids"]) != set(IDS):
            raise ValueError("projected annotation scope differs")
        expected = {field: canonical_id(row["raw_fields"][field]) for field in IDS}
        if json.dumps(row["canonical_ids"], sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError("canonical ID projection differs")
        cell, session = expected["cell_specimen_id"], expected["ephys_session_id"]
        if cell is not None:
            cells[cell] += 1
        if session is not None:
            sessions[session] += 1
            by_session[session].append(row)
    catalog_sessions = Counter(int(match.group(2)) for asset in assets if (match := PATH.fullmatch(asset["path"])))
    outputs, referenced = [], set()
    for asset in assets:
        match = PATH.fullmatch(asset["path"])
        subject, session = (int(match.group(1)), int(match.group(2))) if match else (None, None)
        parents = by_session.get(session, [])
        referenced.update(row["source_row_number"] for row in parents)
        parent = parents[0] if len(parents) == 1 else None
        ids = parent["canonical_ids"] if parent else dict.fromkeys(IDS)
        flags = {"canonical_path_shape": match is not None, "one_annotation_candidate": len(parents) == 1,
                 "canonical_specimen_ID": ids["cell_specimen_id"] is not None,
                 "canonical_donor_ID": ids["donor_id"] is not None,
                 "unique_source_specimen": ids["cell_specimen_id"] is not None and cells[ids["cell_specimen_id"]] == 1,
                 "unique_source_session": session is not None and sessions[session] == 1,
                 "unique_catalog_session": session is not None and catalog_sessions[session] == 1,
                 "donor_matches_path_subject": subject is not None and ids["donor_id"] == subject}
        outputs.append({"asset_id": asset["asset_id"], "path": asset["path"], "path_subject_token": subject,
                        "path_session_token": session, "annotation_candidates": parents, "identity_flags": flags,
                        "path_annotation_consistent": all(flags.values()), "internal_recording_identity_confirmed": False,
                        "molecular_class_qualified": False, "measured_layer_qualified": False,
                        "biological_eligibility_established": False})
    old = {record["specimen_id"] for record in old_records}
    included = {record["specimen_id"] for record in old_records if record["included"] is True}
    if len(old) != 2333 or len(included) != 111:
        raise ValueError("prior source identity inventory differs")
    return {"assets": outputs, "annotation_rows": rows,
            "unreferenced_source_row_numbers": [row["source_row_number"] for row in rows if row["source_row_number"] not in referenced],
            "duplicate_source_specimen_IDs": sorted(value for value, count in cells.items() if count > 1),
            "duplicate_source_session_IDs": sorted(value for value, count in sessions.items() if count > 1),
            "duplicate_catalog_session_tokens": sorted(value for value, count in catalog_sessions.items() if count > 1),
            "prior_specimen_ID_equalities": sorted(set(cells) & old), "prior_included_specimen_ID_equalities": sorted(set(cells) & included),
            "summary": {"assets": len(outputs), "annotation_rows": len(rows),
                        "path_annotation_consistent_assets": sum(row["path_annotation_consistent"] for row in outputs),
                        "assets_without_annotation_candidates": sum(not row["annotation_candidates"] for row in outputs),
                        "assets_with_multiple_annotation_candidates": sum(len(row["annotation_candidates"]) > 1 for row in outputs),
                        "biological_eligibility_established": False}}


def main():
    if digest(REGISTRATION) != "4eed2c338f08dedc397b9536c923b2b20ac428bc89bafa962779651290479caf":
        raise ValueError("sealed identity registration changed")
    reg = yaml.safe_load(REGISTRATION.read_text())
    if (reg["allowed_fields"] != FIELDS or reg["maximum_rows"] != 10000 or reg["maximum_field_characters"] != 4096
        or reg["maximum_CSV_bytes"] != 2097152 or reg["whitelisted_annotation_row_reads_authorized"] is not True
        or any(reg[key] is not False for key in ("external_requests_authorized", "recording_download_authorized",
               "physiological_array_reads_authorized", "cell_class_or_layer_qualification_authorized",
               "parameter_fitting_authorized", "cell_or_network_execution_authorized"))
        or reg["output_directory"] != "results/patchseq-identity-annotations-1176"):
        raise ValueError("registered row interpretation scope differs")
    sources = {Path(reg[key]): reg[key + "_sha256"] for key in
               ("schema_assessment", "annotation_archive", "schema_verification", "catalog_verification", "previous_specimen_inventory")}
    if any(digest(path) != checksum for path, checksum in sources.items()):
        raise ValueError("pinned annotation or catalog source differs")
    schema = yaml.safe_load(Path(reg["schema_verification"]).read_text())
    catalog = yaml.safe_load(Path(reg["catalog_verification"]).read_text())
    if (schema["status"] != "annotation-archive-and-header-independently-verified"
        or catalog["status"] != "patchseq-catalog-independently-verified"
        or schema["annotation_row_values_interpreted"] is not False or catalog["physiological_arrays_read"] is not False):
        raise ValueError("independent source gates not passed")
    root = Path(reg["output_directory"])
    root.mkdir()
    report = {"registration_sha256": digest(REGISTRATION), "collector_sha256": digest(__file__),
              "status": "identity-annotation-failure-retained", "started_at_utc": datetime.now(UTC).isoformat(),
              "source_version": "20200625-June-annotation", "physiological_arrays_read": False,
              "parameter_fitting_performed": False}
    try:
        rows = read_rows(Path(reg["annotation_archive"]), schema["schema"])
        old = yaml.safe_load(Path(reg["previous_specimen_inventory"]).read_text())["records"]
        report.update(join_rows(catalog["assets"], rows, old))
        if any(digest(path) != checksum for path, checksum in sources.items()):
            raise ValueError("source evidence changed during annotation join")
        report["status"] = "identity-annotations-joined-independent-verification-pending"
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError, csv.Error) as exc:
        report["error"] = str(exc)
    report["finished_at_utc"] = datetime.now(UTC).isoformat()
    with (root / "manifest.yaml").open("x") as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report["status"])
    raise SystemExit(0 if report["status"] == "identity-annotations-joined-independent-verification-pending" else 1)


if __name__ == "__main__":
    main()
