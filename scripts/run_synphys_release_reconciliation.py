"""Run only a separately sealed, hash-pinned selected-summary reconciliation."""

from __future__ import annotations

import argparse
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import yaml
from synphys_release_reconciliation import (
    COLUMNS,
    build_selection,
    reconcile_selected,
    stream_sha256,
)

SOURCE_PATHS = {
    "small_database": "results/synphys-small-schema-1102/synphys_r2.1_small.sqlite",
    "medium_database": "results/synphys-medium-schema-1131/synphys_r2.1_medium.sqlite",
    "acquisition": "results/synphys-medium-schema-1131/manifest.yaml",
    "verification": "results/synphys-medium-verification-1132/manifest.yaml",
    "acquisition_assessment": "docs/validation-results/post2008-synphys-medium-schema-assessment-1133.yaml",
    "numeric_inventory": "results/synphys-numeric-targets-1111/manifest.yaml",
    "qc_inventory": "results/synphys-pair-QC-1108/manifest.yaml",
    "identity_inventory": "results/synphys-identity-inventory-1105/manifest.yaml",
}
KNOWN_HASHES = {
    "small_database": "7372499fdd874f057565080d5769baaf2659ef39d9f3bc3c7147dd1e1c280a53",
    "numeric_inventory": "ddb70c6e64092f2c97047f4f38f392ab2eec9feebb9c1b31742f41fc4a1aa674",
    "qc_inventory": "9e8af574ddfbfe259f30251833b25ae093b26237bb8befc272a358ecf4240d30",
    "identity_inventory": "c552df2ac95c8b5a248d706cb74d311906325cc06a683e6aeb0d8a6c5ec344c3",
}
DENIED = ("baseline_joins_authorized", "waveform_reads_authorized", "database_download_authorized",
          "parameter_fitting_authorized", "cell_or_network_execution_authorized")
EXPECTED_MEDIUM_BYTES = 11125997568


def load_yaml(path):
    with Path(path).open("rb") as handle:
        raw = handle.read(67108865)
    if len(raw) > 67108864:
        raise ValueError("source manifest size bound exceeded")
    return yaml.load(raw, Loader=yaml.CSafeLoader)


def validate_scope(reg):
    if (reg.get("selected_reconciliation_authorized") is not True
        or any(reg.get(key) is not False for key in DENIED)
        or reg.get("allowed_columns") != COLUMNS
        or reg.get("selected_counts") != {"average_fit_ids": 98, "synapse_ids": 92}
        or reg.get("bounds") != {"rows_per_parent": 1000, "rows_per_table": 10000,
                                  "blob_bytes": 1048576, "payload_bytes_per_release": 33554432}):
        raise ValueError("selected-only execution scope differs")
    if set(reg["sources"]) != set(SOURCE_PATHS):
        raise ValueError("source inventory scope differs")
    for key, path in SOURCE_PATHS.items():
        source = reg["sources"][key]
        checksum = source["sha256"]
        if (source["path"] != path or not isinstance(checksum, str) or len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
            or (key in KNOWN_HASHES and checksum != KNOWN_HASHES[key])):
            raise ValueError("sealed source identity differs: " + key)
    root = Path(reg["output_directory"])
    if root.parent != Path("results") or not root.name.startswith("synphys-selected-reconciliation-"):
        raise ValueError("exclusive result output scope differs")
    if (stream_sha256(Path(__file__)) != reg["runner_sha256"]
        or stream_sha256(Path(__file__).with_name("synphys_release_reconciliation.py")) != reg["helper_sha256"]
        or stream_sha256(Path(__file__).with_name("verify_synphys_medium_acquisition.py")) != reg["verifier_sha256"]):
        raise ValueError("sealed implementation changed")


def validate_acquisition(reg, acquisition, verification):
    if (acquisition.get("status") != "medium-acquired-schema-only"
        or acquisition.get("row_values_read") is not False
        or acquisition.get("client_exit_code") != 0
        or acquisition.get("bytes") != EXPECTED_MEDIUM_BYTES
        or acquisition.get("sha256") != reg["sources"]["medium_database"]["sha256"]
        or acquisition.get("raw_path") != SOURCE_PATHS["medium_database"]):
        raise ValueError("terminal acquisition gate failed")
    expected_checks = {key: "passed" for key in
                       ("byte_count", "sha256", "sqlite_signature", "schema", "source_integrity")}
    if (verification.get("status") != "medium-object-and-schema-independently-verified"
        or verification.get("collector_imported") is not False
        or verification.get("verifier_sha256") != reg["verifier_sha256"]
        or verification.get("data_rows_read") is not False
        or verification.get("checks") != expected_checks
        or verification.get("acquisition_manifest") != SOURCE_PATHS["acquisition"]
        or verification.get("acquisition_manifest_sha256") != reg["sources"]["acquisition"]["sha256"]
        or verification.get("identity") != {"bytes": EXPECTED_MEDIUM_BYTES, "sqlite_signature": True,
                                            "sha256": reg["sources"]["medium_database"]["sha256"]}
        or verification.get("schema") != acquisition["schema"]):
        raise ValueError("independent acquisition verification gate failed")


def validate_assessment(reg, assessment):
    if (assessment.get("status") != "object-acquired-schema-verified-selected-reconciliation-pending"
        or assessment.get("result") != SOURCE_PATHS["acquisition"]
        or assessment.get("result_sha256") != reg["sources"]["acquisition"]["sha256"]
        or assessment.get("verification_result") != SOURCE_PATHS["verification"]
        or assessment.get("verification_result_sha256") != reg["sources"]["verification"]["sha256"]
        or assessment.get("data_rows_read") is not False):
        raise ValueError("published acquisition assessment gate failed")


def stamp(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def run(registration):
    reg = load_yaml(registration)
    validate_scope(reg)
    root = Path(reg["output_directory"])
    # Claim the unique output before hashing an 11 GB input, so a second
    # invocation cannot concurrently repeat any projection or verification.
    root.mkdir()
    report = {"registration_sha256": stream_sha256(registration),
              "runner_sha256": stream_sha256(Path(__file__)), "helper_sha256": reg["helper_sha256"],
              "started_at_utc": datetime.now(UTC).isoformat(), "status": "engineering-failure-retained",
              "baseline_joins_performed": False, "waveform_values_read": False,
              "parameter_fitting_performed": False, "cell_or_network_execution_performed": False}
    try:
        before = {key: stamp(source["path"]) for key, source in reg["sources"].items()}
        if before["medium_database"][0] != EXPECTED_MEDIUM_BYTES:
            raise ValueError("medium object exact length changed")
        for key, source in reg["sources"].items():
            if stream_sha256(Path(source["path"])) != source["sha256"]:
                raise ValueError("input content changed: " + key)
        acquisition = load_yaml(SOURCE_PATHS["acquisition"])
        verification = load_yaml(SOURCE_PATHS["verification"])
        validate_acquisition(reg, acquisition, verification)
        validate_assessment(reg, load_yaml(SOURCE_PATHS["acquisition_assessment"]))
        targets = load_yaml(SOURCE_PATHS["numeric_inventory"])["records"]
        qc = load_yaml(SOURCE_PATHS["qc_inventory"])["tables"]
        identities = load_yaml(SOURCE_PATHS["identity_inventory"])["tables"]
        selection = build_selection(targets, qc, identities)
        if (len(selection["avg_response_fit"][1]) != 98 or len(selection["synapse"][1]) != 92):
            raise ValueError("selected source counts differ")
        report["selection"] = selection
        report["original_target_lineage"] = targets
        report.update(reconcile_selected(Path(SOURCE_PATHS["small_database"]),
                                         Path(SOURCE_PATHS["medium_database"]), selection))
        if any(stamp(source["path"]) != before[key] for key, source in reg["sources"].items()):
            raise ValueError("input changed during reconciliation")
        report["status"] = ("exact-selected-equivalence-independent-assessment-pending"
                            if report["equal"] else "selected-release-differences-retained")
    except (ValueError, OSError, KeyError, TypeError, sqlite3.Error, yaml.YAMLError) as exc:
        report["error"] = str(exc)
    report["finished_at_utc"] = datetime.now(UTC).isoformat()
    with (root / "manifest.yaml").open("x") as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    report = run(parser.parse_args().registration)
    print(report["status"])
    raise SystemExit(0 if report["status"] == "exact-selected-equivalence-independent-assessment-pending" else 1)


if __name__ == "__main__":
    main()
