"""One bounded metadata archive GET; inspect TAR and CSV header, never rows."""

from __future__ import annotations

import csv
import hashlib
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import yaml

REGISTRATION = Path("docs/validation-results/post2008-patchseq-annotation-archive-registration-1170.yaml")
CSV_NAME = "20200625_patchseq_metadata_mouse.csv"
CSV_MEMBERS = {CSV_NAME, "20200625_patchseq_metadata_mouse/" + CSV_NAME}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def inspect_archive(path):
    inventory, candidates, total = [], [], 0
    with tarfile.open(path, "r:") as archive:
        for count, member in enumerate(archive, start=1):
            parts = member.name.rstrip("/").split("/")
            if (count > 20 or len(member.name.encode()) > 1024 or member.name.startswith("/")
                or "\\" in member.name or any(part in ("", ".", "..") for part in parts)
                or not (member.isfile() or member.isdir()) or member.size < 0):
                raise ValueError("unregistered member count, path or type")
            total += member.size
            if total > 2097152:
                raise ValueError("declared member-byte bound exceeded")
            inventory.append({"name": member.name, "bytes": member.size,
                              "kind": "file" if member.isfile() else "directory"})
            if member.name in CSV_MEMBERS and member.isfile():
                candidates.append(member)
        if len(candidates) != 1:
            raise ValueError("exactly one declared CSV candidate required")
        with archive.extractfile(candidates[0]) as handle:
            raw = handle.readline(65537)
        if not raw or len(raw) > 65536:
            raise ValueError("bounded nonempty CSV header required")
        names = next(csv.reader([raw.decode("utf-8-sig")], strict=True))
        if not 0 < len(names) <= 256 or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("CSV columns empty, duplicated or out of bound")
    return {"members": inventory, "declared_member_bytes": total, "CSV_member": candidates[0].name,
            "header_sha256": hashlib.sha256(raw).hexdigest(), "header_bytes": len(raw), "columns": names,
            "annotation_row_values_interpreted": False, "archive_extracted_to_disk": False}


def response_identity(path, reg):
    with path.open("rb") as handle:
        raw = handle.read(65537)
    if len(raw) > 65536:
        raise ValueError("HTTP header byte bound exceeded")
    blocks = [block.splitlines() for block in raw.decode("ascii").replace("\r\n", "\n").split("\n\n")
              if block.startswith("HTTP/")]
    if not blocks or blocks[-1][0].split()[1:2] != ["200"]:
        raise ValueError("exact successful archive HTTP response required")
    fields = {}
    for line in blocks[-1][1:]:
        if ":" not in line or line[0].isspace():
            raise ValueError("malformed archive HTTP header")
        key, value = line.split(":", 1)
        fields.setdefault(key.lower().strip(), []).append(value.strip())
    for key, expected in {"etag": reg["expected_etag"], "content-length": str(reg["expected_bytes"])}.items():
        if fields.get(key) != [expected]:
            raise ValueError("registered archive response identity differs")
    if fields.get("content-encoding", ["identity"]) != ["identity"]:
        raise ValueError("archive response encoding changed")
    return {key: value[0] for key, value in fields.items() if len(value) == 1}


def validate_registration(reg):
    fixed = {"url": "https://data.nemoarchive.org/other/AIBS/AIBS_patchseq/transcriptome/scell/SMARTseq/processed/analysis/20200611/20200625_patchseq_metadata_mouse.csv.tar",
             "expected_bytes": 1075200, "expected_etag": '"106800-5a9f4c80e2eec"',
             "maximum_requests": 1, "maximum_seconds": 60, "stream_chunk_bytes": 65536,
             "certificate_authority_bundle": "/etc/ssl/cert.pem",
             "certificate_authority_sha256": "9dae8d76e55cb08991f2b672d58999ea15560d910759c16b544f843bdffbb994",
             "output_directory": "results/patchseq-annotation-archive-schema-1171",
             "parent_route_audit": "docs/validation-results/post2008-patchseq-annotation-source-route-audit-1169.yaml",
             "parent_route_audit_sha256": "2f77db317ab618de12909f799616c2502e93eaf38de62e3c5a2778573754c93f",
             "catalog_assessment": "docs/validation-results/post2008-patchseq-catalog-independent-assessment-1168.yaml",
             "catalog_assessment_sha256": "2f9371984cdd663bd69ff2849cfa57cb8196a1f5acbcec0363a4c29fb810e2b4",
             "archive_acquisition_authorized": True, "TAR_and_CSV_header_inspection_authorized": True,
             "annotation_row_interpretation_authorized": False, "NWB_or_morphology_download_authorized": False,
             "experimental_array_reads_authorized": False, "parameter_fitting_authorized": False,
             "cell_or_network_execution_authorized": False}
    if any(type(reg.get(key)) is not type(value) or reg[key] != value for key, value in fixed.items()):
        raise ValueError("fixed archive scope or identity differs")


def main():
    reg = yaml.safe_load(REGISTRATION.read_text())
    validate_registration(reg)
    for key in ("parent_route_audit", "catalog_assessment", "certificate_authority_bundle"):
        field = "certificate_authority_sha256" if key == "certificate_authority_bundle" else key + "_sha256"
        if digest(reg[key]) != reg[field]:
            raise ValueError("archive source lineage changed")
    root = Path(reg["output_directory"])
    root.mkdir()
    path, headers, stderr = root / "annotations.csv.tar", root / "response-headers.txt", root / "transport-stderr.txt"
    report = {"registration_sha256": digest(REGISTRATION), "collector_sha256": digest(__file__),
              "status": "annotation-archive-schema-failure-retained", "started_at_utc": datetime.now(UTC).isoformat(),
              "source_version": "20200625-distinct-from-final-July-annotation", "raw_path": str(path),
              "annotation_row_values_interpreted": False, "physiological_arrays_read": False}
    args = ["/usr/bin/curl", "--disable", "--silent", "--show-error", "--fail", "--proto", "=https",
            "--max-time", "60", "--max-filesize", str(reg["expected_bytes"]), "--cacert", reg["certificate_authority_bundle"],
            "--dump-header", str(headers), "--header", "If-Match: " + reg["expected_etag"],
            "--header", "Accept-Encoding: identity", reg["url"]]
    try:
        count, h = 0, hashlib.sha256()
        with path.open("xb") as output, stderr.open("xb") as error, subprocess.Popen(args, stdout=subprocess.PIPE, stderr=error) as proc:
            try:
                while count <= reg["expected_bytes"]:
                    block = proc.stdout.read(min(65536, reg["expected_bytes"] + 1 - count))
                    if not block:
                        break
                    output.write(block)
                    h.update(block)
                    count += len(block)
                if count > reg["expected_bytes"]:
                    proc.kill()
                code = proc.wait(timeout=65)
            except BaseException:
                proc.kill()
                proc.wait()
                raise
        report.update(bytes=count, sha256=h.hexdigest(), client_exit_code=code)
        if code != 0 or count != reg["expected_bytes"]:
            raise ValueError("archive transport exit or exact byte count failed")
        report["response_headers"] = response_identity(headers, reg)
        report["schema"] = inspect_archive(path)
        report["status"] = "annotation-archive-and-header-acquired-independent-verification-pending"
    except (OSError, ValueError, tarfile.TarError, csv.Error, subprocess.TimeoutExpired) as exc:
        report["error"] = str(exc)
    if path.exists():
        report.update(bytes=path.stat().st_size, sha256=digest(path))
    report["finished_at_utc"] = datetime.now(UTC).isoformat()
    with (root / "manifest.yaml").open("x") as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report["status"])
    raise SystemExit(0 if report["status"] == "annotation-archive-and-header-acquired-independent-verification-pending" else 1)


if __name__ == "__main__":
    main()
