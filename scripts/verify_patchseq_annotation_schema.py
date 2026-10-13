"""Independent raw-block TAR and header verifier; no collector or TAR imports."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

REGISTRATION = Path("docs/validation-results/post2008-patchseq-annotation-archive-registration-1170.yaml")
MANIFEST = Path("results/patchseq-annotation-archive-schema-1171/manifest.yaml")
OUTPUT = Path("results/patchseq-annotation-schema-independent-1173")
NAME = "20200625_patchseq_metadata_mouse.csv"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def octal(raw):
    value = raw.strip(b"\x00 ")
    if not value:
        return 0
    if any(byte < ord("0") or byte > ord("7") for byte in value):
        raise ValueError("non-octal or extended TAR number unsupported")
    return int(value, 8)


def reconstruct(path):
    size = path.stat().st_size
    if size > 1075200 or size % 512:
        raise ValueError("bounded block-aligned archive required")
    inventory, candidates, declared = [], [], 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(512)
            if len(block) != 512:
                raise ValueError("truncated archive header or missing terminator")
            if block == b"\x00"*512:
                if handle.read(512) != b"\x00"*512:
                    raise ValueError("two zero TAR terminators required")
                while tail := handle.read(65536):
                    if any(tail):
                        raise ValueError("unregistered data after TAR terminator")
                break
            expected = octal(block[148:156])
            if sum(block[:148]) + 8*ord(" ") + sum(block[156:]) != expected:
                raise ValueError("independent TAR header checksum failed")
            name = block[:100].split(b"\x00", 1)[0].decode("utf-8")
            prefix = block[345:500].split(b"\x00", 1)[0].decode("utf-8")
            if prefix:
                name = prefix + "/" + name
            kind = block[156:157]
            length = octal(block[124:136])
            if (kind not in (b"0", b"\x00", b"5") or name.startswith("/") or "\\" in name
                or any(part in ("", ".", "..") for part in name.rstrip("/").split("/"))
                or len(name.encode()) > 1024 or len(inventory) >= 20):
                raise ValueError("unsafe or unsupported TAR member")
            declared += length
            if declared > 2097152:
                raise ValueError("declared member-byte bound exceeded")
            offset = handle.tell()
            end = offset + ((length + 511)//512)*512
            if end > size:
                raise ValueError("declared member exceeds archive")
            inventory.append({"name": name, "bytes": length, "kind": "directory" if kind == b"5" else "file"})
            if kind != b"5" and name in {NAME, "20200625_patchseq_metadata_mouse/" + NAME}:
                raw = handle.readline(min(length, 65537))
                if not raw or len(raw) > 65536:
                    raise ValueError("bounded nonempty CSV header required")
                columns = next(csv.reader([raw.decode("utf-8-sig")], strict=True))
                if not 0 < len(columns) <= 256 or any(not value for value in columns) or len(set(columns)) != len(columns):
                    raise ValueError("header column contract differs")
                candidates.append((name, raw, columns))
            handle.seek(end)  # Skip all CSV row bytes without interpreting them.
    if len(candidates) != 1:
        raise ValueError("one exact CSV member required")
    name, raw, columns = candidates[0]
    return {"members": inventory, "declared_member_bytes": declared, "CSV_member": name,
            "header_sha256": hashlib.sha256(raw).hexdigest(), "header_bytes": len(raw), "columns": columns,
            "annotation_row_values_interpreted": False, "archive_extracted_to_disk": False}


def check_headers(path):
    with path.open("rb") as handle:
        content = handle.read(65537)
    if len(content) > 65536:
        raise ValueError("HTTP header bound exceeded")
    final = None
    for group in content.decode("ascii").replace("\r\n", "\n").split("\n\n"):
        if group.startswith("HTTP/"):
            final = group.splitlines()
    if not final or final[0].split()[1:2] != ["200"]:
        raise ValueError("HTTP success identity differs")
    fields = {}
    for line in final[1:]:
        if ":" not in line or line[0].isspace():
            raise ValueError("invalid HTTP header grammar")
        key, value = line.split(":", 1)
        fields.setdefault(key.strip().lower(), []).append(value.strip())
    if (fields.get("etag") != ['"106800-5a9f4c80e2eec"'] or fields.get("content-length") != ["1075200"]
        or fields.get("content-encoding", ["identity"]) != ["identity"]):
        raise ValueError("HTTP object identity differs")
    return {key: values[0] for key, values in fields.items() if len(values) == 1}


def snapshot(path):
    value = path.stat()
    return value.st_size, value.st_mtime_ns, value.st_ino


def main():
    if not MANIFEST.is_file():
        raise ValueError("terminal archive receipt absent; do not rerun acquisition")
    OUTPUT.mkdir()
    report = {"verifier_sha256": digest(__file__), "acquisition_manifest_sha256": digest(MANIFEST),
              "collector_imported": False, "TAR_library_used": False, "network_requests": 0,
              "annotation_row_values_interpreted": False, "status": "annotation-schema-verification-failure-retained",
              "started_at_utc": datetime.now(UTC).isoformat()}
    try:
        reg = yaml.safe_load(REGISTRATION.read_text())
        result = yaml.safe_load(MANIFEST.read_text())
        path = Path("results/patchseq-annotation-archive-schema-1171/annotations.csv.tar")
        headers = path.parent / "response-headers.txt"
        collector = Path(__file__).with_name("acquire_patchseq_annotation_schema.py")
        if (digest(REGISTRATION) != "adc82b4ff911a23ad1d4cb71158e385d367958dada89059ed590a9d46f00722e"
            or result["registration_sha256"] != digest(REGISTRATION) or result["collector_sha256"] != digest(collector)
            or result["status"] != "annotation-archive-and-header-acquired-independent-verification-pending"
            or type(result["client_exit_code"]) is not int or result["client_exit_code"] != 0
            or type(result["bytes"]) is not int or result["bytes"] != 1075200 or path.stat().st_size != 1075200 or Path(result["raw_path"]) != path
            or result["annotation_row_values_interpreted"] is not False or result["physiological_arrays_read"] is not False):
            raise ValueError("terminal archive identity or scope differs")
        pinned = {REGISTRATION: digest(REGISTRATION), MANIFEST: digest(MANIFEST), collector: result["collector_sha256"],
                  path: result["sha256"], headers: digest(headers)}
        for key in ("parent_route_audit", "catalog_assessment", "certificate_authority_bundle"):
            field = "certificate_authority_sha256" if key == "certificate_authority_bundle" else key + "_sha256"
            pinned[Path(reg[key])] = reg[field]
        before = {source: snapshot(source) for source in pinned}
        if any(digest(source) != checksum for source, checksum in pinned.items()):
            raise ValueError("pinned source SHA256 differs")
        rebuilt = reconstruct(path)
        if (json.dumps(rebuilt, sort_keys=True, allow_nan=False) != json.dumps(result["schema"], sort_keys=True, allow_nan=False)
            or check_headers(headers) != result["response_headers"]):
            raise ValueError("independent schema or response-header reconstruction differs")
        if any(snapshot(source) != state for source, state in before.items()):
            raise ValueError("source changed during independent verification")
        report.update(schema=rebuilt, archive_sha256=digest(path), bytes=1075200,
                      status="annotation-archive-and-header-independently-verified")
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        report["error"] = str(exc)
    report["finished_at_utc"] = datetime.now(UTC).isoformat()
    with (OUTPUT / "manifest.yaml").open("x") as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report["status"])
    raise SystemExit(0 if report["status"] == "annotation-archive-and-header-independently-verified" else 1)


if __name__ == "__main__":
    main()
