"""Bounded source-text/archive inspection; never extract or execute native code."""

from __future__ import annotations

import io
import json
import re
import stat
import zipfile
from pathlib import PurePosixPath


def inspect_channel(raw: bytes, maximum_bytes: int) -> dict:
    if len(raw) > maximum_bytes:
        raise ValueError("channel text exceeds registered byte limit")
    text = raw.decode("utf-8")
    return {
        "declared_mechanisms": re.findall(r"\b(?:SUFFIX|POINT_PROCESS)\s+(\w+)", text),
        "dependency_constructs": sorted(
            set(re.findall(r"\b(?:USEION|POINTER|VERBATIM|INCLUDE|EXTERNAL)\b", text))
        ),
        "line_count": len(text.splitlines()),
        "equations_validated": False,
    }


def inspect_package(raw: bytes, rules: dict) -> dict:
    if len(raw) > rules["maximum_compressed_bytes_per_package"]:
        raise ValueError("native package exceeds registered compressed byte limit")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        if len(entries) > rules["maximum_archive_entries"]:
            raise ValueError("native package exceeds entry limit")
        if sum(e.file_size for e in entries) > rules["maximum_total_uncompressed_bytes"]:
            raise ValueError("native package exceeds expanded byte limit")
        names = [e.filename for e in entries]
        if len(set(names)) != len(names):
            raise ValueError("duplicate native package member paths")
        for entry in entries:
            path = PurePosixPath(entry.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in entry.filename
                or ":" in entry.filename
                or stat.S_ISLNK(entry.external_attr >> 16)
            ):
                raise ValueError("unsafe native archive member path or symlink")
        candidates = [
            e
            for e in entries
            if not e.is_dir()
            and PurePosixPath(e.filename).name == rules["expected_manifest_basename"]
        ]
        result = {
            "members": [
                {
                    "name": e.filename,
                    "bytes": e.file_size,
                    "compressed_bytes": e.compress_size,
                    "crc32": e.CRC,
                }
                for e in entries
            ],
            "manifest_candidates": [e.filename for e in candidates],
            "setup_resolved": False,
            "archive_extracted": False,
            "source_executed": False,
            "numerical_parity_validated": False,
            "setup_compatibility_validated": False,
        }
        if len(candidates) != 1:
            result["setup_error"] = "missing-or-ambiguous-manifest"
            return result
        candidate = candidates[0]
        if candidate.file_size > rules["maximum_manifest_json_bytes"]:
            raise ValueError("native manifest exceeds registered JSON byte limit")
        description = json.loads(archive.read(candidate))
        if not isinstance(description, dict):
            raise TypeError("native manifest must be a JSON object")
        result["description"] = description
        biophys = description.get("biophys", [])
        if not isinstance(biophys, list) or any(not isinstance(b, dict) for b in biophys):
            result["setup_error"] = "unrecognized-biophys-description"
            return result
        result["declared_model_types"] = [b["model_type"] for b in biophys if "model_type" in b]
        result["declared_axon_types"] = [b["axon_type"] for b in biophys if "axon_type" in b]
        result["declared_neuron_setup"] = description.get("neuron")
        result["setup_resolved"] = bool(
            result["declared_model_types"] and result["declared_axon_types"]
        )
        if not result["declared_axon_types"]:
            result["setup_error"] = "missing-authoritative-axon-type"
        return result
