"""Source identity and JSON-only checks; never execute model parameters."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlsplit


def acquisition_plan(manifest: dict[str, Any], rules: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for specimen in manifest["records"]:
        for model in specimen["neuronal_models"]:
            for resource in model["well_known_files"]:
                file_id = resource["id"]
                expected_path = rules["path_format"].format(file_id=file_id)
                if (
                    resource["download_link"] != expected_path
                    or resource["attachable_id"] != model["id"]
                    or resource["attachable_type"] != "NeuronalModel"
                    or resource["well_known_file_type"]["name"] != rules["required_file_type"]
                ):
                    raise ValueError("registered parameter resource identity changed")
                records.append(
                    {
                        "specimen_id": specimen["id"],
                        "model_id": model["id"],
                        "template": model["neuronal_model_template"],
                        "file_id": file_id,
                        "resource": resource,
                        "request_url": rules["source_base"] + expected_path,
                    }
                )
    records.sort(key=lambda row: (row["model_id"], row["file_id"]))
    if (
        len(records) != rules["expected_unique_file_count"]
        or len({r["file_id"] for r in records}) != len(records)
        or len({r["model_id"] for r in records}) != rules["expected_model_count"]
    ):
        raise ValueError("parameter resource inventory incomplete or duplicated")
    return records


def validate_url(url: str, base: str) -> None:
    parsed, allowed = urlsplit(url), urlsplit(base)
    if (
        parsed.scheme != "https"
        or parsed.hostname != allowed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
    ):
        raise ValueError("parameter request or redirect outside registered HTTPS source")


def validate_json(raw: bytes, maximum_bytes: int) -> list[str]:
    if len(raw) > maximum_bytes:
        raise ValueError("parameter file exceeds registered byte limit")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise TypeError("parameter JSON must be a top-level object")
    return sorted(parsed)
