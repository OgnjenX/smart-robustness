"""Independent raw-byte/source identity audit; no native code execution."""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import yaml


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(manifest_path):
    manifest = yaml.safe_load(manifest_path.read_text())
    reg = yaml.safe_load(Path(manifest["registration"]).read_text())
    assert digest(manifest["registration"]) == manifest["registration_sha256"]
    for key in ("parent", "dependency_manifest"):
        assert digest(reg[key]) == reg[key + "_sha256"]
    source = yaml.safe_load(Path(reg["dependency_manifest"]).read_text())
    expected_files = {}
    expected_models = {}
    for model in source["models"]:
        template = model["neuronal_model_template"]
        expected_models[model["id"]] = template["name"]
        for resource in template["well_known_files"]:
            expected_files.setdefault(resource["id"], []).append(
                {"model_id": model["id"], "template_id": template["id"], "resource": resource}
            )
    expected = {("channel", i) for i in expected_files} | {("package", i) for i in expected_models}
    seen = set()
    modes = Counter()
    types = Counter()
    total_bytes = 0
    for record in manifest["records"]:
        task = record["task"]
        kind = task["kind"]
        identity = task["file_id" if kind == "channel" else "model_id"]
        key = (kind, identity)
        assert key in expected and key not in seen
        seen.add(key)
        path = manifest_path.parent / task["filename"]
        assert str(path) == record["raw_path"]
        raw = path.read_bytes()
        assert len(raw) == record["raw_bytes"] <= task["maximum_bytes"]
        assert hashlib.sha256(raw).hexdigest() == record["raw_sha256"]
        assert record["status"] == "acquired-and-statically-inspected"
        assert yaml.safe_load(Path(str(path) + ".provenance.yaml").read_text()) == record
        total_bytes += len(raw)
        inspection = record["inspection"]
        if kind == "channel":
            assert task["references"] == expected_files[identity]
            url = reg["channels"]["source_base"] + reg["channels"]["path_format"].format(
                file_id=identity
            )
            text = raw.decode("utf-8")
            assert inspection["declared_mechanisms"] == re.findall(
                r"\b(?:SUFFIX|POINT_PROCESS)\s+(\w+)", text
            )
            assert inspection["line_count"] == len(text.splitlines())
            assert inspection["equations_validated"] is False
        else:
            assert task["template"] == expected_models[identity]
            url = reg["packages"]["endpoint_format"].format(model_id=identity)
            with ZipFile(io.BytesIO(raw)) as archive:
                entries = archive.infolist()
                assert inspection["members"] == [
                    {
                        "name": e.filename,
                        "bytes": e.file_size,
                        "compressed_bytes": e.compress_size,
                        "crc32": e.CRC,
                    }
                    for e in entries
                ]
                candidates = [e for e in entries if Path(e.filename).name == "manifest.json"]
                assert len(candidates) == 1
                assert candidates[0].file_size <= reg["packages"]["maximum_manifest_json_bytes"]
                description = json.loads(archive.read(candidates[0]))
            assert inspection["description"] == description
            declared_types = [b["model_type"] for b in description["biophys"] if "model_type" in b]
            declared_modes = [b["axon_type"] for b in description["biophys"] if "axon_type" in b]
            assert declared_types == inspection["declared_model_types"] == [task["template"]]
            assert declared_modes == inspection["declared_axon_types"]
            assert inspection["setup_resolved"] == bool(declared_types and declared_modes)
            types.update(declared_types)
            modes.update(declared_modes or ["unresolved"])
        assert task["url"] == record["final_url"] == url
    assert seen == expected and len(seen) == 41
    assert manifest["failed_records"] == 0
    assert manifest["native_code_executed"] is False and manifest["archive_extracted"] is False
    return {
        "independent_verification": "passed",
        "manifest_sha256": digest(manifest_path),
        "manifest_bytes": manifest_path.stat().st_size,
        "records": len(seen),
        "total_raw_bytes": total_bytes,
        "model_types": dict(types),
        "axon_types": dict(modes),
        "native_execution_or_numerical_parity_verified": False,
    }


if __name__ == "__main__":
    print(
        yaml.safe_dump(
            verify(Path("results/sst-vip-channel-package-acquisition-1046/manifest.yaml")),
            sort_keys=False,
        )
    )
