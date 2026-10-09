"""Freeze official public release metadata; never follow database links."""

from __future__ import annotations

import ast
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import yaml
from collect_synphys_source_metadata import curl_receipt, digest, save, tls_context

REGISTRATION = Path(
    "docs/validation-results/post2008-synphys-release-catalog-registration-1095.yaml"
)


def revision(raw):
    value = json.loads(raw)["sha"]
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("invalid immutable commit revision")
    return value


def inspect_catalog(raw):
    data = json.loads(raw)
    if not isinstance(data.get("default_url_path"), str) or not isinstance(
        data.get("databases"), list
    ):
        raise TypeError("unknown catalog schema")
    entries, seen = [], set()
    for item in data["databases"]:
        name = item["file"]
        if not isinstance(name, str) or not name.endswith(".sqlite") or "/" in name or name in seen:
            raise ValueError("invalid or duplicate declared database name")
        aliases = item.get("aliases", [])
        if not isinstance(aliases, list) or any(not isinstance(a, str) for a in aliases):
            raise ValueError("invalid catalog aliases")
        seen.add(name)
        entries.append(
            {
                "file": name,
                "aliases": aliases,
                "schema_version": item.get("schema_version"),
                "declared_metadata": item,
            }
        )
    return {
        "default_url_path": data["default_url_path"],
        "entries": entries,
        "catalog_entries": len(entries),
        "database_objects_downloaded": False,
    }


def config_catalog_url(raw):
    # Parse only a literal assignment; never import, exec or evaluate source.
    tree = ast.parse(raw.decode("utf-8"))
    assignments = [
        n.value.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "download_info_url" for t in n.targets)
        and isinstance(n.value, ast.Constant)
        and isinstance(n.value.value, str)
    ]
    if len(assignments) != 1:
        raise ValueError("catalog URL is not a unique source literal")
    return assignments[0]


def archive(name, url, reg, root):
    record = {"name": name, "request_url": url, "retrieved_at_utc": datetime.now(UTC).isoformat()}
    try:
        if urlparse(url).scheme != "https" or urlparse(url).hostname not in reg["allowed_hosts"]:
            raise ValueError("unregistered metadata URL")
        raw, final = curl_receipt(
            {"url": url, "maximum_bytes": reg["maximum_bytes_per_resource"]},
            reg["certificate_authority_bundle"],
        )
        if urlparse(final).hostname not in reg["allowed_hosts"]:
            raise ValueError("redirect outside registered hosts")
        path = root / (name + ".raw")
        with path.open("xb") as handle:
            handle.write(raw)
        record.update(
            final_url=final, raw_path=str(path), raw_bytes=len(raw), raw_sha256=digest(path)
        )
        if len(raw) > reg["maximum_bytes_per_resource"]:
            raise ValueError("metadata exceeds bound")
        record["status"] = "archived-metadata-only"
    except (OSError, ValueError) as exc:
        record.update(status="failed-retained", error_type=type(exc).__name__, error=str(exc))
    save(root / (name + ".yaml"), record)
    return record


def main():
    reg = yaml.safe_load(REGISTRATION.read_text())
    if (
        reg["metadata_acquisition_authorized"] is not True
        or reg["TLS_verification_disabled"] is not False
    ):
        raise ValueError("metadata/TLS authorization changed")
    for key in (
        "database_download_authorized",
        "numerical_synaptic_result_reads_authorized",
        "source_code_execution_authorized",
        "parameter_fitting_authorized",
        "cell_or_network_simulation_authorized",
    ):
        if reg[key] is not False:
            raise ValueError("metadata-only scope changed")
    if (
        reg["maximum_requests"] != 5
        or reg["maximum_total_bytes"] < 5 * reg["maximum_bytes_per_resource"]
    ):
        raise ValueError("registered request bounds changed")
    tls_context(reg)  # Verify pinned CA and hostname/certificate requirements before curl.
    root = Path(reg["output_directory"])
    root.mkdir()
    records, revisions = [], {}
    for name, url in reg["revision_queries"].items():
        record = archive(name + "-revision", url, reg, root)
        records.append(record)
        if record["status"] == "archived-metadata-only":
            revisions[name + "_revision"] = revision(Path(record["raw_path"]).read_bytes())
    inspection = None
    if len(revisions) == 2:
        for name, template in reg["immutable_resource_templates"].items():
            records.append(archive(name, template.format(**revisions), reg, root))
        by_name = {r["name"]: r for r in records}
        if all(r["status"] == "archived-metadata-only" for r in records):
            config_url = config_catalog_url(Path(by_name["config"]["raw_path"]).read_bytes())
            if (
                config_url
                != "https://raw.githubusercontent.com/AllenInstitute/aisynphys/download_urls/download_urls.json"
            ):
                raise ValueError("official configuration catalog provenance changed")
            ast.parse(Path(by_name["cache"]["raw_path"]).read_text())
            inspection = inspect_catalog(Path(by_name["catalog"]["raw_path"]).read_bytes())
    save(
        root / "manifest.yaml",
        {
            "registration_sha256": digest(REGISTRATION),
            "collector_sha256": digest(__file__),
            "transport_sha256": digest(
                Path(__file__).with_name("collect_synphys_source_metadata.py")
            ),
            "revisions": revisions,
            "records": records,
            "catalog": inspection,
            "source_code_executed": False,
            "database_objects_downloaded": False,
        },
    )
    print(
        yaml.safe_dump(
            {
                "archived": sum(r["status"] == "archived-metadata-only" for r in records),
                "catalog_entries": None if inspection is None else inspection["catalog_entries"],
            }
        )
    )


if __name__ == "__main__":
    main()
