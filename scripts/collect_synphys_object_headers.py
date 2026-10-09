"""Archive HEAD-only public object metadata, never database bodies."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import yaml
from collect_synphys_source_metadata import digest, save, tls_context

REGISTRATION = Path(
    "docs/validation-results/post2008-synphys-object-headers-registration-1098.yaml"
)


def object_urls(reg, manifest):
    catalog = manifest["catalog"]
    base = catalog["default_url_path"]
    if base != "https://allen-synphys.s3-us-west-2.amazonaws.com":
        raise ValueError("unregistered object host")
    declared = {item["file"] for item in catalog["entries"]}
    names = reg["objects"]
    if len(names) != 3 or len(set(names)) != 3:
        raise ValueError("object count changed")
    urls = []
    for name in names:
        if name not in declared or name not in {
            "synphys_r2.1_small.sqlite",
            "synphys_r2.1_medium.sqlite",
            "synphys_r2.1_full.sqlite",
        }:
            raise ValueError("undeclared object")
        urls.append(base + "/" + name)
    return urls


def main():
    reg = yaml.safe_load(REGISTRATION.read_text())
    if reg["request_method"] != "HEAD" or reg["redirects_followed"] is not False:
        raise ValueError("HEAD-only boundary changed")
    for key in (
        "database_download_authorized", "numeric_response_reads_authorized",
        "parameter_fitting_authorized", "network_execution_authorized",
    ):
        if reg[key] is not False:
            raise ValueError("metadata scope changed")
    if (reg["maximum_requests"], reg["maximum_seconds_per_request"],
        reg["maximum_header_bytes_per_object"]) != (3, 30, 65536):
        raise ValueError("request bounds changed")
    source = Path(reg["catalog_manifest"])
    if digest(source) != reg["catalog_manifest_sha256"]:
        raise ValueError("catalog identity changed")
    urls = object_urls(reg, yaml.safe_load(source.read_text()))
    tls_context(reg)
    root = Path(reg["output_directory"])
    root.mkdir()
    records = []
    for url in urls:
        name = Path(urlparse(url).path).name
        record = {"url": url, "requested_at_utc": datetime.now(UTC).isoformat()}
        args = [
            "/usr/bin/curl", "--silent", "--show-error", "--head",
            "--proto", "=https", "--max-time", "30", "--cacert",
            reg["certificate_authority_bundle"], url,
        ]
        try:
            with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
                raw = proc.stdout.read(65537)
                if len(raw) > 65536:
                    proc.kill()
                code = proc.wait(timeout=35)
            path = root / (name + ".headers")
            with path.open("xb") as handle:
                handle.write(raw)
            record.update(raw_path=str(path), raw_sha256=digest(path), raw_bytes=len(raw),
                          client_exit_code=code, status="headers-retained")
            if len(raw) > 65536 or code:
                record["status"] = "transport-failure-retained"
        except (OSError, subprocess.TimeoutExpired) as exc:
            record.update(status="failure-retained", error=str(exc))
        records.append(record)
        save(root / (name + ".yaml"), record)
    save(root / "manifest.yaml", {
        "registration_sha256": digest(REGISTRATION), "collector_sha256": digest(__file__),
        "records": records, "database_objects_downloaded": False,
    })
    print("HEAD-only acquisition terminal; inspect retained HTTP statuses independently.")


if __name__ == "__main__":
    main()
