"""Archive registered source documentation/listing only; no database reads."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen
from xml.etree import ElementTree

import yaml

REGISTRATION = Path(
    "docs/validation-results/post2008-synphys-source-metadata-registration-1087.yaml"
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def validate(reg):
    if reg["metadata_acquisition_authorized"] is not True:
        raise ValueError("metadata acquisition not authorized")
    for key in (
        "physiological_response_or_numeric_synaptic_result_reads_authorized",
        "database_download_authorized",
        "new_parameter_fitting_authorized",
        "cell_or_network_simulation_authorized",
    ):
        if reg[key] is not False:
            raise ValueError("metadata-only scope changed")
    resources = reg["resources"]
    if (
        len(resources) != reg["maximum_resources"]
        or sum(r["maximum_bytes"] for r in resources) > reg["maximum_total_bytes"]
    ):
        raise ValueError("resource bounds changed")
    names = set()
    for r in resources:
        parsed = urlparse(r["url"])
        if parsed.scheme != "https" or parsed.hostname not in reg["allowed_hosts"]:
            raise ValueError("unregistered host")
        if not r["name"].replace("-", "").isalnum() or r["name"] in names:
            raise ValueError("invalid or duplicate archive name")
        if r["maximum_bytes"] <= 0 or parsed.path.endswith((".sqlite", ".nwb", ".h5")):
            raise ValueError("invalid bounds or physiological database request")
        names.add(r["name"])


def inspect_listing(raw):
    tree = ElementTree.fromstring(raw)
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    if tree.tag != "{http://s3.amazonaws.com/doc/2006-03-01/}ListBucketResult":
        raise ValueError("not an S3 bucket listing")
    truncated = tree.findtext("s:IsTruncated", namespaces=ns)
    if truncated not in ("true", "false"):
        raise ValueError("missing listing truncation state")
    return {
        "bucket": tree.findtext("s:Name", namespaces=ns),
        "truncated": truncated == "true",
        "prefixes": [
            n.findtext("s:Prefix", namespaces=ns) for n in tree.findall("s:CommonPrefixes", ns)
        ],
        "objects": [
            {
                "key": n.findtext("s:Key", namespaces=ns),
                "size": int(n.findtext("s:Size", namespaces=ns)),
                "last_modified": n.findtext("s:LastModified", namespaces=ns),
            }
            for n in tree.findall("s:Contents", ns)
        ],
        "numeric_synaptic_outcomes_read": False,
    }


def acquire(resource, root, hosts):
    record = {"resource": resource, "retrieved_at_utc": datetime.now(UTC).isoformat()}
    try:
        with urlopen(resource["url"], timeout=30) as response:
            record["final_url"] = response.geturl()
            if urlparse(record["final_url"]).hostname not in hosts:
                raise ValueError("redirect outside registered hosts")
            raw = response.read(resource["maximum_bytes"] + 1)
        path = root / (resource["name"] + ".raw")
        with path.open("xb") as handle:
            handle.write(raw)
        record.update(raw_path=str(path), raw_bytes=len(raw), raw_sha256=digest(path))
        if len(raw) > resource["maximum_bytes"]:
            raise ValueError("resource exceeds bounded acquisition")
        if resource["name"] == "bucket-listing":
            record["inspection"] = inspect_listing(raw)
        else:
            if b"<html" not in raw.lower() and b"<!doctype html" not in raw.lower():
                raise ValueError("documentation response is not HTML")
        record["status"] = "archived-metadata-only"
    except (OSError, ValueError, ElementTree.ParseError) as exc:
        record.update(status="failed-retained", error_type=type(exc).__name__, error=str(exc))
    save(root / (resource["name"] + ".yaml"), record)
    return record


def main():
    reg = yaml.safe_load(REGISTRATION.read_text())
    validate(reg)
    root = Path(reg["output_directory"])
    root.mkdir()  # Exclusive snapshot. Never overwrite interrupted evidence.
    records = [acquire(r, root, reg["allowed_hosts"]) for r in reg["resources"]]
    save(
        root / "manifest.yaml",
        {
            "registration": str(REGISTRATION),
            "registration_sha256": digest(REGISTRATION),
            "collector_sha256": digest(__file__),
            "records": records,
            "response_or_database_reads": False,
            "network_simulation": False,
        },
    )
    print(
        yaml.safe_dump(
            {
                "resources": len(records),
                "archived": sum(r["status"] == "archived-metadata-only" for r in records),
            },
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
