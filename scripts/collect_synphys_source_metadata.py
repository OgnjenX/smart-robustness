"""Archive registered source documentation/listing only; no database reads."""

from __future__ import annotations

import argparse
import hashlib
import ssl
import subprocess
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


def load_registration(path):
    reg = yaml.safe_load(path.read_text())
    if "parent_registration" not in reg:
        validate(reg)
        return reg
    for key in ("parent_registration", "previous_assessment", "prior_result"):
        if digest(reg[key]) != reg[key + "_sha256"]:
            raise ValueError("transport retry lineage changed")
    if reg["insecure_transport_authorized"] is not False:
        raise ValueError("insecure transport not allowed")
    for key in (
        "database_download_authorized",
        "physiological_response_or_numeric_synaptic_result_reads_authorized",
        "new_parameter_fitting_authorized",
        "cell_or_network_simulation_authorized",
    ):
        if reg[key] is not False:
            raise ValueError("retry metadata-only scope changed")
    parent = yaml.safe_load(Path(reg["parent_registration"]).read_text())
    if parent["output_directory"] == reg["output_directory"]:
        raise ValueError("retry must preserve original output directory")
    # Resource URLs, bounds and permissions are inherited, never overridden.
    parent.update(
        {
            k: reg[k]
            for k in (
                "output_directory",
                "certificate_authority_bundle",
                "certificate_authority_sha256",
            )
        }
    )
    parent["transport"] = reg.get("transport", "urllib")
    if parent["transport"] not in ("urllib", "verified-system-curl"):
        raise ValueError("unregistered transport")
    validate(parent)
    return parent


def tls_context(reg):
    if "certificate_authority_bundle" not in reg:
        return None
    if digest(reg["certificate_authority_bundle"]) != reg["certificate_authority_sha256"]:
        raise ValueError("registered CA bytes changed")
    context = ssl.create_default_context(cafile=reg["certificate_authority_bundle"])
    if not context.check_hostname or context.verify_mode != ssl.CERT_REQUIRED:
        raise ValueError("TLS verification must remain enabled")
    return context


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


def curl_receipt(resource, ca):
    """Bound stdout reads even if a server omits Content-Length. No insecure flags."""
    command = [
        "/usr/bin/curl",
        "--silent",
        "--show-error",
        "--fail",
        "--location",
        "--proto",
        "=https",
        "--proto-redir",
        "=https",
        "--max-time",
        "30",
        "--max-filesize",
        str(resource["maximum_bytes"]),
        "--cacert",
        ca,
        "--write-out",
        "\n%{url_effective}\n%{response_code}",
        resource["url"],
    ]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        receipt = process.stdout.read(resource["maximum_bytes"] + 1025)
        if len(receipt) == resource["maximum_bytes"] + 1025:
            process.kill()
        stderr = process.stderr.read(4096)
        code = process.wait(timeout=35)
    if code != 0:
        raise OSError(f"verified curl exit {code}: {stderr.decode(errors='replace')}")
    try:
        raw, final, status = receipt.rsplit(b"\n", 2)
        final = final.decode("ascii")
        if int(status) != 200:
            raise ValueError("metadata HTTP status is not 200")
    except (ValueError, UnicodeError) as exc:
        raise ValueError("malformed bounded curl receipt") from exc
    return raw, final


def acquire(resource, root, hosts, context=None, *, transport="urllib", ca=None):
    record = {"resource": resource, "retrieved_at_utc": datetime.now(UTC).isoformat()}
    try:
        if transport == "verified-system-curl":
            raw, record["final_url"] = curl_receipt(resource, ca)
        else:
            with urlopen(resource["url"], timeout=30, context=context) as response:
                record["final_url"] = response.geturl()
                raw = response.read(resource["maximum_bytes"] + 1)
        if urlparse(record["final_url"]).hostname not in hosts:
            raise ValueError("redirect outside registered hosts")
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, default=REGISTRATION)
    path = parser.parse_args().registration
    reg = load_registration(path)
    context = tls_context(reg)
    root = Path(reg["output_directory"])
    root.mkdir()  # Exclusive snapshot. Never overwrite interrupted evidence.
    transport = reg.get("transport", "urllib")
    records = [
        acquire(
            r,
            root,
            reg["allowed_hosts"],
            context,
            transport=transport,
            ca=reg.get("certificate_authority_bundle"),
        )
        for r in reg["resources"]
    ]
    save(
        root / "manifest.yaml",
        {
            "registration": str(path),
            "registration_sha256": digest(path),
            "collector_sha256": digest(__file__),
            "certificate_authority_bundle": reg.get("certificate_authority_bundle"),
            "certificate_authority_sha256": reg.get("certificate_authority_sha256"),
            "TLS_verification_disabled": False,
            "transport": transport,
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
