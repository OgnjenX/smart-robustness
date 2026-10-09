"""Archive the preregistered Allen metadata inventory without executing models."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import yaml

from smart_robustness.validation.sst_vip_source_inventory import inventory

REGISTRATION = Path(
    "docs/validation-results/post2008-sst-vip-cell-source-inventory-registration-1027.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_path = args.output.with_suffix(".raw.json")
    if args.output.exists() or raw_path.exists():
        raise FileExistsError("inventory artifacts already exist; do not overwrite")
    registration_bytes = REGISTRATION.read_bytes()
    registration = yaml.safe_load(registration_bytes)
    if any(
        registration[key] is not False
        for key in (
            "network_execution_authorized",
            "parameter_fitting_authorized",
            "isolated_simulation_authorized",
        )
    ):
        raise ValueError("source-only authorization changed")
    source = registration["sources"]
    url = source["endpoint"] + "?" + urlencode({"q": source["metadata_query"]})
    with urlopen(url, timeout=60) as response:
        raw = response.read()
    # Preserve the received bytes even if parsing or completeness validation fails.
    with raw_path.open("xb") as handle:
        handle.write(raw)
    result = inventory(json.loads(raw))
    result.update(
        {
            "schema_version": 1,
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "request_url": url,
            "registration": str(REGISTRATION),
            "registration_sha256": hashlib.sha256(registration_bytes).hexdigest(),
            "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "classifier_sha256": hashlib.sha256(
                Path("src/smart_robustness/validation/sst_vip_source_inventory.py").read_bytes()
            ).hexdigest(),
            "raw_metadata": str(raw_path),
            "raw_metadata_sha256": hashlib.sha256(raw).hexdigest(),
            "raw_metadata_bytes": len(raw),
        }
    )
    with args.output.open("x") as handle:
        yaml.safe_dump(result, handle, sort_keys=False)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "total_records",
                    "included_records",
                    "excluded_records",
                    "stratum_counts",
                    "raw_metadata_sha256",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
