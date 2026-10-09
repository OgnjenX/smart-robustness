"""Sealed full-cohort metadata inspection; experimental traces remain immutable."""

from __future__ import annotations

import os
import platform
from importlib.metadata import version
from pathlib import Path

import yaml
from collect_sst_vip_native_resources import save_yaml, sha

from smart_robustness.validation.sst_vip_nwb_schema import inspect_file

REGISTRATION = Path("docs/validation-results/post2008-sst-vip-nwb-schema-registration-1057.yaml")


def main():
    os.environ["HDF5_PLUGIN_PRELOAD"] = "::"
    import h5py

    reg = yaml.safe_load(REGISTRATION.read_text())
    if version("h5py") != reg["parser_dependency"].split("==")[1]:
        raise ValueError("registered parser version changed")
    for key in ("parent", "receipt_manifest", "physiology_manifest"):
        if sha(Path(reg[key])) != reg[key + "_sha256"]:
            raise ValueError("audit source lineage changed")
    for key in (
        "network_execution_authorized",
        "isolated_simulation_authorized",
        "parameter_fitting_authorized",
        "native_model_code_import_authorized",
    ):
        if reg[key] is not False:
            raise ValueError("metadata-only authorization changed")
    receipts = yaml.safe_load(Path(reg["receipt_manifest"]).read_text())
    specimens = yaml.safe_load(Path(reg["physiology_manifest"]).read_text())["physiology_records"]
    by_id = {s["id"]: s for s in specimens}
    nwb = [
        r
        for r in receipts["records"]
        if r["preflight_record"]["source"]["resource"]["well_known_file_type"]["name"]
        == "NWBDownload"
    ]
    if (
        len(nwb) != reg["expected_nwb_files"]
        or sum(len(s["ephys_sweeps"]) for s in specimens) != reg["expected_metadata_sweeps"]
    ):
        raise ValueError("registered metadata cohort changed")
    directory = Path(reg["output_directory"])
    directory.mkdir()
    records = []
    for receipt in nwb:
        source = receipt["preflight_record"]["source"]
        record = {
            "source": source,
            "raw_sha256": receipt["raw_sha256"],
            "status": "inspection-failed",
        }
        path = Path(receipt["raw_path"])
        try:
            if (
                receipt["status"] != "complete-receipt-awaiting-format-validation"
                or sha(path) != receipt["raw_sha256"]
            ):
                raise ValueError("NWB receipt changed before inspection")
            if len(source["specimen_ids"]) != 1:
                raise ValueError("ambiguous specimen attachment")
            specimen = by_id[source["specimen_ids"][0]]
            record["inspection"] = inspect_file(path, specimen["ephys_sweeps"], reg)
            expected_identity = {
                "general/subject/specimen_id": specimen["id"],
                "general/subject/ephys_result_id": specimen["ephys_result"]["id"],
            }
            record["expected_identity"] = expected_identity
            record["identity_discrepancies"] = {
                key: {"expected": expected, "observed": observed}
                for key, expected in expected_identity.items()
                if (observed := record["inspection"].get("file_identity_metadata", {}).get(key))
                is not None
                and str(observed) != str(expected)
            }
            record["status"] = "metadata-inspected-not-physiology-validated"
        except (OSError, KeyError, ValueError, TypeError, AttributeError) as exc:
            record.update(error_type=type(exc).__name__, error=str(exc))
        record["raw_unchanged_after_inspection"] = sha(path) == receipt["raw_sha256"]
        if not record["raw_unchanged_after_inspection"]:
            raise ValueError("experimental NWB changed during read-only inspection")
        save_yaml(directory / f"{source['file_id']}.yaml", record)
        records.append(record)
    result = {
        "schema_version": 1,
        "registration": str(REGISTRATION),
        "registration_sha256": sha(REGISTRATION),
        "implementation_sha256": {
            p: sha(Path(p))
            for p in (
                "scripts/audit_sst_vip_nwb_schema.py",
                "src/smart_robustness/validation/sst_vip_nwb_schema.py",
                "src/smart_robustness/validation/sst_vip_nwb_metadata.py",
            )
        },
        "environment": {
            "python": platform.python_version(),
            "h5py": h5py.__version__,
            "hdf5": h5py.version.hdf5_version,
            "numpy": version("numpy"),
            "plugin_preload": os.environ["HDF5_PLUGIN_PRELOAD"],
        },
        "records": records,
        "trace_arrays_read": False,
        "physiology_validated": False,
    }
    save_yaml(directory / "manifest.yaml", result)
    print(f"{len(records)} NWB files audited; errors preserved; no cell execution")


if __name__ == "__main__":
    main()
