"""Independent full-cohort metadata reconstruction; no inspector imports or trace reads."""

from __future__ import annotations

import hashlib
import os
import posixpath
import re
from collections import Counter
from pathlib import Path

import yaml


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def embedded_ephys_id(identifier):
    match = re.fullmatch(r"Allen Institute for Brain Science, Ephys Result (\d+)", identifier)
    require(match is not None, "unrecognized embedded identifier")
    return int(match.group(1))


def resolve(root, path):
    import h5py

    for _ in range(33):
        path = posixpath.normpath("/" + path.lstrip("/"))
        names = path.strip("/").split("/")
        group, redirect = root, None
        for i, name in enumerate(names):
            link = group.get(name, getlink=True)
            require(not isinstance(link, h5py.ExternalLink), "external link rejected")
            if isinstance(link, h5py.SoftLink):
                redirect = posixpath.join("/", *names[:i], link.path, *names[i + 1 :])
                break
            require(isinstance(link, h5py.HardLink), "missing or unsupported link")
            group = group[name]
        if redirect is None:
            return group
        path = redirect
    raise ValueError("link resolution limit exceeded")


def metadata_value(root, path):
    value = resolve(root, path)
    require(
        not value.external and not value.is_virtual and value.size <= 4096,
        "unsafe metadata storage or size",
    )
    return value[()]


def text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def reconstruct_series(root, base):
    # Only metadata and attributes are read. The data dataset is never indexed.
    data = resolve(root, base + "/timeseries/data")
    require(not data.external and not data.is_virtual, "external trace storage")
    starting = resolve(root, base + "/timeseries/starting_time")
    return {
        "data_shape": list(data.shape),
        "data_dtype": str(data.dtype),
        "unit": text(data.attrs["unit"]),
        "conversion": float(data.attrs["conversion"]),
        "sampling_rate": float(starting.attrs["rate"]),
        "index_start": int(metadata_value(root, base + "/idx_start")),
        "count": int(metadata_value(root, base + "/count")),
        "trace_array_read": False,
    }


def verify():
    os.environ["HDF5_PLUGIN_PRELOAD"] = "::"
    import h5py

    path = Path("results/sst-vip-nwb-schema-audit-1058/manifest.yaml")
    m = yaml.load(path.read_text(), Loader=yaml.CSafeLoader)
    reg = yaml.safe_load(Path(m["registration"]).read_text())
    require(digest(m["registration"]) == m["registration_sha256"], "registration changed")
    for key in ("parent", "receipt_manifest", "physiology_manifest"):
        require(digest(reg[key]) == reg[key + "_sha256"], "source lineage changed")
    for name, expected in m["implementation_sha256"].items():
        require(digest(name) == expected, "sealed inspector changed")
    source = yaml.load(Path(reg["physiology_manifest"]).read_text(), Loader=yaml.CSafeLoader)
    specimens = {s["id"]: s for s in source["physiology_records"]}
    receipts = yaml.load(Path(reg["receipt_manifest"]).read_text(), Loader=yaml.CSafeLoader)
    nwb = {
        r["preflight_record"]["source"]["file_id"]: r
        for r in receipts["records"]
        if r["preflight_record"]["source"]["resource"]["well_known_file_type"]["name"]
        == "NWBDownload"
    }
    require(len(m["records"]) == len(nwb) == 111, "full NWB cohort incomplete")
    seen = set()
    missing_names = Counter()
    missing_units = Counter()
    available_names = Counter()
    version_pairs = Counter()
    total_sweeps = 0
    matched_ephys_ids = 0
    unresolved_identifiers = []
    for record in m["records"]:
        fid = record["source"]["file_id"]
        require(fid not in seen and fid in nwb, "duplicate or unknown file")
        seen.add(fid)
        receipt = nwb[fid]
        require(
            record["source"] == receipt["preflight_record"]["source"], "source attachment changed"
        )
        require(
            yaml.load((path.parent / f"{fid}.yaml").read_text(), Loader=yaml.CSafeLoader) == record,
            "file sidecar changed",
        )
        raw_path = Path(receipt["raw_path"])
        require(
            digest(raw_path) == receipt["raw_sha256"] == record["raw_sha256"], "raw NWB changed"
        )
        specimen = specimens[record["source"]["specimen_ids"][0]]
        result = record["inspection"]
        require(result["metadata_sweeps"] == specimen["ephys_sweeps"], "metadata sweeps changed")
        require(
            not result["errors"] and record["raw_unchanged_after_inspection"],
            "file audit has unresolved errors",
        )
        with h5py.File(raw_path, "r") as root:
            epochs = resolve(root, "epochs")
            sweeps = sorted(int(n[6:]) for n in epochs if n.startswith("Sweep_"))
            experiments = sorted(int(n[11:]) for n in epochs if n.startswith("Experiment_"))
            expected = {s["sweep_number"] for s in specimen["ephys_sweeps"]}
            missing = expected - set(sweeps)
            require(
                result["sweep_numbers"] == sweeps and result["experiment_numbers"] == experiments,
                "epoch sets differ",
            )
            require(
                result["missing_metadata_sweeps"] == sorted(missing)
                and result["extra_file_sweeps"] == sorted(set(sweeps) - expected),
                "missing/extra sweep classification differs",
            )
            require(
                result["experiment_without_sweep"] == sorted(set(experiments) - set(sweeps)),
                "experiment classification differs",
            )
            identifier = text(metadata_value(root, "identifier"))
            require(
                result["file_identity_metadata"]["identifier"] == identifier, "identifier changed"
            )
            try:
                embedded = embedded_ephys_id(identifier)
            except ValueError:
                unresolved_identifiers.append(
                    {
                        "file_id": fid,
                        "identifier": identifier,
                        "attachment_ephys_result_id": specimen["ephys_result"]["id"],
                    }
                )
            else:
                require(
                    embedded == specimen["ephys_result"]["id"], "embedded ephys attachment mismatch"
                )
                matched_ephys_ids += 1
            raw_version = metadata_value(root, "general/generated_by")
            entries = [
                [text(k), text(v)] for k, v in zip(raw_version[::2], raw_version[1::2], strict=True)
            ]
            labels = [v for k, v in entries if k == "version"]
            require(len(labels) == 1, "ambiguous version")
            decoded = [int(x) for x in labels[0].split(".")[:2]]
            native = [0, 0]
            try:
                version_label = next(
                    raw_version[i + 1] for i, v in enumerate(raw_version) if v == "version"
                )
                native = [int(x) for x in version_label.split(".")[:2]]
            except (StopIteration, TypeError, ValueError):
                pass
            recorded = result["pipeline_version"]
            require(
                recorded["decoded_entries"] == entries
                and recorded["decoded_pipeline_version"] == decoded
                and recorded["literal_sdk_pipeline_version"] == native,
                "version reconstruction differs",
            )
            version_pairs[(tuple(decoded), tuple(native))] += 1
            require(
                [s["sweep_number"] for s in result["sweeps"]] == sweeps,
                "sweep inspection coverage differs",
            )
            for sweep in result["sweeps"]:
                number = sweep["sweep_number"]
                require(not sweep["errors"], "sweep audit errors retained")
                for kind in ("stimulus", "response"):
                    reconstructed = reconstruct_series(root, f"epochs/Sweep_{number}/{kind}")
                    require(reconstructed == sweep[kind], "series metadata reconstruction differs")
                    if number in experiments:
                        require(
                            reconstruct_series(root, f"epochs/Experiment_{number}/{kind}")
                            == sweep["experiment_" + kind],
                            "experiment metadata differs",
                        )
                require(
                    sweep["sampling_rates_agree"]
                    == (sweep["stimulus"]["sampling_rate"] == sweep["response"]["sampling_rate"]),
                    "rate agreement differs",
                )
            for row in specimen["ephys_sweeps"]:
                if row["sweep_number"] in missing:
                    missing_names[row["stimulus_name"]] += 1
                    missing_units[row["stimulus_units"]] += 1
                else:
                    available_names[row["stimulus_name"]] += 1
            total_sweeps += len(sweeps)
        require(digest(raw_path) == receipt["raw_sha256"], "raw changed during verification")
    require(seen == set(nwb), "independent cohort incomplete")
    return {
        "independent_reconstruction": "passed",
        "manifest_sha256": digest(path),
        "files": len(seen),
        "file_sweeps": total_sweeps,
        "embedded_ephys_result_ids_matching_attachments": matched_ephys_ids,
        "unresolved_embedded_identifiers": unresolved_identifiers,
        "embedded_identity_validation_complete": not unresolved_identifiers,
        "missing_stimulus_names": dict(missing_names),
        "missing_stimulus_units": dict(missing_units),
        "available_stimulus_names": dict(available_names),
        "version_pairs": {str(k): v for k, v in version_pairs.items()},
        "trace_arrays_read": False,
        "physiological_or_model_validation": False,
    }


if __name__ == "__main__":
    print(yaml.safe_dump(verify(), sort_keys=False))
