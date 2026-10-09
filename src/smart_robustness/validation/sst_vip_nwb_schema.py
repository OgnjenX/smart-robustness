"""Read-only Allen legacy NWB metadata audit; never read trace arrays."""

from __future__ import annotations

import math
import posixpath

import numpy as np

from smart_robustness.validation.sst_vip_nwb_metadata import pipeline_versions, text_value


def safe_get(root, path, maximum_steps=32):
    import h5py

    parts = posixpath.normpath("/" + path.lstrip("/")).strip("/").split("/")
    current, prefix, steps = root, [], 0
    while parts:
        name = parts.pop(0)
        link = current.get(name, getlink=True)
        if isinstance(link, h5py.ExternalLink):
            raise TypeError("external HDF5 link forbidden")
        if isinstance(link, h5py.SoftLink):
            steps += 1
            if steps > maximum_steps:
                raise ValueError("soft-link resolution limit exceeded")
            destination = posixpath.normpath(posixpath.join("/", *prefix, link.path))
            parts = destination.strip("/").split("/") + parts
            current, prefix = root, []
            continue
        if not isinstance(link, h5py.HardLink):
            raise TypeError("missing or unsupported HDF5 link")
        current = current[name]
        prefix.append(name)
    return current


def small_value(root, path, rules):
    obj = safe_get(root, path, rules["maximum_soft_link_resolution_steps"])
    if obj.external or obj.is_virtual:
        raise ValueError("external dataset storage forbidden")
    if obj.size > rules["maximum_small_metadata_elements"]:
        raise ValueError("small metadata element limit exceeded")
    return obj[()]


def scalar(root, path, rules):
    value = np.asarray(small_value(root, path, rules))
    if value.size != 1:
        raise ValueError("expected scalar metadata")
    return value.item()


def finite_positive(value, name):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("invalid " + name)
    return result


def series_inventory(root, base, rules):
    data = safe_get(root, base + "/timeseries/data", rules["maximum_soft_link_resolution_steps"])
    if data.external or data.is_virtual:
        raise ValueError("external or virtual trace storage forbidden")
    if len(data.shape) != 1 or not data.shape[0] or data.dtype.kind not in "fiu":
        raise ValueError("unsupported trace shape or dtype")
    conversion = float(data.attrs["conversion"])
    if not math.isfinite(conversion):
        raise ValueError("nonfinite conversion")
    unit = text_value(data.attrs["unit"])
    if not unit.startswith(("A", "V")):
        raise ValueError("unsupported trace unit")
    starting = safe_get(
        root, base + "/timeseries/starting_time", rules["maximum_soft_link_resolution_steps"]
    )
    rate = finite_positive(starting.attrs["rate"], "sampling rate")
    start = scalar(root, base + "/idx_start", rules)
    count = scalar(root, base + "/count", rules)
    if (
        not isinstance(start, int)
        or not isinstance(count, int)
        or start < 0
        or count <= 0
        or start + count > data.shape[0]
    ):
        raise ValueError("invalid epoch index/count bounds")
    return {
        "data_shape": list(data.shape),
        "data_dtype": str(data.dtype),
        "unit": unit,
        "conversion": conversion,
        "sampling_rate": rate,
        "index_start": start,
        "count": count,
        "trace_array_read": False,
    }


def inspect_file(path, metadata_sweeps, rules):
    import h5py

    result = {"path": str(path), "errors": [], "sweeps": [], "trace_arrays_read": False}
    with h5py.File(path, "r") as root:
        epochs = safe_get(root, "epochs", rules["maximum_soft_link_resolution_steps"])
        if len(epochs) > rules["maximum_epoch_names_per_file"]:
            raise ValueError("epoch name limit exceeded")
        sweep_numbers = sorted(
            int(n.removeprefix("Sweep_")) for n in epochs if n.startswith("Sweep_")
        )
        experiment_numbers = sorted(
            int(n.removeprefix("Experiment_")) for n in epochs if n.startswith("Experiment_")
        )
        expected = [s["sweep_number"] for s in metadata_sweeps]
        if len(set(expected)) != len(expected):
            raise ValueError("duplicate sweep metadata identity")
        result.update(
            sweep_numbers=sweep_numbers,
            experiment_numbers=experiment_numbers,
            metadata_sweeps=metadata_sweeps,
            missing_metadata_sweeps=sorted(set(expected) - set(sweep_numbers)),
            extra_file_sweeps=sorted(set(sweep_numbers) - set(expected)),
        )
        result["experiment_without_sweep"] = sorted(set(experiment_numbers) - set(sweep_numbers))
        try:
            values = small_value(root, "general/generated_by", rules)
            if np.asarray(values).ndim != 1:
                raise ValueError("unsupported generated_by shape")
            result["pipeline_version"] = pipeline_versions(values)
            result["generated_by_value_types"] = [type(v).__name__ for v in values]
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            result["errors"].append({"field": "generated_by", "error": str(exc)})
        for name in (
            "identifier",
            "general/subject/specimen_id",
            "general/subject/ephys_result_id",
        ):
            try:
                value = scalar(root, name, rules)
                result.setdefault("file_identity_metadata", {})[name] = (
                    text_value(value) if isinstance(value, (str, bytes)) else value
                )
            except (KeyError, ValueError, TypeError, AttributeError) as exc:
                result.setdefault("unresolved_identity_metadata", []).append(
                    {"field": name, "error": str(exc)}
                )
        for number in sweep_numbers:
            entry = {"sweep_number": number, "errors": []}
            for kind in ("stimulus", "response"):
                try:
                    entry[kind] = series_inventory(root, f"epochs/Sweep_{number}/{kind}", rules)
                except (KeyError, ValueError, TypeError, AttributeError, OSError) as exc:
                    entry["errors"].append({"field": kind, "error": str(exc)})
            if "stimulus" in entry and "response" in entry:
                entry["sampling_rates_agree"] = (
                    entry["stimulus"]["sampling_rate"] == entry["response"]["sampling_rate"]
                )
            if number in experiment_numbers:
                for kind in ("stimulus", "response"):
                    try:
                        entry["experiment_" + kind] = series_inventory(
                            root, f"epochs/Experiment_{number}/{kind}", rules
                        )
                    except (KeyError, ValueError, TypeError, AttributeError, OSError) as exc:
                        entry["errors"].append({"field": "experiment_" + kind, "error": str(exc)})
            result["sweeps"].append(entry)
    return result
