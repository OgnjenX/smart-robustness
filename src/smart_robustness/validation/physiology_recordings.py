"""Read-only trace loading and fixed repeat scoring; callers enforce execution seals."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np

from smart_robustness.validation.physiology_metrics import (
    native_trace_values,
    recruitment_gate,
    spike_agreement,
    upward_crossings,
    vector,
)
from smart_robustness.validation.sst_vip_nwb_schema import safe_get, series_inventory


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def conversion_branch(version):
    major, minor = version
    return major > 1 or (major == 1 and minor > 0)


def load_recording(path, expected_hash, inventory, versions, *, authorized=False):
    if not authorized:
        raise ValueError("trace reading requires separate execution authorization")
    if digest(path) != expected_hash:
        raise ValueError("recording hash changed before reading")
    os.environ["HDF5_PLUGIN_PRELOAD"] = "::"
    import h5py

    rules = {"maximum_soft_link_resolution_steps": 32, "maximum_small_metadata_elements": 4096}
    decoded, literal = (
        versions["decoded_pipeline_version"],
        versions["literal_sdk_pipeline_version"],
    )
    if conversion_branch(decoded) != conversion_branch(literal):
        raise ValueError("pipeline conversion branches disagree")
    try:
        traces = {}
        with h5py.File(path, "r") as root:
            for prefix, epoch in (("", "Sweep"), ("experiment_", "Experiment")):
                for kind in ("stimulus", "response"):
                    base = f"epochs/{epoch}_{inventory['sweep_number']}/{kind}"
                    current = series_inventory(root, base, rules)
                    if current != inventory[prefix + kind]:
                        raise ValueError("trace metadata differs from sealed inventory")
                    if current["unit"] != {"stimulus": "Amps", "response": "Volts"}[kind]:
                        raise ValueError("trace units do not match registered SI labels")
                    if not conversion_branch(decoded) and current["conversion"] != 1.0:
                        raise ValueError("unresolved old-version physical scaling")
                    if not prefix:
                        dataset = safe_get(root, base + "/timeseries/data")
                        traces[kind] = native_trace_values(
                            dataset[()], current["conversion"], decoded
                        )
        stimulus = traces["stimulus"]
        for prefix in ("", "experiment_"):
            s, r = inventory[prefix + "stimulus"], inventory[prefix + "response"]
            if any(
                s[key] != r[key] for key in ("data_shape", "sampling_rate", "index_start", "count")
            ):
                raise ValueError("unaligned stimulus and response")
        if inventory["stimulus"]["index_start"] != 0:
            raise ValueError("native full-sweep start convention violated")
        epoch = inventory["experiment_stimulus"]
        return {
            **traces,
            "rate": inventory["stimulus"]["sampling_rate"],
            "start": epoch["index_start"],
            "count": epoch["count"],
            "source_sha256": expected_hash,
            "stimulus_sha256": hashlib.sha256(stimulus.tobytes()).hexdigest(),
        }
    finally:
        if digest(path) != expected_hash:
            raise ValueError("recording changed during read-only access")


def group_key(recording):
    return json.dumps(
        [recording["stimulus_sha256"], recording["rate"], recording["start"], recording["count"]],
        separators=(",", ":"),
    )


def epoch_events(events, recording):
    times = vector(events)
    if np.any(np.diff(times) < 0):
        raise ValueError("unsorted event times")
    start = recording["start"] / recording["rate"]
    stop = (recording["start"] + recording["count"]) / recording["rate"]
    return times[(times >= start) & (times < stop)]


def experimental_events(recording, threshold=0.0):
    return epoch_events(
        upward_crossings(recording["response"], recording["rate"], threshold=threshold), recording
    )


def score_group(recordings, predictions, parity_passes):
    if not recordings or not (len(recordings) == len(predictions) == len(parity_passes)):
        raise ValueError("repeat group coverage changed")
    if len({group_key(r) for r in recordings}) != 1:
        raise ValueError("different stimuli or epochs cannot be repeat reliability trials")
    observed = [experimental_events(r) for r in recordings]
    pairs = [spike_agreement(a, b)["f1"] for a, b in itertools.combinations(observed, 2)]
    enough = len(recordings) >= 2 and all(len(events) >= 4 for events in observed)
    reliability = float(np.mean(pairs)) if enough and all(p is not None for p in pairs) else None
    repeat_reports = []
    for r, events, prediction, parity in zip(
        recordings, observed, predictions, parity_passes, strict=True
    ):
        report = recruitment_gate(
            events, epoch_events(prediction, r), r["count"] / r["rate"], reliability
        )
        report["parity_passed"] = bool(parity)
        report["passed"] = report["passed"] and bool(parity)
        report["diagnostic_observed_spike_counts"] = {
            str(threshold): len(experimental_events(r, threshold)) for threshold in (-0.01, 0.01)
        }
        repeat_reports.append(report)
    return {
        "group_key": group_key(recordings[0]),
        "repeats": len(recordings),
        "minimum_observed_spikes_each_repeat_met": enough,
        "experimental_pairwise_F1": pairs,
        "mean_pairwise_F1": reliability,
        "repeat_reports": repeat_reports,
        "passed": all(r["passed"] for r in repeat_reports),
    }
