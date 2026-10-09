"""Independent reconstruction of fixed physiological gates from saved array blobs."""

from __future__ import annotations

import itertools
import json
from collections import defaultdict

import numpy as np
from verify_sst_vip_physiology_arrays import ROOT, read_array


def finite(values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError("expected finite one-dimensional data")
    return a


def within_epoch(times, r):
    a = finite(times)
    if np.any(np.diff(a) < 0):
        raise ValueError("unsorted event times")
    return a[(a >= r["start"] / r["rate"]) & (a < (r["start"] + r["count"]) / r["rate"])]


def detect(r, threshold):
    v = finite(r["response"])
    times = []
    indices = np.flatnonzero((v[:-1] < threshold) & (v[1:] >= threshold))
    for i in indices:
        t = (i + (threshold - v[i]) / (v[i + 1] - v[i])) / r["rate"]
        if not times or t - times[-1] >= 0.001:
            times.append(float(t))
    return within_epoch(times, r)


def coincidence(a, b):
    a, b = finite(a), finite(b)
    left = right = matches = 0
    while left < len(a) and right < len(b):
        if abs(a[left] - b[right]) <= 0.005:
            matches += 1
            left += 1
            right += 1
        elif a[left] < b[right]:
            left += 1
        else:
            right += 1
    total = len(a) + len(b)
    return {
        "observed_spikes": len(a),
        "predicted_spikes": len(b),
        "matches": matches,
        "f1": 2 * matches / total if total else None,
    }


def key(r):
    return json.dumps(
        [r["stimulus_sha256"], r["rate"], r["start"], r["count"]], separators=(",", ":")
    )


def group_score(recordings, predictions, parity):
    if not recordings or len(recordings) != len(predictions) or len(recordings) != len(parity):
        raise ValueError("repeat group coverage changed")
    if len({key(r) for r in recordings}) != 1:
        raise ValueError("different stimuli or epochs cannot be repeat reliability trials")
    observed = [detect(r, 0.0) for r in recordings]
    pairs = [coincidence(a, b)["f1"] for a, b in itertools.combinations(observed, 2)]
    enough = len(observed) >= 2 and min(map(len, observed)) >= 4
    reliability = float(np.mean(pairs)) if enough and None not in pairs else None
    reports = []
    for r, events, prediction, exact in zip(recordings, observed, predictions, parity, strict=True):
        agreement = coincidence(events, within_epoch(prediction, r))
        duration = r["count"] / r["rate"]
        observed_rate = agreement["observed_spikes"] / duration
        predicted_rate = agreement["predicted_spikes"] / duration
        error = abs(predicted_rate - observed_rate)
        reliable = reliability is not None and 0.5 <= reliability <= 1.0
        passed = (
            reliable
            and agreement["f1"] is not None
            and agreement["f1"] >= 0.7
            and error <= max(5.0, 0.2 * observed_rate)
            and bool(exact)
        )
        reports.append(
            {
                **agreement,
                "observed_rate_hz": observed_rate,
                "predicted_rate_hz": predicted_rate,
                "rate_error_hz": error,
                "reliable_recording": reliable,
                "passed": bool(passed),
                "parity_passed": bool(exact),
                "diagnostic_observed_spike_counts": {
                    str(t): len(detect(r, t)) for t in (-0.01, 0.01)
                },
            }
        )
    return {
        "group_key": key(recordings[0]),
        "repeats": len(recordings),
        "minimum_observed_spikes_each_repeat_met": enough,
        "experimental_pairwise_F1": pairs,
        "mean_pairwise_F1": reliability,
        "repeat_reports": reports,
        "passed": all(r["passed"] for r in reports),
    }


def reconstruct_summary(directory, records):
    buckets, failures = defaultdict(list), []
    for r in records:
        if r["recording"]["status"] != "complete":
            failures.append(
                {
                    "specimen_id": r["specimen_id"],
                    "model_id": r["model_id"],
                    "sweep_number": r["sweep_number"],
                    "status": "recording-failure",
                }
            )
        else:
            buckets[(r["specimen_id"], r["model_id"], key(r["recording"]["metadata"]))].append(r)
    groups = []
    for (specimen, model, _), repeats in buckets.items():
        if any(r["attempts"]["candidate0"]["status"] != "complete" for r in repeats):
            report = {"passed": False, "status": "candidate-exception"}
        else:
            recordings = [
                {
                    **r["recording"]["metadata"],
                    "response": read_array(directory, r["recording"]["arrays"]["response"]),
                }
                for r in repeats
            ]
            predictions = [
                read_array(
                    directory, r["attempts"]["candidate0"]["arrays"]["interpolated_spike_times"]
                )
                for r in repeats
            ]
            try:
                report = group_score(recordings, predictions, [r["parity_passed"] for r in repeats])
            except ValueError as exc:
                report = {"passed": False, "status": "invalid-scoring-events", "error": str(exc)}
        groups.append(
            {
                "specimen_id": specimen,
                "model_id": model,
                "sweep_numbers": [r["sweep_number"] for r in repeats],
                **report,
            }
        )
    models = []
    for specimen, model in sorted({(r["specimen_id"], r["model_id"]) for r in records}):
        selected = [g for g in groups if (g["specimen_id"], g["model_id"]) == (specimen, model)]
        missing = any((f["specimen_id"], f["model_id"]) == (specimen, model) for f in failures)
        models.append(
            {
                "specimen_id": specimen,
                "model_id": model,
                "passed": bool(selected) and all(g["passed"] for g in selected) and not missing,
            }
        )
    return {"groups": groups, "recording_failures": failures, "models": models}


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    if len(manifest["records"]) != 509:
        raise ValueError("terminal scoring coverage incomplete")
    actual = reconstruct_summary(ROOT, manifest["records"])
    if actual != manifest["summary"]:
        raise ValueError("physiological summary differs from independent reconstruction")
    print(
        json.dumps(
            {
                "terminal_summary_reconstructed": True,
                "groups": len(actual["groups"]),
                "models": len(actual["models"]),
                "passing_models": sum(m["passed"] for m in actual["models"]),
                "source_and_array_integrity_verification_required_separately": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
