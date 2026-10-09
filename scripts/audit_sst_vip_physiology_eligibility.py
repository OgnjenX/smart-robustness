"""Metadata-only physiology partition; no NWB traces or cell simulations."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

import yaml

REG = Path("docs/validation-results/post2008-sst-vip-physiology-eligibility-registration-1065.yaml")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify(metadata, structure):
    reasons = []
    label = metadata["stimulus_name"]
    role = {"Noise 1": "source-fitting-protocol", "Noise 2": "source-holdout-protocol"}.get(
        label, "potential-source-parameter-estimation-protocol"
    )
    if label not in ("Noise 1", "Noise 2"):
        reasons.append("outside-noise-validation-scope")
    if metadata["stimulus_units"] != "Amps":
        reasons.append("not-source-labelled-current-clamp")
    if structure is None:
        reasons.append("missing-file-sweep")
    else:
        if structure["errors"]:
            reasons.append("schema-errors")
        if not structure.get("sampling_rates_agree", False):
            reasons.append("sampling-rate-disagreement")
        for prefix in ("", "experiment_"):
            s, r = structure.get(prefix + "stimulus"), structure.get(prefix + "response")
            if s is None or r is None:
                reasons.append(prefix + "missing-series")
                continue
            if s["unit"] != "Amps" or r["unit"] != "Volts":
                reasons.append(prefix + "non-SI-labelled-series")
            if s["data_shape"] != r["data_shape"] or any(
                s[key] != r[key] for key in ("sampling_rate", "index_start", "count")
            ):
                reasons.append(prefix + "series-alignment-disagreement")
    return {"role": role, "eligible_noise_sweep": not reasons, "reasons": reasons}


def main():
    reg = yaml.safe_load(REG.read_text())
    for key in ("schema_manifest", "parameter_manifest", "parent"):
        if sha(reg[key]) != reg[key + "_sha256"]:
            raise ValueError("eligibility lineage changed")
    for path, expected in reg["implementation_sha256"].items():
        if sha(path) != expected:
            raise ValueError("eligibility implementation changed")
    models = {}
    for record in yaml.safe_load(Path(reg["parameter_manifest"]).read_text())["records"]:
        source = record["source"]
        if not source["template"]["name"].startswith("Biophysical"):
            models.setdefault(source["specimen_id"], []).append(source["model_id"])
    records, totals = [], Counter()
    audit = yaml.safe_load(Path(reg["schema_manifest"]).read_text())
    for record in audit["records"]:
        inspected = record["inspection"]
        specimen = record["source"]["specimen_ids"][0]
        identifier = inspected["file_identity_metadata"].get("identifier", "")
        match = re.fullmatch(r"Allen Institute for Brain Science, Ephys Result (\d+)", identifier)
        recognized = (
            bool(match)
            and int(match.group(1))
            == record["expected_identity"]["general/subject/ephys_result_id"]
        )
        if match and not recognized:
            raise ValueError("recognized embedded identity mismatch")
        by_number = {s["sweep_number"]: s for s in inspected["sweeps"]}
        sweeps = []
        for metadata in inspected["metadata_sweeps"]:
            classification = classify(metadata, by_number.get(metadata["sweep_number"]))
            sweeps.append(
                {
                    "sweep_number": metadata["sweep_number"],
                    "stimulus_name": metadata["stimulus_name"],
                    **classification,
                }
            )
            totals[classification["role"]] += classification["eligible_noise_sweep"]
        noise2 = sum(
            s["eligible_noise_sweep"] and s["role"] == "source-holdout-protocol" for s in sweeps
        )
        primary = recognized and bool(models.get(specimen)) and noise2 >= 2
        totals["primary_candidate_specimens"] += primary
        records.append(
            {
                "specimen_id": specimen,
                "model_ids": sorted(models.get(specimen, [])),
                "embedded_identity_recognized": recognized,
                "eligible_noise2_sweeps": noise2,
                "primary_candidate": primary,
                "sweeps": sweeps,
            }
        )
    if len(records) != 111 or sum(len(r["sweeps"]) for r in records) != 8380:
        raise ValueError("full metadata cohort changed")
    result = {
        "registration_sha256": sha(REG),
        "records": records,
        "totals": dict(totals),
        "trace_arrays_read": False,
        "cell_simulations_performed": False,
        "historical_per_model_training_provenance_verified": False,
    }
    destination = Path(reg["output_path"])
    with destination.open("x") as handle:
        yaml.safe_dump(result, handle, sort_keys=False)
    print(dict(totals))


if __name__ == "__main__":
    main()
