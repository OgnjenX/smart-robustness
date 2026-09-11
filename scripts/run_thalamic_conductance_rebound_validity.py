"""Run the registered classic relay/TRN source-port rebound validity scan."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import yaml

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.classic_sector import first_order_population_parameters
from smart_robustness.models.compartmental_hh import create_compartmental_hh_population
from smart_robustness.models.modeldb112923 import first_order_population_facts
from smart_robustness.validation.thalamic_matching import (
    ConductanceReboundProtocol,
    classic_rebound_valid,
    run_conductance_rebound_protocol,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", default="configs/baselines/classic_smart_calibrated_v1.yaml"
    )
    parser.add_argument(
        "--protocol", default="configs/models/thalamic_conductance_rebound_protocol_v1.yaml"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = load_frozen_classic_baseline(args.baseline)
    raw = yaml.safe_load(Path(args.protocol).read_text())
    timing = raw["timing"]
    protocol = ConductanceReboundProtocol(**timing)
    facts = {item.canonical_name: item for item in first_order_population_facts()}

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    cells = {}
    all_valid = True
    for cell_class, cell_raw in raw["cells"].items():
        params = first_order_population_parameters(
            facts[cell_class], conventions=baseline.runtime_conventions()
        )
        external = cell_raw["tonic_external_input"]
        external_values: tuple[float | None, ...] = (None,)
        if external is not None:
            raw_values = external.get("values", [external.get("value")])
            external_values = tuple(float(value) for value in raw_values)
        conditions = []
        passing_conditions = []
        for external_value in external_values:
            tonic_external = None
            if external is not None and external_value is not None:
                tonic_external = (
                    str(external["record_id"]),
                    str(external["channel"]),
                    external_value,
                )
            result = run_conductance_rebound_protocol(
                population_factory=create_compartmental_hh_population,
                population_params=params,
                inhibition_scales=tuple(
                    float(x) for x in cell_raw["inhibition_scales"]
                ),
                inhibitory_gate_baselines={
                    str(key): float(value)
                    for key, value in cell_raw["inhibitory_gate_baselines"].items()
                },
                tonic_synaptic_gate_baselines={
                    str(key): float(value)
                    for key, value in cell_raw["tonic_synaptic_gate_baselines"].items()
                },
                tonic_external_input=tonic_external,
                protocol=protocol,
                brian=brian,
            )
            for condition in result.conditions:
                passed = classic_rebound_valid(condition, protocol=protocol)
                record = {
                    "tonic_external_value": external_value,
                    **asdict(condition),
                    "operational_rebound_pass": passed,
                }
                conditions.append(record)
                if passed:
                    passing_conditions.append(record)
        selected = (
            min(
                passing_conditions,
                key=lambda item: (
                    item["inhibition_scale"],
                    -1.0
                    if item["tonic_external_value"] is None
                    else item["tonic_external_value"],
                ),
            )
            if passing_conditions
            else None
        )
        all_valid &= selected is not None
        cells[cell_class] = {
            "conditions": conditions,
            "selected_lowest_passing_condition": selected,
            "classic_assay_valid": selected is not None,
        }

    output = {
        "schema_version": 1,
        "status": (
            "completed-classic-conductance-rebound-valid"
            if all_valid
            else "completed-classic-conductance-rebound-invalid"
        ),
        "baseline_manifest": args.baseline,
        "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
        "runtime_fingerprint": baseline.runtime_fingerprint,
        "protocol": args.protocol,
        "network_outcomes_used": False,
        "all_classic_targets_valid": all_valid,
        "cells": cells,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(output, sort_keys=False))


if __name__ == "__main__":
    main()
