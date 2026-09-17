"""Run registered conservation and isolated checks before point-HH outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.classic_sector import first_order_population_parameters
from smart_robustness.models.compartmental_hh import create_compartmental_hh_population
from smart_robustness.models.modeldb112923 import first_order_population_facts
from smart_robustness.models.point_hh import (
    conserved_cortical_point_hh_parameters,
    create_conserved_cortical_point_hh_population,
)
from smart_robustness.models.selective import (
    CORTICAL_CELL_CLASSES,
    make_selective_population_factory,
)
from smart_robustness.synapses import kinness_gap_total_conductance_nS
from smart_robustness.validation.isolated_cell_matching import (
    CurrentStepProtocol,
    passive_normalized_currents_pA,
    run_current_step_protocol,
)


def _plain(value):
    return json.loads(json.dumps(value))


def _total_density(cell, attribute: str) -> float:
    return sum(
        float(getattr(compartment, attribute) or 0.0)
        * compartment.lateral_area_cm2
        for compartment in cell.compartments
    )


def _relative_error(actual: float, expected: float) -> float:
    scale = max(abs(expected), 1e-30)
    return abs(actual - expected) / scale


def _adapter_manifest(population) -> dict[str, object]:
    return {
        "cell_spec": repr(population.cell_spec),
        "compiled": repr(population.compiled),
        "variables": sorted(population.group.variables),
    }


def _manifest_sha256(manifest: dict[str, object]) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _port_checks(params, transformed) -> tuple[dict[str, object], float]:
    source = params["cell_spec"]
    point = transformed["cell_spec"]
    checks: dict[str, object] = {}
    maximum_error = 0.0
    for key in ("synaptic_ports", "external_input_ports"):
        records = []
        for old, new in zip(params[key], transformed[key], strict=True):
            old_total = (
                old.conductance_density_mS_cm2
                * source.compartment(old.compartment).lateral_area_cm2
            )
            new_total = (
                new.conductance_density_mS_cm2
                * point.soma.lateral_area_cm2
            )
            error = _relative_error(new_total, old_total)
            maximum_error = max(maximum_error, error)
            records.append(
                {
                    "record_id": old.record_id,
                    "old_compartment": old.compartment,
                    "new_compartment": new.compartment,
                    "relative_error": error,
                }
            )
        checks[key] = records

    gap_records = []
    for old, new in zip(
        params["gap_junction_ports"],
        transformed["gap_junction_ports"],
        strict=True,
    ):
        old_compartment = source.compartment(old.compartment)
        old_total = kinness_gap_total_conductance_nS(
            old.conductance_density_mS_cm2,
            diameter_mm=old_compartment.diameter_mm,
            length_mm=old_compartment.length_mm,
        )
        new_total = kinness_gap_total_conductance_nS(
            new.conductance_density_mS_cm2,
            diameter_mm=point.soma.diameter_mm,
            length_mm=point.soma.length_mm,
        )
        error = _relative_error(new_total, old_total)
        maximum_error = max(maximum_error, error)
        gap_records.append(
            {
                "record_id": old.record_id,
                "old_compartment": old.compartment,
                "new_compartment": new.compartment,
                "old_total_nS": old_total,
                "new_total_nS": new_total,
                "relative_error": error,
            }
        )
    checks["gap_junction_ports"] = gap_records

    injection_records = []
    for old, new in zip(
        params["injection_ports"],
        transformed["injection_ports"],
        strict=True,
    ):
        old_area = source.compartment(old.compartment).lateral_area_cm2
        new_area = point.soma.lateral_area_cm2
        errors = [
            _relative_error(new_value * new_area, old_value * old_area)
            for old_value, new_value in zip(
                old.sensitivities_pA_cm2,
                new.sensitivities_pA_cm2,
                strict=True,
            )
        ]
        maximum_error = max(maximum_error, *errors)
        injection_records.append(
            {
                "record_id": old.record_id,
                "old_compartment": old.compartment,
                "new_compartment": new.compartment,
                "relative_errors": errors,
            }
        )
    checks["injection_ports"] = injection_records
    return checks, maximum_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="configs/baselines/classic_smart_calibrated_v1.yaml",
    )
    parser.add_argument(
        "--study",
        default="configs/robustness/cortical_point_hh_conserved_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    baseline = load_frozen_classic_baseline(args.baseline)
    conventions = baseline.runtime_conventions()
    params_by_class = {
        facts.canonical_name: first_order_population_parameters(
            facts,
            conventions=conventions,
        )
        for facts in first_order_population_facts()
    }
    tolerance = 1e-12
    protocol = CurrentStepProtocol(
        training_currents_pA=(0.0,),
        holdout_currents_pA=(1.0,),
        pre_ms=20.0,
        step_ms=30.0,
        post_ms=10.0,
        dt_ms=0.02,
    )

    cortical_results: dict[str, object] = {}
    all_cortical_checks = True
    for cell_class in sorted(CORTICAL_CELL_CLASSES):
        params = params_by_class[cell_class]
        transformed = conserved_cortical_point_hh_parameters(params)
        source = params["cell_spec"]
        point = transformed["cell_spec"]
        source_area = sum(c.lateral_area_cm2 for c in source.compartments)
        point_area = point.soma.lateral_area_cm2
        conservation_errors = {
            "area": _relative_error(point_area, source_area),
            "capacitance": _relative_error(
                point.soma.capacitance_pF(params["specific_capacitance_uF_cm2"]),
                sum(
                    c.capacitance_pF(params["specific_capacitance_uF_cm2"])
                    for c in source.compartments
                ),
            ),
            "leak": _relative_error(
                _total_density(point, "g_leak_mS_cm2"),
                _total_density(source, "g_leak_mS_cm2"),
            ),
            "na": _relative_error(
                _total_density(point, "g_na_mS_cm2"),
                _total_density(source, "g_na_mS_cm2"),
            ),
            "k": _relative_error(
                _total_density(point, "g_k_mS_cm2"),
                _total_density(source, "g_k_mS_cm2"),
            ),
            "ca": _relative_error(
                _total_density(point, "g_ca_mS_cm2"),
                _total_density(source, "g_ca_mS_cm2"),
            ),
        }
        port_checks, maximum_port_error = _port_checks(params, transformed)
        currents = passive_normalized_currents_pA(params, (0.0, 10.0, 30.0))
        control_probe = run_current_step_protocol(
            population_factory=create_compartmental_hh_population,
            population_params=params,
            currents_pA=currents,
            protocol=protocol,
            brian=brian,
        )
        point_probe = run_current_step_protocol(
            population_factory=create_conserved_cortical_point_hh_population,
            population_params=params,
            currents_pA=currents,
            protocol=protocol,
            brian=brian,
        )
        class_pass = bool(
            len(point.compartments) == 1
            and all(error <= tolerance for error in conservation_errors.values())
            and maximum_port_error <= tolerance
            and control_probe.finite
            and point_probe.finite
        )
        all_cortical_checks &= class_pass
        cortical_results[cell_class] = {
            "source_compartments": [c.name for c in source.compartments],
            "point_compartments": [c.name for c in point.compartments],
            "conservation_relative_errors": conservation_errors,
            "maximum_port_relative_error": maximum_port_error,
            "port_checks": port_checks,
            "fixed_currents_pA": list(currents),
            "classic_control_probe": _plain(asdict(control_probe)),
            "point_hh_probe": _plain(asdict(point_probe)),
            "all_checks_pass": class_pass,
        }

    selective_factory = make_selective_population_factory(
        create_conserved_cortical_point_hh_population,
        CORTICAL_CELL_CLASSES,
    )
    thalamic_results: dict[str, object] = {}
    thalamic_identical = True
    for cell_class, params in params_by_class.items():
        if cell_class in CORTICAL_CELL_CLASSES:
            continue
        brian.start_scope()
        control = create_compartmental_hh_population(
            name=f"precheck_control_{cell_class}",
            size=1,
            params=params,
            brian=brian,
        )
        selected = selective_factory(
            name=f"precheck_selected_{cell_class}",
            size=1,
            params=params,
            brian=brian,
        )
        control_manifest = _adapter_manifest(control)
        selected_manifest = _adapter_manifest(selected)
        identical = control_manifest == selected_manifest
        thalamic_identical &= identical
        thalamic_results[cell_class] = {
            "identical": identical,
            "control_sha256": _manifest_sha256(control_manifest),
            "selected_sha256": _manifest_sha256(selected_manifest),
        }

    all_checks_pass = bool(all_cortical_checks and thalamic_identical)
    result = {
        "schema_version": 1,
        "status": "completed-cortical-point-hh-prechecks",
        "baseline_manifest": args.baseline,
        "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
        "runtime_fingerprint": baseline.runtime_fingerprint,
        "study": args.study,
        "gap_rule_amendment": (
            "docs/validation-results/"
            "mechanism-cortical-point-hh-gap-amendment-927a.yaml"
        ),
        "network_outcomes_observed": False,
        "relative_tolerance": tolerance,
        "cortical_classes": cortical_results,
        "thalamic_adapter_manifests": thalamic_results,
        "all_pre_network_checks_pass": all_checks_pass,
        "figure6_authorized": all_checks_pass,
        "compensation_or_refitting_used": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(result, sort_keys=False))
    if not all_checks_pass:
        raise RuntimeError("registered cortical point-HH prechecks failed")


if __name__ == "__main__":
    main()
