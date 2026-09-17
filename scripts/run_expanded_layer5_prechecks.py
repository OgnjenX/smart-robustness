"""Run registered structural and isolated checks before expanded-L5 outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.classic_sector import first_order_population_parameters
from smart_robustness.models.compartmental_hh import create_compartmental_hh_population
from smart_robustness.models.expanded_layer5 import (
    EXPANDED_LAYER5_TOPOLOGY,
    LAYER5_CELL_CLASS,
    create_expanded_layer5_population,
    expanded_layer5_parameters,
)
from smart_robustness.models.modeldb112923 import first_order_population_facts
from smart_robustness.models.selective import make_selective_population_factory
from smart_robustness.synapses import kinness_gap_total_conductance_nS


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1e-30)


def _total_density(cell, attribute: str) -> float:
    return sum(
        float(getattr(compartment, attribute) or 0.0)
        * compartment.lateral_area_cm2
        for compartment in cell.compartments
    )


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
    expanded = transformed["cell_spec"]
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
                * expanded.compartment(new.compartment).lateral_area_cm2
            )
            error = _relative_error(new_total, old_total)
            maximum_error = max(maximum_error, error)
            records.append(
                {
                    "record_id": old.record_id,
                    "old_compartment": old.compartment,
                    "new_compartment": new.compartment,
                    "old_total_mS": old_total,
                    "new_total_mS": new_total,
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
        new_compartment = expanded.compartment(new.compartment)
        old_total = kinness_gap_total_conductance_nS(
            old.conductance_density_mS_cm2,
            diameter_mm=old_compartment.diameter_mm,
            length_mm=old_compartment.length_mm,
        )
        new_total = kinness_gap_total_conductance_nS(
            new.conductance_density_mS_cm2,
            diameter_mm=new_compartment.diameter_mm,
            length_mm=new_compartment.length_mm,
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
        new_area = expanded.compartment(new.compartment).lateral_area_cm2
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


def _isolated_probe(
    factory,
    params,
    *,
    distal_compartment: str,
    brian,
) -> dict[str, object]:
    currents_pA = np.asarray((0.0, 500.0, 1500.0))
    site_results: dict[str, object] = {}
    all_finite = True
    for site in ("soma", distal_compartment):
        brian.start_scope()
        brian.defaultclock.dt = 0.02 * brian.ms
        population = factory(
            name=f"expanded_layer5_precheck_{site}",
            size=len(currents_pA),
            params=params,
            brian=brian,
        )
        spikes = brian.SpikeMonitor(population.group)
        voltage = brian.StateMonitor(
            population.group,
            ("v_soma", f"v_{distal_compartment}"),
            record=True,
            dt=0.1 * brian.ms,
        )
        network = brian.Network(population.group, spikes, voltage)
        network.run(20 * brian.ms)
        getattr(population.group, f"i_drive_{site}")[:] = currents_pA * brian.pA
        network.run(30 * brian.ms)
        soma = np.asarray(voltage.v_soma / brian.mV, dtype=float)
        distal = np.asarray(
            getattr(voltage, f"v_{distal_compartment}") / brian.mV,
            dtype=float,
        )
        spike_indices = np.asarray(spikes.i, dtype=int)
        finite = bool(np.all(np.isfinite(soma)) and np.all(np.isfinite(distal)))
        all_finite &= finite
        site_results[site] = {
            "currents_pA": currents_pA.tolist(),
            "spike_counts": [
                int(np.count_nonzero(spike_indices == index))
                for index in range(len(currents_pA))
            ],
            "soma_min_mV": np.min(soma, axis=1).tolist(),
            "soma_max_mV": np.max(soma, axis=1).tolist(),
            "distal_min_mV": np.min(distal, axis=1).tolist(),
            "distal_max_mV": np.max(distal, axis=1).tolist(),
            "finite": finite,
        }
    return {"sites": site_results, "all_finite": all_finite}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="configs/baselines/classic_smart_calibrated_v1.yaml",
    )
    parser.add_argument(
        "--study",
        default="configs/robustness/expanded_layer5_branched_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    baseline = load_frozen_classic_baseline(args.baseline)
    params_by_class = {
        facts.canonical_name: first_order_population_parameters(
            facts,
            conventions=baseline.runtime_conventions(),
        )
        for facts in first_order_population_facts()
    }
    params = params_by_class[LAYER5_CELL_CLASS]
    transformed = expanded_layer5_parameters(params)
    source = params["cell_spec"]
    expanded = transformed["cell_spec"]
    tolerance = 1e-12

    conservation_errors = {
        "area": _relative_error(
            sum(c.lateral_area_cm2 for c in expanded.compartments),
            sum(c.lateral_area_cm2 for c in source.compartments),
        ),
        "capacitance": _relative_error(
            sum(
                c.capacitance_pF(params["specific_capacitance_uF_cm2"])
                for c in expanded.compartments
            ),
            sum(
                c.capacitance_pF(params["specific_capacitance_uF_cm2"])
                for c in source.compartments
            ),
        ),
        **{
            attribute.removeprefix("g_").removesuffix("_mS_cm2"): _relative_error(
                _total_density(expanded, attribute),
                _total_density(source, attribute),
            )
            for attribute in (
                "g_leak_mS_cm2",
                "g_na_mS_cm2",
                "g_k_mS_cm2",
                "g_ca_mS_cm2",
            )
        },
    }
    port_checks, maximum_port_error = _port_checks(params, transformed)

    brian.start_scope()
    expanded_population = create_expanded_layer5_population(
        name="expanded_layer5_topology_precheck",
        size=1,
        params=params,
        brian=brian,
    )
    topology_observed = expanded_population.compiled.axial_topology_pairs
    topology_pass = topology_observed == EXPANDED_LAYER5_TOPOLOGY

    selective_factory = make_selective_population_factory(
        create_expanded_layer5_population,
        {LAYER5_CELL_CLASS},
    )
    adapter_results: dict[str, object] = {}
    changed_classes: list[str] = []
    for cell_class, class_params in params_by_class.items():
        brian.start_scope()
        control = create_compartmental_hh_population(
            name=f"adapter_manifest_{cell_class}",
            size=1,
            params=class_params,
            brian=brian,
        )
        control_manifest = _adapter_manifest(control)
        brian.start_scope()
        selected = selective_factory(
            name=f"adapter_manifest_{cell_class}",
            size=1,
            params=class_params,
            brian=brian,
        )
        selected_manifest = _adapter_manifest(selected)
        identical = control_manifest == selected_manifest
        if not identical:
            changed_classes.append(cell_class)
        adapter_results[cell_class] = {
            "identical": identical,
            "control_sha256": _manifest_sha256(control_manifest),
            "selected_sha256": _manifest_sha256(selected_manifest),
        }
    selective_dispatch_pass = changed_classes == [LAYER5_CELL_CLASS]

    isolated_probes = {
        "classic_control": _isolated_probe(
            create_compartmental_hh_population,
            params,
            distal_compartment="distal_dendrite",
            brian=brian,
        ),
        "expanded_layer5_branched": _isolated_probe(
            create_expanded_layer5_population,
            params,
            distal_compartment="distal_tuft",
            brian=brian,
        ),
    }
    all_checks_pass = bool(
        all(error <= tolerance for error in conservation_errors.values())
        and maximum_port_error <= tolerance
        and topology_pass
        and selective_dispatch_pass
        and all(probe["all_finite"] for probe in isolated_probes.values())
    )
    result = {
        "schema_version": 1,
        "status": "completed-expanded-layer5-prechecks",
        "baseline_manifest": args.baseline,
        "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
        "runtime_fingerprint": baseline.runtime_fingerprint,
        "study": args.study,
        "registration": (
            "docs/validation-results/mechanism-expanded-layer5-registration-932.yaml"
        ),
        "port_amendment": (
            "docs/validation-results/"
            "mechanism-expanded-layer5-port-amendment-932a.yaml"
        ),
        "network_outcomes_observed": False,
        "relative_tolerance": tolerance,
        "source_compartments": [c.name for c in source.compartments],
        "expanded_compartments": [c.name for c in expanded.compartments],
        "registered_topology": [list(pair) for pair in EXPANDED_LAYER5_TOPOLOGY],
        "observed_topology": [list(pair) for pair in topology_observed],
        "topology_pass": topology_pass,
        "conservation_relative_errors": conservation_errors,
        "maximum_port_relative_error": maximum_port_error,
        "port_checks": port_checks,
        "adapter_manifests": adapter_results,
        "changed_adapter_classes": changed_classes,
        "selective_dispatch_pass": selective_dispatch_pass,
        "isolated_probes": isolated_probes,
        "all_pre_network_checks_pass": all_checks_pass,
        "figure6_authorized": all_checks_pass,
        "compensation_or_refitting_used": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(json.loads(json.dumps(result)), sort_keys=False))
    if not all_checks_pass:
        raise RuntimeError("registered expanded layer-5 prechecks failed")


if __name__ == "__main__":
    main()
