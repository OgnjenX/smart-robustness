"""Run registered structural and isolated checks before any network outcome."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.classic_sector import first_order_population_parameters
from smart_robustness.models.compartmental_hh import create_compartmental_hh_population
from smart_robustness.models.mechanism_interventions import (
    create_layer5_distal_nak_disabled_population,
)
from smart_robustness.models.modeldb112923 import first_order_population_facts

EXPECTED_REMOVED_VARIABLES = frozenset(
    {
        "alpha_h_distal_dendrite",
        "alpha_m_distal_dendrite",
        "alpha_n_distal_dendrite",
        "beta_h_distal_dendrite",
        "beta_m_distal_dendrite",
        "beta_n_distal_dendrite",
        "g_k_distal_dendrite",
        "g_na_distal_dendrite",
        "h_distal_dendrite",
        "i_k_distal_dendrite",
        "i_na_distal_dendrite",
        "m_distal_dendrite",
        "n_distal_dendrite",
    }
)


def _layer5_params(baseline) -> dict[str, object]:
    facts = next(
        item
        for item in first_order_population_facts()
        if item.canonical_name == "layer5_excitatory_v1"
    )
    return first_order_population_parameters(
        facts,
        conventions=baseline.runtime_conventions(),
    )


def _passive_values(population, brian) -> dict[str, dict[str, float]]:
    return {
        name: {
            "capacitance_pF": float(
                getattr(population.group, f"C_{name}")[0] / brian.pfarad
            ),
            "leak_conductance_nS": float(
                getattr(population.group, f"g_l_{name}")[0] / brian.nsiemens
            ),
            "leak_reversal_mV": float(
                getattr(population.group, f"e_l_{name}")[0] / brian.mV
            ),
        }
        for name in population.compartments
    }


def _isolated_probe(factory, params, brian) -> dict[str, object]:
    currents_pA = np.asarray((0.0, 500.0, 1500.0))
    brian.start_scope()
    brian.defaultclock.dt = 0.02 * brian.ms
    population = factory(
        name=f"precheck_{factory.__name__}",
        size=len(currents_pA),
        params=params,
        brian=brian,
    )
    spikes = brian.SpikeMonitor(population.group)
    voltage = brian.StateMonitor(
        population.group,
        ("v_soma", "v_distal_dendrite"),
        record=True,
        dt=0.1 * brian.ms,
    )
    network = brian.Network(population.group, spikes, voltage)
    network.run(20 * brian.ms)
    rest_soma = np.asarray(population.group.v_soma / brian.mV, dtype=float)
    rest_distal = np.asarray(
        population.group.v_distal_dendrite / brian.mV,
        dtype=float,
    )
    population.group.i_drive_distal_dendrite = currents_pA * brian.pA
    network.run(30 * brian.ms)
    soma = np.asarray(voltage.v_soma / brian.mV, dtype=float)
    distal = np.asarray(voltage.v_distal_dendrite / brian.mV, dtype=float)
    spike_indices = np.asarray(spikes.i, dtype=int)
    return {
        "currents_pA": currents_pA.tolist(),
        "rest_soma_mV": rest_soma.tolist(),
        "rest_distal_mV": rest_distal.tolist(),
        "spike_counts": [
            int(np.count_nonzero(spike_indices == index))
            for index in range(len(currents_pA))
        ],
        "soma_min_mV": np.min(soma, axis=1).tolist(),
        "soma_max_mV": np.max(soma, axis=1).tolist(),
        "distal_min_mV": np.min(distal, axis=1).tolist(),
        "distal_max_mV": np.max(distal, axis=1).tolist(),
        "finite": bool(np.all(np.isfinite(soma)) and np.all(np.isfinite(distal))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="configs/baselines/classic_smart_calibrated_v1.yaml",
    )
    parser.add_argument(
        "--study",
        default="configs/robustness/layer5_distal_nak_ablation_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    baseline = load_frozen_classic_baseline(args.baseline)
    params = _layer5_params(baseline)
    brian.start_scope()
    control = create_compartmental_hh_population(
        name="layer5_precheck_control",
        size=1,
        params=params,
        brian=brian,
    )
    ablated = create_layer5_distal_nak_disabled_population(
        name="layer5_precheck_ablated",
        size=1,
        params=params,
        brian=brian,
    )

    control_variables = set(control.group.variables)
    ablated_variables = set(ablated.group.variables)
    removed_variables = control_variables - ablated_variables
    added_variables = ablated_variables - control_variables
    structural_checks = {
        "removed_variables_exact": removed_variables == EXPECTED_REMOVED_VARIABLES,
        "no_variables_added": not added_variables,
        "compartments_match": ablated.compartments == control.compartments,
        "synaptic_ports_match": (
            ablated.compiled.synaptic_ports == control.compiled.synaptic_ports
        ),
        "external_input_ports_match": (
            ablated.compiled.external_input_ports
            == control.compiled.external_input_ports
        ),
        "injection_ports_match": (
            ablated.compiled.injection_ports == control.compiled.injection_ports
        ),
        "axial_parameters_match": (
            ablated.compiled.axial_parameter_names
            == control.compiled.axial_parameter_names
        ),
        "passive_values_match": (
            _passive_values(ablated, brian) == _passive_values(control, brian)
        ),
    }
    probes = {
        "classic_control": _isolated_probe(
            create_compartmental_hh_population,
            params,
            brian,
        ),
        "layer5_distal_nak_disabled": _isolated_probe(
            create_layer5_distal_nak_disabled_population,
            params,
            brian,
        ),
    }
    all_checks_pass = bool(
        all(structural_checks.values())
        and all(probe["finite"] for probe in probes.values())
    )
    result = {
        "schema_version": 1,
        "status": "completed-layer5-distal-nak-prechecks",
        "baseline_manifest": args.baseline,
        "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
        "runtime_fingerprint": baseline.runtime_fingerprint,
        "study": args.study,
        "network_outcomes_observed": False,
        "removed_variables": sorted(removed_variables),
        "added_variables": sorted(added_variables),
        "structural_checks": structural_checks,
        "passive_values": _passive_values(control, brian),
        "isolated_probes": probes,
        "all_pre_network_checks_pass": all_checks_pass,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(result, sort_keys=False))
    if not all_checks_pass:
        raise RuntimeError("registered layer-5 distal Na/K prechecks failed")


if __name__ == "__main__":
    main()
