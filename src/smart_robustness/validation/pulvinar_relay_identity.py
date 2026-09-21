"""Rest-only V2 relay source-identity diagnostic; no synaptic stimulation."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..classic_sector import first_order_population_parameters
from ..modeldb_projections import MODELDB_FULL
from ..models.compartmental_hh import create_compartmental_hh_population
from ..models.modeldb112923 import second_order_population_facts
from ..standalone import build_and_run_cpp_standalone

ARMS = ("frozen_dispatch", "source_faithful_modeldb")


def relay_parameters(baseline, arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError("unregistered relay identity arm")
    facts = next(
        fact for fact in second_order_population_facts()
        if fact.canonical_name == "thalamic_relay_v2"
    )
    conventions = baseline.runtime_conventions()
    if arm == "source_faithful_modeldb":
        conventions = replace(
            conventions,
            intrinsic_cell_convention="modeldb_112923",
            axial_convention="kinness_serialized_edge",
        )
    return first_order_population_parameters(
        facts, conventions=conventions, catalog=MODELDB_FULL,
    )


@dataclass
class RelayIdentityAssay:
    network: Any
    populations: dict[str, Any]
    state_monitors: dict[str, Any]
    spike_monitors: dict[str, Any]


def build_relay_identity_assay(*, baseline, dt_ms: float, brian) -> RelayIdentityAssay:
    if dt_ms not in (0.01, 0.005):
        raise ValueError("unregistered integration step")
    brian.defaultclock.dt = dt_ms * brian.ms
    populations = {
        arm: create_compartmental_hh_population(
            name=f"relay_identity_{arm}", size=1,
            params=relay_parameters(baseline, arm), brian=brian,
        )
        for arm in ARMS
    }
    states = {
        arm: brian.StateMonitor(
            population.group,
            ["v_soma", "v_proximal_dendrite", "v_distal_dendrite"],
            record=True, when="end",
        )
        for arm, population in populations.items()
    }
    spikes = {
        arm: brian.SpikeMonitor(population.group)
        for arm, population in populations.items()
    }
    for monitor in states.values():
        monitor.active = False
    network = brian.Network(
        *(population.group for population in populations.values()),
        *states.values(),
        *spikes.values(),
    )
    return RelayIdentityAssay(network, populations, states, spikes)


def summarize_rest(arrays: dict[str, np.ndarray]) -> dict:
    rows = []
    for arm in ARMS:
        soma = arrays[f"{arm}_v_soma_mV"]
        proximal = arrays[f"{arm}_v_proximal_dendrite_mV"]
        distal = arrays[f"{arm}_v_distal_dendrite_mV"]
        spikes = arrays[f"{arm}_spike_time_ms"]
        finite = bool(all(np.all(np.isfinite(x)) for x in (soma, proximal, distal, spikes)))
        rows.append({
            "arm": arm,
            "finite": finite,
            "soma_mean_mV": float(np.mean(soma)),
            "soma_peak_to_peak_mV": float(np.ptp(soma)),
            "proximal_peak_to_peak_mV": float(np.ptp(proximal)),
            "distal_peak_to_peak_mV": float(np.ptp(distal)),
            "spike_count": int(spikes.size),
        })
    return {"arms": rows}


def source_faithful_numerical_gate(coarse: dict, fine: dict) -> dict:
    c = next(row for row in coarse["arms"] if row["arm"] == "source_faithful_modeldb")
    f = next(row for row in fine["arms"] if row["arm"] == "source_faithful_modeldb")
    failures = []
    for label, row in (("coarse", c), ("fine", f)):
        reasons = []
        if not row["finite"]:
            reasons.append("nonfinite")
        if row["soma_peak_to_peak_mV"] > 0.1:
            reasons.append("soma_not_quiet")
        if row["spike_count"] != 0:
            reasons.append("spikes")
        if reasons:
            failures.append({"scope": label, "reasons": reasons})
    if abs(c["soma_mean_mV"] - f["soma_mean_mV"]) > 0.5:
        failures.append({"scope": "cross-step", "reasons": ["soma_mean"]})
    if abs(c["soma_peak_to_peak_mV"] - f["soma_peak_to_peak_mV"]) > 0.05:
        failures.append({"scope": "cross-step", "reasons": ["soma_peak_to_peak"]})
    return {"pass": not failures, "failures": failures}


def simulate_rest(*, baseline, dt_ms: float) -> dict[str, np.ndarray]:
    """Run both zero-input arms fresh and retain only the final 100 ms."""
    import brian2 as brian

    with tempfile.TemporaryDirectory(prefix="smart-relay-identity-", dir="/private/tmp") as temp:
        try:
            brian.set_device("cpp_standalone", directory=temp, build_on_run=False)
            brian.start_scope()
            assay = build_relay_identity_assay(baseline=baseline, dt_ms=dt_ms, brian=brian)
            assay.network.run(900 * brian.ms)
            for monitor in assay.state_monitors.values():
                monitor.active = True
            assay.network.run(100 * brian.ms)
            build_and_run_cpp_standalone(brian, temp)
            first_state = assay.state_monitors[ARMS[0]]
            arrays = {
                "time_ms": np.array(first_state.t / brian.ms),
                "dt_ms": np.array(dt_ms),
            }
            for arm in ARMS:
                state = assay.state_monitors[arm]
                if not np.array_equal(np.array(state.t / brian.ms), arrays["time_ms"]):
                    raise ValueError("identity arms have different recording times")
                arrays[f"{arm}_v_soma_mV"] = np.array(state.v_soma[0] / brian.mV)
                arrays[f"{arm}_v_proximal_dendrite_mV"] = np.array(
                    state.v_proximal_dendrite[0] / brian.mV
                )
                arrays[f"{arm}_v_distal_dendrite_mV"] = np.array(
                    state.v_distal_dendrite[0] / brian.mV
                )
                arrays[f"{arm}_spike_time_ms"] = np.array(
                    assay.spike_monitors[arm].t / brian.ms
                )
            return arrays
        finally:
            brian.device.reinit()
            brian.set_device("runtime")
