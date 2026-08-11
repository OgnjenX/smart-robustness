from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import create_population
from .synapses import connect_conductance


@dataclass
class BuiltCircuit:
    network: Any
    spikes: dict[str, Any]
    rates: dict[str, Any]


def build_minimal_benchmark(config: dict[str, Any], brian: Any) -> BuiltCircuit:
    """Build an explicitly reduced SMART match/mismatch benchmark.

    Population names preserve the mechanistic roles of the first-order loop;
    sizes and weights are exploratory and never presented as the full paper model.
    """
    model = config["model"]
    net_cfg = config["network"]
    pops = {}
    for pop_name, size in net_cfg["populations"].items():
        pops[pop_name] = create_population(
            model["name"], name=pop_name, size=int(size), params=model["parameters"], brian=brian
        )

    drive = net_cfg["drive_pA"]
    for name, value in drive.items():
        pops[name].i_drive = float(value) * brian.pA

    condition = config["condition"]
    if condition == "match":
        pops["cortex_exc"].i_drive += net_cfg["match_bias_pA"] * brian.pA
    else:
        pops["nonspecific"].i_drive += net_cfg["mismatch_bias_pA"] * brian.pA

    synapses = []
    for index, edge in enumerate(net_cfg["connections"]):
        synapses.append(
            connect_conductance(
                pops[edge["pre"]],
                pops[edge["post"]],
                receptor=edge["receptor"],
                probability=float(edge["p"]),
                weight_ns=float(edge["weight_nS"]),
                delay_ms=float(edge["delay_ms"]),
                depletion=float(edge.get("depletion", 0.1)),
                recovery_ms=float(edge.get("recovery_ms", 100.0)),
                brian=brian,
                name=f"syn_{index}_{edge['pre']}_{edge['post']}",
            )
        )

    spikes = {name: brian.SpikeMonitor(pop, name=f"spikes_{name}") for name, pop in pops.items()}
    rates = {
        name: brian.PopulationRateMonitor(pop, name=f"rates_{name}") for name, pop in pops.items()
    }
    objects = [*pops.values(), *synapses, *spikes.values(), *rates.values()]
    return BuiltCircuit(brian.Network(objects), spikes, rates)

