"""Opt-in collapsed L5 SST-like distal feedback for registered sensitivity maps."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from ..models.currents import biexponential_normalization
from ..models.ports import SynapticPortSpec
from ..projections import Receptor
from .compartmental_hh import create_compartmental_hh_population

L5_SST_LIKE_ROUTE_ID = "post2008.l5_sst_like.collapsed_feedback.v1"
L5_SST_LIKE_PORT_NAME = "post2008_l5_sst_like_distal_gaba"
L5_EXCITATORY_SUFFIX = "layer5_excitatory_v1"
DISTAL_COMPARTMENT = "distal_dendrite"
GABA_REVERSAL_MV = -70.0
GABA_RISE_MS = 1.0
GABA_FALL_MS = 7.0


def validate_route_parameters(*, total_conductance_nS: float, delay_ms: float) -> None:
    """Validate prospective route coordinates without reading network outcomes."""

    if not math.isfinite(total_conductance_nS) or total_conductance_nS < 0:
        raise ValueError("total conductance must be finite and nonnegative")
    if not math.isfinite(delay_ms) or delay_ms <= 0:
        raise ValueError("collapsed feedback delay must be finite and positive")


def l5_sst_like_port(*, cell_spec: Any, total_conductance_nS: float) -> SynapticPortSpec:
    """Compile a distal GABA-A port with exactly normalized total conductance."""

    if total_conductance_nS <= 0:
        raise ValueError("a nonzero route port requires positive conductance")
    distal = cell_spec.compartment(DISTAL_COMPARTMENT)
    density = total_conductance_nS / (distal.lateral_area_cm2 * 1e6)
    return SynapticPortSpec(
        name=L5_SST_LIKE_PORT_NAME,
        record_id=L5_SST_LIKE_ROUTE_ID,
        compartment=DISTAL_COMPARTMENT,
        receptor=Receptor.GABA,
        reversal_mV=GABA_REVERSAL_MV,
        conductance_density_mS_cm2=density,
        rise_ms=GABA_RISE_MS,
        fall_ms=GABA_FALL_MS,
        normalization=biexponential_normalization(GABA_RISE_MS, GABA_FALL_MS),
        voltage_block=False,
    )


def make_l5_sst_like_population_factory(
    *,
    total_conductance_nS: float,
    base_factory: Callable[..., Any] = create_compartmental_hh_population,
) -> Callable[..., Any]:
    """Add the registered port only to the represented L5 pyramidal population."""

    if not math.isfinite(total_conductance_nS) or total_conductance_nS <= 0:
        raise ValueError("the nonzero population factory requires positive conductance")

    def factory(*args: Any, **kwargs: Any):
        if args:
            raise TypeError("the SST-like population factory accepts keyword arguments only")
        name = kwargs.get("name")
        params = kwargs.get("params")
        if not isinstance(name, str) or not isinstance(params, dict):
            raise TypeError("population name and parameter mapping are required")
        if name.endswith(L5_EXCITATORY_SUFFIX):
            ports = params.get("synaptic_ports")
            cell_spec = params.get("cell_spec")
            if not isinstance(ports, tuple) or cell_spec is None:
                raise TypeError("L5 parameters require ports and an explicit cell spec")
            if any(port.record_id == L5_SST_LIKE_ROUTE_ID for port in ports):
                raise RuntimeError("the SST-like port is already present")
            params = dict(params)
            params["synaptic_ports"] = ports + (
                l5_sst_like_port(
                    cell_spec=cell_spec,
                    total_conductance_nS=total_conductance_nS,
                ),
            )
            kwargs = dict(kwargs)
            kwargs["params"] = params
        return base_factory(**kwargs)

    return factory


def create_l5_sst_like_synapse(
    *,
    pre_group: Any,
    post_population: Any,
    port: SynapticPortSpec,
    delay_ms: float,
    brian: Any,
    name: str,
) -> Any:
    """Create the registered route with SMART's last-two-arrival union gate."""

    validate_route_parameters(total_conductance_nS=1.0, delay_ms=delay_ms)
    if port.record_id != L5_SST_LIKE_ROUTE_ID:
        raise ValueError("the SST-like connector requires its registered port")
    last_wave = (
        f"last_amplitude*{port.normalization}"
        f"*(exp(-clip(last_elapsed/({port.fall_ms}*ms), 0, 100))"
        f"-exp(-clip(last_elapsed/({port.rise_ms}*ms), 0, 100)))"
    )
    previous_wave = (
        f"previous_amplitude*{port.normalization}"
        f"*(exp(-clip(previous_elapsed/({port.fall_ms}*ms), 0, 100))"
        f"-exp(-clip(previous_elapsed/({port.rise_ms}*ms), 0, 100)))"
    )
    model = (
        "w : 1 (constant)\n"
        "delivered : integer\n"
        "last_arrival : second\n"
        "previous_arrival : second\n"
        "last_amplitude : 1\n"
        "previous_amplitude : 1\n"
        "last_elapsed=t-last_arrival : second\n"
        "previous_elapsed=t-previous_arrival : second\n"
        f"last_wave={last_wave} : 1\n"
        f"previous_wave={previous_wave} : 1\n"
        "pre_signal=last_wave+previous_wave-last_wave*previous_wave : 1\n"
        f"{port.name}_gate_post=w*pre_signal : 1 (summed)"
    )
    update = (
        "previous_arrival = last_arrival\n"
        "previous_amplitude = last_amplitude\n"
        "last_arrival = t\n"
        "last_amplitude = 1\n"
        "delivered += 1"
    )
    synapse = brian.Synapses(
        pre_group,
        post_population.group,
        model=model,
        on_pre=update,
        method="rk4",
        name=name,
    )
    size = int(pre_group.N)
    if size != int(post_population.group.N):
        raise ValueError("the SST-like route requires equal-size same-index populations")
    indices = np.arange(size, dtype=int)
    synapse.connect(i=indices, j=indices)
    synapse.w = 1.0
    synapse.delay = delay_ms * brian.ms
    synapse.last_arrival = 0 * brian.second
    synapse.previous_arrival = 0 * brian.second
    synapse.last_amplitude = 0
    synapse.previous_amplitude = 0
    synapse.delivered = 0
    return synapse


def connect_l5_sst_like_feedback(
    sector: Any,
    *,
    total_conductance_nS: float,
    delay_ms: float,
    brian: Any,
) -> Any:
    """Add the registered same-index collapsed feedback route to a built sector."""

    validate_route_parameters(
        total_conductance_nS=total_conductance_nS, delay_ms=delay_ms
    )
    if total_conductance_nS == 0:
        return sector
    if L5_SST_LIKE_ROUTE_ID in sector.projections:
        raise RuntimeError("the SST-like route is already connected")
    population = sector.populations["layer5_excitatory_v1"]
    port = next(
        (
            candidate
            for candidate in population.compiled.synaptic_ports
            if candidate.record_id == L5_SST_LIKE_ROUTE_ID
        ),
        None,
    )
    if port is None:
        raise RuntimeError("the L5 population is missing its registered SST-like port")
    synapse = create_l5_sst_like_synapse(
        pre_group=population.group,
        post_population=population,
        port=port,
        delay_ms=delay_ms,
        brian=brian,
        name="post2008_l5_sst_like_same_index_feedback",
    )
    sector.projections[L5_SST_LIKE_ROUTE_ID] = synapse
    sector.network.add(synapse)
    return sector


def make_l5_sst_like_sector_builder(
    *,
    base_builder: Callable[..., Any],
    total_conductance_nS: float,
    delay_ms: float,
    base_population_factory: Callable[..., Any] = create_compartmental_hh_population,
) -> Callable[..., Any]:
    """Wrap a sector builder while preserving an exact no-op at zero resource."""

    validate_route_parameters(
        total_conductance_nS=total_conductance_nS, delay_ms=delay_ms
    )
    if total_conductance_nS == 0:

        def exact_null_builder(*args: Any, **kwargs: Any):
            return base_builder(*args, **kwargs)

        return exact_null_builder

    population_factory = make_l5_sst_like_population_factory(
        total_conductance_nS=total_conductance_nS,
        base_factory=base_population_factory,
    )

    def builder(*args: Any, **kwargs: Any):
        if args:
            raise TypeError("the SST-like sector builder accepts keyword arguments only")
        if kwargs.get("population_factory") is not None:
            raise ValueError("nested population-factory transformation is forbidden")
        brian = kwargs.get("brian")
        if brian is None:
            import brian2 as brian
        kwargs = dict(kwargs)
        kwargs["population_factory"] = population_factory
        kwargs["brian"] = brian
        sector = base_builder(**kwargs)
        return connect_l5_sst_like_feedback(
            sector,
            total_conductance_nS=total_conductance_nS,
            delay_ms=delay_ms,
            brian=brian,
        )

    return builder
