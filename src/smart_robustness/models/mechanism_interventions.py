"""Registered intrinsic-mechanism interventions for SMART robustness tests."""

from __future__ import annotations

from typing import Any

from .adapter import SmartPopulationAdapter, validate_smart_population_adapter
from .compartmental_hh import create_compartmental_hh_population

LAYER5_CELL_CLASS = "layer5_excitatory_v1"
LAYER5_DISTAL_NAK_DISABLED = frozenset({"distal_dendrite"})


def create_layer5_distal_nak_disabled_population(
    *,
    name: str,
    size: int,
    params: dict[str, Any],
    brian=None,
) -> SmartPopulationAdapter:
    """Remove only layer-5 distal fast Na/K while retaining the compartment."""

    cell_class = str(params.get("cell_class", ""))
    if cell_class != LAYER5_CELL_CLASS:
        raise ValueError(
            "the layer-5 distal Na/K intervention applies only to "
            f"{LAYER5_CELL_CLASS}, got {cell_class!r}"
        )
    if "disabled_nak_compartments" in params:
        raise ValueError("disabled_nak_compartments is controlled by the intervention")
    intervention_params = dict(params)
    intervention_params["disabled_nak_compartments"] = LAYER5_DISTAL_NAK_DISABLED
    population = create_compartmental_hh_population(
        name=name,
        size=size,
        params=intervention_params,
        brian=brian,
    )
    validate_smart_population_adapter(population)
    return population

