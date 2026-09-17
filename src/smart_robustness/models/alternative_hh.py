"""Morphology-preserving Pospischil-type somatic HH substitution."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .adapter import SmartPopulationAdapter, validate_smart_population_adapter
from .alternative_hh_parameters import (
    POSPISCHIL_FS_MEAN,
    POSPISCHIL_RS_MEAN,
    AlternativeHHParameters,
)
from .compartmental_hh import create_compartmental_hh_population


def _literature_parameters(cell_class: str) -> AlternativeHHParameters:
    return POSPISCHIL_FS_MEAN if "inhibitory" in cell_class else POSPISCHIL_RS_MEAN


def create_somatic_alternative_hh_population(
    *,
    name: str,
    size: int,
    params: dict[str, Any],
    brian=None,
    alternative_hh_parameters: AlternativeHHParameters | None = None,
) -> SmartPopulationAdapter:
    """Replace somatic SMART Na/K with a Pospischil-type conductance model."""

    selected = alternative_hh_parameters or _literature_parameters(
        str(params.get("cell_class", ""))
    )
    if not isinstance(selected, AlternativeHHParameters):
        raise TypeError("alternative_hh_parameters must be AlternativeHHParameters")
    model_params = dict(params)
    model_params["somatic_spike_model"] = "pospischil_hh"
    model_params["alternative_hh_parameters"] = selected.as_dict()
    population = create_compartmental_hh_population(
        name=name,
        size=size,
        params=model_params,
        brian=brian,
    )
    validate_smart_population_adapter(population)
    return population


def make_somatic_alternative_hh_factory(
    parameters: AlternativeHHParameters | Mapping[str, AlternativeHHParameters],
):
    """Bind one preset or a preregistered per-cell-class HH parameter map."""

    if isinstance(parameters, AlternativeHHParameters):
        default = parameters
        by_class: Mapping[str, AlternativeHHParameters] = {}
    elif isinstance(parameters, Mapping) and parameters:
        if not all(
            isinstance(value, AlternativeHHParameters) for value in parameters.values()
        ):
            raise TypeError("each alternative-HH value must be AlternativeHHParameters")
        default = next(iter(parameters.values()))
        by_class = dict(parameters)
    else:
        raise TypeError(
            "parameters must be AlternativeHHParameters or a non-empty mapping"
        )

    def factory(*, name: str, size: int, params: dict[str, Any], brian=None):
        cell_class = str(params.get("cell_class", ""))
        selected = by_class.get(cell_class, default)
        return create_somatic_alternative_hh_population(
            name=name,
            size=size,
            params=params,
            brian=brian,
            alternative_hh_parameters=selected,
        )

    return factory


def load_alternative_hh_parameter_map(
    path: str | Path,
) -> dict[str, AlternativeHHParameters]:
    """Load a frozen per-cell-class alternative-HH parameter manifest."""

    raw = yaml.safe_load(Path(path).read_text())
    fixed = raw.get("fixed_parameters", {})
    parameters = {
        str(name): AlternativeHHParameters.from_mapping(fixed | values)
        for name, values in raw.get("cell_classes", {}).items()
    }
    if not parameters:
        raise ValueError(f"alternative-HH parameter manifest {path} is empty")
    return parameters
