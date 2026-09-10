"""Morphology-preserving somatic GIF substitution for complete SMART."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .adapter import SmartPopulationAdapter, validate_smart_population_adapter
from .compartmental_hh import create_compartmental_hh_population
from .gif_parameters import GIFParameters, literature_gif_parameters


def create_somatic_gif_population(
    *,
    name: str,
    size: int,
    params: dict[str, Any],
    brian=None,
    gif_parameters: GIFParameters | None = None,
) -> SmartPopulationAdapter:
    """Replace somatic Na/K spike generation while preserving SMART structure."""

    selected = gif_parameters or literature_gif_parameters(str(params.get("cell_class", "")))
    if not isinstance(selected, GIFParameters):
        raise TypeError("gif_parameters must be a GIFParameters instance")
    model_params = dict(params)
    model_params["somatic_spike_model"] = "gif"
    model_params["gif_parameters"] = selected.as_dict()
    population = create_compartmental_hh_population(
        name=name,
        size=size,
        params=model_params,
        brian=brian,
    )
    validate_smart_population_adapter(population)
    return population


def make_somatic_gif_factory(parameters: GIFParameters | Mapping[str, GIFParameters]):
    """Bind one preset or a preregistered per-cell-class parameter mapping."""

    if isinstance(parameters, GIFParameters):
        default = parameters
        by_class: Mapping[str, GIFParameters] = {}
    elif isinstance(parameters, Mapping) and parameters:
        if not all(isinstance(value, GIFParameters) for value in parameters.values()):
            raise TypeError("each cell-class GIF value must be GIFParameters")
        default = next(iter(parameters.values()))
        by_class = dict(parameters)
    else:
        raise TypeError("parameters must be a GIFParameters instance or non-empty mapping")

    def factory(*, name: str, size: int, params: dict[str, Any], brian=None):
        cell_class = str(params.get("cell_class", ""))
        selected = by_class.get(cell_class, default)
        return create_somatic_gif_population(
            name=name,
            size=size,
            params=params,
            brian=brian,
            gif_parameters=selected,
        )

    return factory


def load_gif_parameter_map(path: str | Path) -> dict[str, GIFParameters]:
    """Load a frozen full map or a manifest that overrides a frozen base."""

    manifest_path = Path(path)
    raw = yaml.safe_load(manifest_path.read_text())
    if "base_parameter_manifest" in raw:
        base_path = Path(raw["base_parameter_manifest"])
        if not base_path.is_absolute():
            base_path = Path.cwd() / base_path
        parameters = load_gif_parameter_map(base_path)
    else:
        parameters = {}
    fixed = raw.get("fixed_parameters", {})
    for name, values in raw.get("cell_classes", {}).items():
        parameters[name] = GIFParameters.from_mapping(values | fixed)
    for name, values in raw.get("overrides", {}).items():
        inherited = parameters[name].as_dict() if name in parameters else {}
        parameters[name] = GIFParameters.from_mapping(inherited | values | fixed)
    if not parameters:
        raise ValueError(f"GIF parameter manifest {manifest_path} is empty")
    return parameters
