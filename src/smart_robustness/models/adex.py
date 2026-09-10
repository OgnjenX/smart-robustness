"""Morphology-preserving somatic AdEx substitution for complete SMART."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .adapter import SmartPopulationAdapter, validate_smart_population_adapter
from .adex_parameters import LITERATURE_REGULAR_SPIKING, AdExParameters
from .compartmental_hh import create_compartmental_hh_population


def create_somatic_adex_population(
    *,
    name: str,
    size: int,
    params: dict[str, Any],
    brian=None,
    adex_parameters: AdExParameters = LITERATURE_REGULAR_SPIKING,
) -> SmartPopulationAdapter:
    """Replace somatic Na/K spike generation while preserving SMART structure."""

    if not isinstance(adex_parameters, AdExParameters):
        raise TypeError("adex_parameters must be an AdExParameters instance")
    model_params = dict(params)
    model_params["somatic_spike_model"] = "adex"
    model_params["adex_parameters"] = adex_parameters.as_dict()
    population = create_compartmental_hh_population(
        name=name,
        size=size,
        params=model_params,
        brian=brian,
    )
    validate_smart_population_adapter(population)
    return population


def make_somatic_adex_factory(
    parameters: AdExParameters | Mapping[str, AdExParameters],
):
    """Bind one preset or a preregistered per-cell-class parameter mapping."""

    if isinstance(parameters, AdExParameters):
        default = parameters
        by_class: Mapping[str, AdExParameters] = {}
    elif isinstance(parameters, Mapping) and parameters:
        if not all(isinstance(value, AdExParameters) for value in parameters.values()):
            raise TypeError("each cell-class AdEx value must be AdExParameters")
        default = LITERATURE_REGULAR_SPIKING
        by_class = dict(parameters)
    else:
        raise TypeError("parameters must be an AdExParameters instance or non-empty mapping")

    def factory(*, name: str, size: int, params: dict[str, Any], brian=None):
        cell_class = str(params.get("cell_class", ""))
        selected = by_class.get(cell_class, default)
        return create_somatic_adex_population(
            name=name,
            size=size,
            params=params,
            brian=brian,
            adex_parameters=selected,
        )

    return factory


def load_adex_parameter_map(path: str | Path) -> dict[str, AdExParameters]:
    """Load a frozen full map or a manifest that overrides a frozen base."""

    manifest_path = Path(path)
    raw = yaml.safe_load(manifest_path.read_text())
    if "base_parameter_manifest" in raw:
        base_path = Path(raw["base_parameter_manifest"])
        if not base_path.is_absolute():
            base_path = Path.cwd() / base_path
        parameters = load_adex_parameter_map(base_path)
    else:
        parameters = {}
    fixed = raw.get("fixed_parameters", {})
    for name, values in raw.get("cell_classes", {}).items():
        parameters[name] = AdExParameters.from_mapping(values | fixed)
    for name, values in raw.get("overrides", {}).items():
        inherited = parameters[name].as_dict() if name in parameters else {}
        parameters[name] = AdExParameters.from_mapping(inherited | values | fixed)
    if not parameters:
        raise ValueError(f"AdEx parameter manifest {manifest_path} is empty")
    return parameters
