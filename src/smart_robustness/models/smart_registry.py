"""Named factories for complete, morphology-preserving SMART populations."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .adapter import SmartPopulationAdapter

SmartPopulationFactory = Callable[..., "SmartPopulationAdapter"]


def _classic_factory(**kwargs: Any) -> SmartPopulationAdapter:
    from .compartmental_hh import create_compartmental_hh_population

    return create_compartmental_hh_population(**kwargs)


def _literature_adex_factory(**kwargs: Any) -> SmartPopulationAdapter:
    from .adex import create_somatic_adex_population

    return create_somatic_adex_population(**kwargs)

_SMART_FACTORIES: dict[str, SmartPopulationFactory] = {
    "classic_multicompartment_hh": _classic_factory,
    "somatic_adex_literature": _literature_adex_factory,
}


def available_smart_models() -> tuple[str, ...]:
    return tuple(sorted(_SMART_FACTORIES))


def get_smart_population_factory(model_name: str) -> SmartPopulationFactory:
    try:
        return _SMART_FACTORIES[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown complete SMART model {model_name!r}; "
            f"available: {available_smart_models()}"
        ) from exc


def create_smart_population(model_name: str, **kwargs: Any) -> SmartPopulationAdapter:
    return get_smart_population_factory(model_name)(**kwargs)
