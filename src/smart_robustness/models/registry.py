from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .classic_hh import create_classic_hh_population

Factory = Callable[..., Any]

_FACTORIES: dict[str, Factory] = {"classic_hh": create_classic_hh_population}
_PLANNED = {"adex", "gif", "point_hh", "multicompartment_hh"}


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_FACTORIES))


def create_population(model_name: str, **kwargs: Any) -> Any:
    if model_name in _PLANNED:
        raise NotImplementedError(
            f"{model_name!r} is a planned robustness backend; it is not implemented as an alias."
        )
    try:
        factory = _FACTORIES[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model {model_name!r}; available: {available_models()}") from exc
    return factory(**kwargs)
