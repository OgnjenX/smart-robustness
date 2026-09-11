"""Population-selective somatic spike-generator substitutions for SMART."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .compartmental_hh import create_compartmental_hh_population

CORTICAL_EXCITATORY_CELL_CLASSES = frozenset(
    {
        "layer4_excitatory_v1",
        "layer23_excitatory_v1",
        "layer5_excitatory_v1",
        "layer6i_excitatory_v1",
        "layer6ii_excitatory_v1",
    }
)
CORTICAL_INHIBITORY_CELL_CLASSES = frozenset(
    {
        "layer4_inhibitory_v1",
        "layer23_inhibitory_v1",
    }
)
CORTICAL_CELL_CLASSES = (
    CORTICAL_EXCITATORY_CELL_CLASSES | CORTICAL_INHIBITORY_CELL_CLASSES
)
THALAMIC_RELAY_TRN_CELL_CLASSES = frozenset({"thalamic_relay", "trn"})


def make_selective_population_factory(
    alternative_factory: Callable[..., Any],
    alternative_cell_classes: Iterable[str],
    *,
    control_factory: Callable[..., Any] = create_compartmental_hh_population,
):
    """Use an alternative factory only for the registered cell classes.

    This keeps the circuit assembly path identical between arms and changes
    only the somatic spike generator selected inside each target population.
    """

    if not callable(alternative_factory):
        raise TypeError("alternative_factory must be callable")
    if not callable(control_factory):
        raise TypeError("control_factory must be callable")
    selected = frozenset(alternative_cell_classes)
    if not selected or any(not isinstance(name, str) or not name for name in selected):
        raise ValueError("alternative_cell_classes must contain non-empty names")

    def factory(*, name: str, size: int, params: dict[str, Any], brian=None):
        cell_class = str(params.get("cell_class", ""))
        implementation = alternative_factory if cell_class in selected else control_factory
        return implementation(name=name, size=size, params=params, brian=brian)

    factory.alternative_cell_classes = selected
    return factory
