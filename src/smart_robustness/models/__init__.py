from .integration import register_sanndra_scalar_rk4
from .registry import available_models, create_population
from .smart_registry import (
    available_smart_models,
    create_smart_population,
    get_smart_population_factory,
)
from .table3 import TABLE3_CELLS, CellSpec, CompartmentSpec, get_cell_spec

__all__ = [
    "TABLE3_CELLS",
    "CellSpec",
    "CompartmentSpec",
    "available_models",
    "available_smart_models",
    "create_population",
    "create_smart_population",
    "get_cell_spec",
    "get_smart_population_factory",
]

# Keep the historical integration variant opt-in by name while making it
# available to every model factory without modifying the audited cell runtime.
register_sanndra_scalar_rk4()
