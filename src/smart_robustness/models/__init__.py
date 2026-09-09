from .integration import register_sanndra_scalar_rk4
from .registry import available_models, create_population
from .table3 import TABLE3_CELLS, CellSpec, CompartmentSpec, get_cell_spec

__all__ = [
    "TABLE3_CELLS",
    "CellSpec",
    "CompartmentSpec",
    "available_models",
    "create_population",
    "get_cell_spec",
]

# Keep the historical integration variant opt-in by name while making it
# available to every model factory without modifying the audited cell runtime.
register_sanndra_scalar_rk4()
