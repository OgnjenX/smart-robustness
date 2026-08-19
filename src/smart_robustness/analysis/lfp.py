"""SMART local-field-potential and current-source-density analysis."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from smart_robustness.models.table3 import CellSpec

SMART_EXTRACELLULAR_CONDUCTIVITY_MS_CM = 15.0
SMART_CORTICAL_DEPTH_UM = 1200.0

# Absolute compartment-centre depths measured from the inferior cortical
# border in Supplementary Figure 18. Names match Table 3's cell library.
FIGURE18_COMPARTMENT_DEPTH_UM: dict[str, dict[str, float]] = {
    "layer23_inhibitory": {"soma": 1125.0, "proximal_dendrite": 1175.0},
    "layer23_excitatory": {"soma": 850.0, "proximal_dendrite": 1087.5},
    "layer4_excitatory": {"soma": 575.0, "proximal_dendrite": 702.5},
    "layer4_inhibitory": {"soma": 475.0, "proximal_dendrite": 525.0},
    "layer5_excitatory": {
        "soma": 150.0,
        "proximal_dendrite": 425.0,
        "distal_dendrite": 850.0,
    },
    "layer6ii_excitatory": {
        "soma": 50.0,
        "proximal_dendrite": 150.0,
        "distal_dendrite": 300.0,
    },
    "layer6i_excitatory": {"soma": 50.0, "proximal_dendrite": 150.0},
}


@dataclass(frozen=True)
class Figure16ElectrodeGeometry:
    """One reproducible realization of the Methods 4.11 electrode geometry.

    The paper reports placement distributions rather than the random draws used
    for Figure 16.  Consequently, ``seed`` and ``fingerprint`` are part of the
    result and must accompany derived LFPs.  Compartments are flattened in
    cell-major Table 3 order.
    """

    seed: int
    selected_cell_index: int
    tip_depth_um: np.ndarray
    compartment_depth_um: np.ndarray
    cell_lateral_distance_um: np.ndarray
    distance_um: np.ndarray
    compartment_labels: tuple[tuple[int, str], ...]
    fingerprint: str

    @property
    def tip_spacing_um(self) -> float:
        return float(self.tip_depth_um[1] - self.tip_depth_um[0])


@dataclass(frozen=True)
class Figure16PopulationField:
    """LFP/CSD output paired with the exact geometry that generated it."""

    geometry: Figure16ElectrodeGeometry
    transmembrane_current_pA: np.ndarray
    potential_uV: np.ndarray
    current_source_density_uV_per_um: np.ndarray


@dataclass(frozen=True)
class Figure16CorticalField:
    """Whole-cortex LFP/CSD and the two Figure 16 depth regions."""

    seed: int
    population_fields: tuple[tuple[str, Figure16PopulationField], ...]
    potential_uV: np.ndarray
    current_source_density_uV_per_um: np.ndarray
    inferior_300um_tip_depth_um: np.ndarray
    inferior_300um_potential_uV: np.ndarray
    superior_300um_tip_depth_um: np.ndarray
    superior_300um_potential_uV: np.ndarray
    fingerprint: str


def figure16_electrode_geometry(
    cell: CellSpec,
    population_size: int,
    *,
    cortical_class: str | None = None,
    selected_cell_index: int,
    seed: int,
    tip_count: int = 54,
) -> Figure16ElectrodeGeometry:
    """Construct the stochastic 54-tip geometry specified in Methods 4.11.

    The first and last tips lie at the inferior and superior boundaries of the
    1.2-mm cortical sheet, with all other tips evenly spaced. Figure 18 fixes
    each cortical compartment's absolute centre depth. One selected cell
    receives a uniformly sampled lateral distance
    of 10--200 um; every other cell receives 10--1000 um.  Euclidean distance
    to each compartment centre supplies ``r_l`` in Equation 31.

    One lateral coordinate per cell is an explicit reconstruction assumption:
    the source states the distributions and orientation but does not preserve
    the realized three-dimensional layout.
    """

    if isinstance(population_size, bool) or not isinstance(population_size, int):
        raise TypeError("population_size must be an integer")
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if isinstance(selected_cell_index, bool) or not isinstance(selected_cell_index, int):
        raise TypeError("selected_cell_index must be an integer")
    if not 0 <= selected_cell_index < population_size:
        raise ValueError("selected_cell_index must identify a population cell")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(tip_count, bool) or not isinstance(tip_count, int):
        raise TypeError("tip_count must be an integer")
    if tip_count < 2:
        raise ValueError("tip_count must be at least two")
    if not cell.compartments:
        raise ValueError("cell must contain at least one compartment")

    resolved_class = cortical_class or cell.name
    try:
        source_depths = FIGURE18_COMPARTMENT_DEPTH_UM[resolved_class]
    except KeyError as error:
        raise ValueError(
            f"Figure 18 has no cortical geometry for class {resolved_class!r}"
        ) from error
    expected_names = {part.name for part in cell.compartments}
    if set(source_depths) != expected_names:
        raise ValueError("Figure 18 depths do not match the cell's compartments")
    local_centres_um = np.asarray([source_depths[part.name] for part in cell.compartments])
    tip_depth_um = np.linspace(0.0, SMART_CORTICAL_DEPTH_UM, tip_count)

    rng = np.random.default_rng(seed)
    lateral_um = rng.uniform(10.0, 1000.0, size=population_size)
    lateral_um[selected_cell_index] = rng.uniform(10.0, 200.0)
    compartment_depth_um = np.tile(local_centres_um, population_size)
    compartment_lateral_um = np.repeat(lateral_um, len(cell.compartments))
    distance_um = np.hypot(
        tip_depth_um[:, None] - compartment_depth_um[None, :],
        compartment_lateral_um[None, :],
    )
    labels = tuple(
        (cell_index, part.name)
        for cell_index in range(population_size)
        for part in cell.compartments
    )

    payload = {
        "algorithm": "numpy.default_rng.uniform-v1",
        "cell": cell.name,
        "cortical_class": resolved_class,
        "compartment_depth_um": dict(source_depths),
        "cortical_depth_um": SMART_CORTICAL_DEPTH_UM,
        "population_size": population_size,
        "selected_cell_index": selected_cell_index,
        "seed": seed,
        "tip_count": tip_count,
        "tip_depth_um": tip_depth_um.tolist(),
        "cell_lateral_distance_um": lateral_um.tolist(),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    for array in (tip_depth_um, compartment_depth_um, lateral_um, distance_um):
        array.setflags(write=False)
    return Figure16ElectrodeGeometry(
        seed=seed,
        selected_cell_index=selected_cell_index,
        tip_depth_um=tip_depth_um,
        compartment_depth_um=compartment_depth_um,
        cell_lateral_distance_um=lateral_um,
        distance_um=distance_um,
        compartment_labels=labels,
        fingerprint=fingerprint,
    )


def figure16_population_field(
    cell: CellSpec,
    compartment_currents_pA: dict[str, np.ndarray],
    *,
    cortical_class: str | None = None,
    selected_cell_index: int,
    seed: int,
    conductivity_mS_cm: float = SMART_EXTRACELLULAR_CONDUCTIVITY_MS_CM,
) -> Figure16PopulationField:
    """Convert monitored compartment currents into Methods 4.11 LFP and CSD.

    Each mapping value must have shape ``(cell, time)``. The function performs
    the otherwise error-prone conversion to the cell-major compartment order
    used by :func:`figure16_electrode_geometry`.
    """

    expected = tuple(part.name for part in cell.compartments)
    if set(compartment_currents_pA) != set(expected):
        missing = sorted(set(expected) - set(compartment_currents_pA))
        extra = sorted(set(compartment_currents_pA) - set(expected))
        raise ValueError(f"compartment currents do not match cell: missing={missing}, extra={extra}")
    arrays = [np.asarray(compartment_currents_pA[name], dtype=float) for name in expected]
    shape = arrays[0].shape
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("each compartment current must have nonempty (cell, time) shape")
    if any(array.shape != shape for array in arrays):
        raise ValueError("all compartment-current arrays must have identical shape")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("compartment currents must be finite")
    # (compartment, cell, time) -> (cell, compartment, time) -> (source, time)
    currents = np.stack(arrays, axis=0).transpose(1, 0, 2).reshape(-1, shape[1])
    geometry = figure16_electrode_geometry(
        cell,
        shape[0],
        cortical_class=cortical_class,
        selected_cell_index=selected_cell_index,
        seed=seed,
    )
    potential = extracellular_potential_uV(
        currents, geometry.distance_um, conductivity_mS_cm=conductivity_mS_cm
    )
    csd = current_source_density_uV_per_um(potential, geometry.tip_spacing_um)
    for array in (currents, potential, csd):
        array.setflags(write=False)
    return Figure16PopulationField(geometry, currents, potential, csd)


def figure16_cortical_field(
    populations: dict[str, tuple[CellSpec, dict[str, np.ndarray], int]],
    *,
    seed: int,
    conductivity_mS_cm: float = SMART_EXTRACELLULAR_CONDUCTIVITY_MS_CM,
) -> Figure16CorticalField:
    """Sum all cortical population fields and retain Figure 16 depth regions.

    Each population value is ``(cell_spec, compartment_currents, selected_cell_index)``.
    A master seed deterministically allocates independent geometry seeds in
    sorted population-name order. The caption identifies the lower and upper
    0.3 mm regions but does not specify a tip-reduction rule, so all regional
    tip traces are retained for a separately declared analysis convention.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not populations:
        raise ValueError("at least one cortical population is required")
    rng = np.random.default_rng(seed)
    fields: list[tuple[str, Figure16PopulationField]] = []
    time_count: int | None = None
    for name in sorted(populations):
        try:
            cell, currents, selected_index = populations[name]
        except (TypeError, ValueError) as error:
            raise ValueError(
                "each population must contain cell, currents, and selected index"
            ) from error
        sub_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        cortical_class = name.removesuffix("_v1").removesuffix("_v2")
        field = figure16_population_field(
            cell,
            currents,
            cortical_class=cortical_class,
            selected_cell_index=selected_index,
            seed=sub_seed,
            conductivity_mS_cm=conductivity_mS_cm,
        )
        if time_count is None:
            time_count = field.potential_uV.shape[1]
        elif field.potential_uV.shape[1] != time_count:
            raise ValueError("all cortical populations must have the same time axis")
        fields.append((name, field))

    potential = np.sum([field.potential_uV for _, field in fields], axis=0)
    tip_depth_um = fields[0][1].geometry.tip_depth_um
    inferior = tip_depth_um <= 300.0
    superior = tip_depth_um >= SMART_CORTICAL_DEPTH_UM - 300.0
    csd = current_source_density_uV_per_um(
        potential, fields[0][1].geometry.tip_spacing_um
    )
    payload = {
        "seed": seed,
        "populations": [
            {"name": name, "geometry_fingerprint": field.geometry.fingerprint}
            for name, field in fields
        ],
        "region_um": 300.0,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    arrays = (
        potential,
        csd,
        tip_depth_um[inferior],
        potential[inferior],
        tip_depth_um[superior],
        potential[superior],
    )
    for array in arrays:
        array.setflags(write=False)
    return Figure16CorticalField(
        seed=seed,
        population_fields=tuple(fields),
        potential_uV=potential,
        current_source_density_uV_per_um=csd,
        inferior_300um_tip_depth_um=arrays[2],
        inferior_300um_potential_uV=arrays[3],
        superior_300um_tip_depth_um=arrays[4],
        superior_300um_potential_uV=arrays[5],
        fingerprint=fingerprint,
    )


def extracellular_potential_uV(
    transmembrane_current_pA: np.ndarray,
    distance_um: np.ndarray,
    *,
    conductivity_mS_cm: float = SMART_EXTRACELLULAR_CONDUCTIVITY_MS_CM,
) -> np.ndarray:
    """Evaluate Grossberg and Versace Equation 31 at each electrode tip.

    Currents are shaped ``(compartment, time)`` and use Equation 32's sign:
    positive values equal axial current entering the compartment. An electrode
    geometry determines whether each signed contribution is a source or sink.
    Distances are shaped ``(tip, compartment)``. The result is ``(tip, time)``
    in microvolts. The conversion is direct because pA/um equals microvolts
    times S/m, and 1 mS/cm equals 0.1 S/m.
    """

    currents = np.asarray(transmembrane_current_pA, dtype=float)
    distances = np.asarray(distance_um, dtype=float)
    if currents.ndim != 2 or distances.ndim != 2:
        raise ValueError("currents and distances must both be two-dimensional")
    if distances.shape[1] != currents.shape[0]:
        raise ValueError("distance compartments must match current compartments")
    if not np.all(np.isfinite(currents)) or not np.all(np.isfinite(distances)):
        raise ValueError("currents and distances must be finite")
    if np.any(distances <= 0):
        raise ValueError("electrode distances must be positive")
    if not np.isfinite(conductivity_mS_cm) or conductivity_mS_cm <= 0:
        raise ValueError("extracellular conductivity must be finite and positive")
    conductivity_s_m = conductivity_mS_cm * 0.1
    return (1.0 / (4.0 * np.pi * conductivity_s_m)) * (1.0 / distances) @ currents


def current_source_density_uV_per_um(
    electrode_potential_uV: np.ndarray, tip_spacing_um: float
) -> np.ndarray:
    """Apply paper-literal Equation 33, whose denominator is ``Delta x``.

    End tips do not have two neighbors and are therefore omitted. Input shape
    is ``(tip, time)`` and output shape is ``(tip - 2, time)`` in uV/um.
    Although the surrounding prose calls this a second derivative, the printed
    equation has no square on its denominator; classic SMART preserves that
    source fact rather than silently correcting it.
    """

    potential = np.asarray(electrode_potential_uV, dtype=float)
    if potential.ndim != 2 or potential.shape[0] < 3:
        raise ValueError("potential must contain at least three electrode-tip traces")
    if not np.all(np.isfinite(potential)):
        raise ValueError("electrode potential must be finite")
    if not np.isfinite(tip_spacing_um) or tip_spacing_um <= 0:
        raise ValueError("tip spacing must be finite and positive")
    return (potential[:-2] - 2.0 * potential[1:-1] + potential[2:]) / tip_spacing_um


def standard_current_source_density_uV_per_um2(
    electrode_potential_uV: np.ndarray, tip_spacing_um: float
) -> np.ndarray:
    """Return the conventional centered second derivative using ``Delta x`` squared.

    This alternate is useful for robustness comparisons but is not the printed
    Grossberg--Versace Equation 33.
    """

    literal = current_source_density_uV_per_um(electrode_potential_uV, tip_spacing_um)
    return literal / tip_spacing_um
