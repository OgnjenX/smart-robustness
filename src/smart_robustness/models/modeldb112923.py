"""Facts recovered from the official ModelDB 112923 SMART backup.

The raw archive is copyright-marked and has no explicit redistribution
license, so it is not vendored. This module records model facts and hashes;
the original files remain downloadable from ModelDB.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .table3 import CellSpec, CompartmentSpec

ARCHIVE_URL = "https://modeldb.science/download/112923"
ARCHIVE_SHA256 = "5fc6e0042ef093665ad2f1c24c87f9cc0d796629c896ee8a4efe34827a4462c5"
SMART_NML_SHA256 = "16fc196f2ed68d1c02a39395fa83bb0ffab8fcf8edffe6fd11022bb6c6e299e1"
CA_REBOUND_SHA256 = "2fc9615bf119e7602e75c00b5cd29e74dfaf89ba96d8eb6b0f17e36932f1ad0d"
AHP_ACH_SHA256 = "682355d883b98f727087c5f959f2c776d9e47edfd6ff5896d1cf394d283ff560"


@dataclass(frozen=True, slots=True)
class Figure8SourceFacts:
    pulse_pA: float = 300.0
    e_k_mV: float = -90.0
    e_na_mV: float = 50.0
    e_ca_mV: float = 180.0
    calcium_density_mS_cm2: float = 250.0
    missing_leak_density: bool = True


FIGURE8_SOURCE_FACTS = Figure8SourceFacts()


class Figure8GeometryConvention(StrEnum):
    """Unit interpretation for the legacy version-1 Ca_rebound.xml geometry."""

    CENTIMETERS = "centimeters"
    MILLIMETERS = "millimeters"

# ``SMART.nml`` population order and compartment counts. The first ten sheets
# are externally sized (9x9 in the paper protocol); the final two are fixed 1x1.
FIRST_ORDER_POPULATIONS: tuple[tuple[str, int, tuple[int, int]], ...] = (
    ("Relay", 3, (9, 9)),
    ("Reticular", 3, (9, 9)),
    ("Layer_5", 3, (9, 9)),
    ("Layer_6_II", 3, (9, 9)),
    ("Layer_6_I", 2, (9, 9)),
    ("Layer_4_INT", 2, (9, 9)),
    ("Layer_2_3", 2, (9, 9)),
    ("Layer_4", 2, (9, 9)),
    ("Layer_2_3_INT", 2, (9, 9)),
    ("Relay_INT", 2, (9, 9)),
    ("INTRALAMINAR", 3, (1, 1)),
    ("Thalamic_MATRIX", 3, (1, 1)),
)
FIRST_ORDER_PROJECTION_COUNT = 56

SECOND_ORDER_POPULATIONS: tuple[tuple[str, int, tuple[int, int]], ...] = (
    ("Reticular_V2", 3, (9, 9)),
    ("Relay_V2", 3, (9, 9)),
    ("Layer_6_II_V2", 3, (9, 9)),
    ("Layer_5_V2", 3, (9, 9)),
    ("Layer_6_I_V2", 2, (9, 9)),
    ("Layer_2_3_V2", 2, (9, 9)),
    ("Layer_4_V2", 2, (9, 9)),
    ("Layer_2_3_INT_V2", 2, (9, 9)),
    ("Relay_INT_V2", 2, (9, 9)),
    ("INTRALAMINAR_V2", 3, (1, 1)),
    ("Thalamic_MATRIX_V2", 3, (1, 1)),
    ("Layer_4_INT_V2", 2, (9, 9)),
)


@dataclass(frozen=True, slots=True)
class FirstOrderPopulationFacts:
    source_name: str
    canonical_name: str
    cell: CellSpec
    shape: tuple[int, int]
    e_na_mV: float = 50.0
    e_k_mV: float = -90.0
    e_ca_mV: float = 180.0
    calcium_gate_convention: str | None = None
    ahp_density_mS_cm2: float | None = None
    ahp_reversal_mV: float | None = None
    ahp_rise_ms: float | None = None
    ahp_fall_ms: float | None = None
    depletion_epsilon: float | None = None
    depletion_recovery_ms: float | None = None


def _source_compartment(
    name: str,
    diameter: float,
    length: float,
    axial: float,
    leak_reversal: float,
    leak_density: float,
    sodium: float | None = None,
    potassium: float | None = None,
    calcium: float | None = None,
) -> CompartmentSpec:
    # KInNeSS serializes compartment dimensions in centimetres.  Internally,
    # CompartmentSpec uses millimetres to match Grossberg & Versace Table 3.
    cm_to_mm = 10.0
    return CompartmentSpec(
        name,
        diameter * cm_to_mm,
        length * cm_to_mm,
        axial,
        leak_reversal,
        leak_density,
        sodium,
        potassium,
        calcium,
    )


def _source_cell(name: str, *compartments: CompartmentSpec) -> CellSpec:
    return CellSpec(f"modeldb112923_{name}", tuple(compartments))


def first_order_population_facts() -> tuple[FirstOrderPopulationFacts, ...]:
    """Intrinsic facts serialized in the first area of ``SMART.nml``.

    Root-soma ``inpResistance`` is omitted by KInNeSS. Its placeholder repeats
    the first child value and is ignored by ``kinness_serialized_edge``.
    """

    c = _source_compartment
    return (
        FirstOrderPopulationFacts(
            "Relay",
            "thalamic_relay",
            _source_cell(
                "relay",
                c("soma", 0.005, 0.006, 8, -60, 0.01, 100, 100, 0.1),
                c("proximal_dendrite", 0.0005, 0.008, 8, -60, 0.01, calcium=0.1),
                c("distal_dendrite", 0.0004, 0.008, 8.2, -60, 0.1, calcium=0.1),
            ),
            (9, 9),
            e_k_mV=-100,
        ),
        FirstOrderPopulationFacts(
            "Reticular",
            "trn",
            _source_cell(
                "reticular",
                c("soma", 0.005, 0.005, 10, -69, 0.1, 100, 80, 100),
                c("proximal_dendrite", 0.001, 0.005, 10, -69, 0.1, calcium=100),
                c("distal_dendrite", 0.001, 0.005, 10, -69, 0.1, calcium=100),
            ),
            (9, 9),
            e_k_mV=-100,
            e_ca_mV=120,
            calcium_gate_convention="modeldb_reticular_112923",
        ),
        FirstOrderPopulationFacts(
            "Layer_5",
            "layer5_excitatory_v1",
            _source_cell(
                "layer5",
                c("soma", 0.01, 0.015, 6, -73, 0.1, 50, 30),
                c("proximal_dendrite", 0.006, 0.04, 6, -73, 0.03),
                c("distal_dendrite", 0.006, 0.05, 6, -73, 0.03, 50, 30),
            ),
            (9, 9),
            ahp_density_mS_cm2=0.4,
            ahp_reversal_mV=-70,
            ahp_rise_ms=5,
            ahp_fall_ms=20,
            depletion_epsilon=0.5,
            depletion_recovery_ms=100,
        ),
        FirstOrderPopulationFacts(
            "Layer_6_II",
            "layer6ii_excitatory_v1",
            _source_cell(
                "layer6ii",
                c("soma", 0.006, 0.01, 25, -64, 0.1, 50, 30),
                c("proximal_dendrite", 0.0008, 0.01, 25, -64, 0.1),
                c("distal_dendrite", 0.0008, 0.02, 25, -64, 0.01),
            ),
            (9, 9),
            ahp_density_mS_cm2=0.5,
            ahp_reversal_mV=-70,
            ahp_rise_ms=5,
            ahp_fall_ms=20,
            depletion_epsilon=0.5,
            depletion_recovery_ms=100,
        ),
        FirstOrderPopulationFacts(
            "Layer_6_I",
            "layer6i_excitatory_v1",
            _source_cell(
                "layer6i",
                c("soma", 0.008, 0.01, 80, -70, 0.15, 50, 30),
                c("proximal_dendrite", 0.005, 0.01, 80, -70, 0.9),
            ),
            (9, 9),
            depletion_epsilon=1.0,
            depletion_recovery_ms=400,
        ),
        FirstOrderPopulationFacts(
            "Layer_4_INT",
            "layer4_inhibitory_v1",
            _source_cell(
                "layer4_inhibitory",
                c("soma", 0.002, 0.002, 100, -50, 0.01, 50, 30),
                c("proximal_dendrite", 0.001, 0.005, 100, -50, 0.01),
            ),
            (9, 9),
        ),
        FirstOrderPopulationFacts(
            "Layer_2_3",
            "layer23_excitatory_v1",
            _source_cell(
                "layer23_excitatory",
                c("soma", 0.005, 0.005, 100, -65, 0.05, 50, 30),
                c("proximal_dendrite", 0.002, 0.0225, 100, -65, 0.05),
            ),
            (9, 9),
        ),
        FirstOrderPopulationFacts(
            "Layer_4",
            "layer4_excitatory_v1",
            _source_cell(
                "layer4_excitatory",
                c("soma", 0.005, 0.005, 40, -65, 0.01, 50, 30),
                c("proximal_dendrite", 0.001, 0.025, 40, -65, 0.01),
            ),
            (9, 9),
        ),
        FirstOrderPopulationFacts(
            "Layer_2_3_INT",
            "layer23_inhibitory_v1",
            _source_cell(
                "layer23_inhibitory",
                c("soma", 0.002, 0.002, 60, -49, 0.01, 50, 30),
                c("proximal_dendrite", 0.001, 0.005, 60, -49, 0.01),
            ),
            (9, 9),
        ),
        FirstOrderPopulationFacts(
            "Relay_INT",
            "thalamic_interneuron",
            _source_cell(
                "relay_interneuron",
                c("soma", 0.002, 0.002, 60, -49, 0.01, 50, 30),
                c("proximal_dendrite", 0.001, 0.01, 60, -49, 0.01),
            ),
            (9, 9),
        ),
        FirstOrderPopulationFacts(
            "INTRALAMINAR",
            "thalamic_nonspecific",
            _source_cell(
                "intralaminar",
                c("soma", 0.008, 0.008, 10, -64, 0.09, 50, 30, 0.1),
                c("proximal_dendrite", 0.0015, 0.01, 10, -64, 0.1, calcium=250),
                c("distal_dendrite", 0.0015, 0.01, 10, -64, 0.1, calcium=250),
            ),
            (1, 1),
        ),
        FirstOrderPopulationFacts(
            "Thalamic_MATRIX",
            "thalamic_matrix",
            _source_cell(
                "thalamic_matrix",
                c("soma", 0.008, 0.008, 10, -64, 0.09, 50, 30, 0.1),
                c("proximal_dendrite", 0.0015, 0.01, 10, -64, 0.1, calcium=250),
                c("distal_dendrite", 0.0015, 0.01, 10, -64, 0.1, calcium=250),
            ),
            (1, 1),
        ),
    )


def first_order_structural_counts() -> tuple[int, int]:
    """Return executable-source cell and compartment totals for V1."""

    cells = sum(width * height for _, _, (width, height) in FIRST_ORDER_POPULATIONS)
    compartments = sum(
        count * width * height for _, count, (width, height) in FIRST_ORDER_POPULATIONS
    )
    return cells, compartments


def second_order_population_facts() -> tuple[FirstOrderPopulationFacts, ...]:
    """Intrinsic facts serialized in the pulvinar/V2 half of ``SMART.nml``."""

    v1 = {fact.canonical_name: fact for fact in first_order_population_facts()}
    mapping = (
        ("Reticular_V2", "trn_v2", "trn"),
        ("Relay_V2", "thalamic_relay_v2", "thalamic_relay"),
        ("Layer_6_II_V2", "layer6ii_excitatory_v2", "layer6ii_excitatory_v1"),
        ("Layer_5_V2", "layer5_excitatory_v2", "layer5_excitatory_v1"),
        ("Layer_6_I_V2", "layer6i_excitatory_v2", "layer6i_excitatory_v1"),
        ("Layer_2_3_V2", "layer23_excitatory_v2", "layer23_excitatory_v1"),
        ("Layer_4_V2", "layer4_excitatory_v2", "layer4_excitatory_v1"),
        ("Layer_2_3_INT_V2", "layer23_inhibitory_v2", "layer23_inhibitory_v1"),
        ("Relay_INT_V2", "thalamic_interneuron_v2", "thalamic_interneuron"),
        ("INTRALAMINAR_V2", "thalamic_nonspecific_v2", "thalamic_nonspecific"),
        ("Thalamic_MATRIX_V2", "thalamic_matrix_v2", "thalamic_matrix"),
        ("Layer_4_INT_V2", "layer4_inhibitory_v2", "layer4_inhibitory_v1"),
    )
    results = []
    for source_name, canonical_name, template_name in mapping:
        template = v1[template_name]
        cell = template.cell
        if canonical_name == "layer5_excitatory_v2":
            c = _source_compartment
            cell = _source_cell(
                "layer5_v2",
                c("soma", 0.01, 0.015, 5, -72, 0.1, 50, 30),
                c("proximal_dendrite", 0.006, 0.04, 5, -72, 0.03),
                c("distal_dendrite", 0.006, 0.05, 5, -72, 0.03, 50, 30),
            )
        else:
            cell = CellSpec(
                template.cell.name.replace("modeldb112923_", "modeldb112923_v2_"),
                template.cell.compartments,
            )
        results.append(
            FirstOrderPopulationFacts(
                source_name=source_name,
                canonical_name=canonical_name,
                cell=cell,
                shape=template.shape,
                e_na_mV=template.e_na_mV,
                e_k_mV=template.e_k_mV,
                e_ca_mV=template.e_ca_mV,
                calcium_gate_convention=template.calcium_gate_convention,
                ahp_density_mS_cm2=template.ahp_density_mS_cm2,
                ahp_reversal_mV=template.ahp_reversal_mV,
                ahp_rise_ms=template.ahp_rise_ms,
                ahp_fall_ms=template.ahp_fall_ms,
                depletion_epsilon=template.depletion_epsilon,
                depletion_recovery_ms=template.depletion_recovery_ms,
            )
        )
    return tuple(results)


def full_network_structural_counts() -> tuple[int, int]:
    facts = first_order_population_facts() + second_order_population_facts()
    return (
        sum(f.shape[0] * f.shape[1] for f in facts),
        sum(f.shape[0] * f.shape[1] * len(f.cell.compartments) for f in facts),
    )


def figure8_relay_spec(
    *,
    leak_density_mS_cm2: float,
    geometry_convention: Figure8GeometryConvention | str = Figure8GeometryConvention.CENTIMETERS,
) -> CellSpec:
    """Build the dedicated ``Ca_rebound.xml`` relay cell.

    ``Ca_rebound.xml`` does not serialize a leak density for this cell. The
    caller must therefore provide the KInNeSS default or a declared calibrated
    candidate; no value is silently inherited from Table 3.
    """

    if leak_density_mS_cm2 <= 0:
        raise ValueError("leak_density_mS_cm2 must be an explicit positive candidate")

    geometry = Figure8GeometryConvention(geometry_convention)

    def compartment(
        name: str,
        source_diameter: float,
        source_length: float,
        axial_kohm_cm: float,
        *,
        sodium: float | None = None,
        potassium: float | None = None,
    ) -> CompartmentSpec:
        source_to_mm = 10.0 if geometry is Figure8GeometryConvention.CENTIMETERS else 1.0
        return CompartmentSpec(
            name=name,
            diameter_mm=source_diameter * source_to_mm,
            length_mm=source_length * source_to_mm,
            axial_resistance_kohm_cm=axial_kohm_cm,
            e_leak_mV=-62.3,
            g_leak_mS_cm2=leak_density_mS_cm2,
            g_na_mS_cm2=sodium,
            g_k_mS_cm2=potassium,
            g_ca_mS_cm2=250.0,
        )

    return CellSpec(
        "modeldb112923_figure8_relay",
        (
            compartment("soma", 0.02, 0.04, 25.0, sodium=50.0, potassium=30.0),
            compartment("proximal_dendrite", 0.005, 0.05, 1.0),
            compartment("distal_dendrite", 0.003, 0.1, 1.0),
        ),
    )


def ahp_ach_layer5_spec(*, soma_axial_resistance_kohm_cm: float) -> CellSpec:
    """Build the dedicated layer-5 cell in ``Layer_5_and_Maynert_AHP_ACh.nml``.

    The source serializes input resistance for both dendrites but not the soma.
    Our compartment compiler requires one value per endpoint, so the unresolved
    KInNeSS root-compartment default must be supplied explicitly.
    """

    if soma_axial_resistance_kohm_cm <= 0:
        raise ValueError("soma_axial_resistance_kohm_cm must be a positive candidate")

    cm_to_mm = 10.0
    return CellSpec(
        "modeldb112923_ahp_ach_layer5",
        (
            CompartmentSpec(
                name="soma",
                diameter_mm=0.01 * cm_to_mm,
                length_mm=0.015 * cm_to_mm,
                axial_resistance_kohm_cm=soma_axial_resistance_kohm_cm,
                e_leak_mV=-78.0,
                g_leak_mS_cm2=0.1,
                g_na_mS_cm2=50.0,
                g_k_mS_cm2=30.0,
            ),
            CompartmentSpec(
                name="proximal_dendrite",
                diameter_mm=0.001 * cm_to_mm,
                length_mm=0.01 * cm_to_mm,
                axial_resistance_kohm_cm=35.0,
                e_leak_mV=-78.0,
                g_leak_mS_cm2=0.1,
            ),
            CompartmentSpec(
                name="distal_dendrite",
                diameter_mm=0.001 * cm_to_mm,
                length_mm=0.02 * cm_to_mm,
                axial_resistance_kohm_cm=30.0,
                e_leak_mV=-78.0,
                g_leak_mS_cm2=0.1,
            ),
        ),
    )


def ahp_density_to_total_nS(density_mS_cm2: float, cell: CellSpec) -> float:
    """Convert a somatic channel density to total conductance for Brian2."""

    if density_mS_cm2 <= 0:
        raise ValueError("density_mS_cm2 must be positive")
    return density_mS_cm2 * cell.compartment("soma").lateral_area_cm2 * 1e6
