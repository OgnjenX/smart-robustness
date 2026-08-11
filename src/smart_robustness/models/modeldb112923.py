"""Facts recovered from the official ModelDB 112923 SMART backup.

The raw archive is copyright-marked and has no explicit redistribution
license, so it is not vendored. This module records model facts and hashes;
the original files remain downloadable from ModelDB.
"""

from __future__ import annotations

from dataclasses import dataclass

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


def first_order_structural_counts() -> tuple[int, int]:
    """Return executable-source cell and compartment totals for V1."""

    cells = sum(width * height for _, _, (width, height) in FIRST_ORDER_POPULATIONS)
    compartments = sum(
        count * width * height for _, count, (width, height) in FIRST_ORDER_POPULATIONS
    )
    return cells, compartments


def figure8_relay_spec(*, leak_density_mS_cm2: float) -> CellSpec:
    """Build the dedicated ``Ca_rebound.xml`` relay cell.

    ``Ca_rebound.xml`` does not serialize a leak density for this cell. The
    caller must therefore provide the KInNeSS default or a declared calibrated
    candidate; no value is silently inherited from Table 3.
    """

    if leak_density_mS_cm2 <= 0:
        raise ValueError("leak_density_mS_cm2 must be an explicit positive candidate")

    def compartment(
        name: str,
        diameter_mm: float,
        length_mm: float,
        axial_kohm_cm: float,
        *,
        sodium: float | None = None,
        potassium: float | None = None,
    ) -> CompartmentSpec:
        return CompartmentSpec(
            name=name,
            diameter_mm=diameter_mm,
            length_mm=length_mm,
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
