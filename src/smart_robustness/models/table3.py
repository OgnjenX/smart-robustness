"""Grossberg--Versace (2008) Table 3 cellular parameters.

The source table reports compartment geometry in millimetres, axial resistance
in kOhm*cm, reversal potentials in mV, and channel densities in mS/cm^2.
Missing channels are represented by ``None`` rather than zero so that absence
cannot be confused with an explicitly reported zero-valued parameter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CompartmentSpec:
    name: str
    diameter_mm: float
    length_mm: float
    axial_resistance_kohm_cm: float
    e_leak_mV: float
    g_leak_mS_cm2: float
    g_na_mS_cm2: float | None = None
    g_k_mS_cm2: float | None = None
    g_ca_mS_cm2: float | None = None

    @property
    def lateral_area_cm2(self) -> float:
        """Cylindrical lateral membrane area, matching the paper's pi*D*L term."""
        return math.pi * self.diameter_mm * self.length_mm / 100.0

    def capacitance_pF(self, specific_capacitance_uF_cm2: float = 1.0) -> float:
        """Convert a specific membrane capacitance to total compartment pF."""
        return specific_capacitance_uF_cm2 * self.lateral_area_cm2 * 1e6

    def conductance_nS(self, channel: str) -> float:
        """Convert a reported channel density to total compartment conductance."""
        densities = {
            "leak": self.g_leak_mS_cm2,
            "na": self.g_na_mS_cm2,
            "k": self.g_k_mS_cm2,
            "ca": self.g_ca_mS_cm2,
        }
        try:
            density = densities[channel]
        except KeyError as exc:
            raise ValueError(f"unknown channel {channel!r}; expected {tuple(densities)}") from exc
        if density is None:
            return 0.0
        return density * self.lateral_area_cm2 * 1e6


@dataclass(frozen=True)
class CellSpec:
    name: str
    compartments: tuple[CompartmentSpec, ...]

    @property
    def soma(self) -> CompartmentSpec:
        return self.compartments[0]

    def compartment(self, name: str) -> CompartmentSpec:
        for compartment in self.compartments:
            if compartment.name == name:
                return compartment
        raise KeyError(f"{self.name!r} has no compartment {name!r}")


def _c(
    name: str,
    diameter: float,
    length: float,
    axial: float,
    e_leak: float,
    g_leak: float,
    g_na: float | None = None,
    g_k: float | None = None,
    g_ca: float | None = None,
) -> CompartmentSpec:
    return CompartmentSpec(name, diameter, length, axial, e_leak, g_leak, g_na, g_k, g_ca)


TABLE3_CELLS: dict[str, CellSpec] = {
    "thalamic_relay": CellSpec(
        "thalamic_relay",
        (
            _c("soma", 0.05, 0.06, 8, -60, 0.01, 100, 100),
            _c("proximal_dendrite", 0.005, 0.008, 8, -60, 0.01, g_ca=10),
            _c("distal_dendrite", 0.005, 0.008, 8, -60, 0.01, g_ca=10),
        ),
    ),
    "thalamic_matrix": CellSpec(
        "thalamic_matrix",
        (
            _c("soma", 0.05, 0.06, 8, -60, 0.01, 100, 100),
            _c("proximal_dendrite", 0.005, 0.008, 8, -60, 0.01, g_ca=10),
            _c("distal_dendrite", 0.005, 0.008, 8, -60, 0.01, g_ca=10),
        ),
    ),
    "thalamic_interneuron": CellSpec(
        "thalamic_interneuron",
        (
            _c("soma", 0.02, 0.02, 60, -49, 0.01, 50, 30),
            _c("proximal_dendrite", 0.01, 0.1, 60, -49, 0.01),
        ),
    ),
    "trn": CellSpec(
        "trn",
        (
            _c("soma", 0.05, 0.05, 10, -69, 0.1, 100, 100),
            _c("proximal_dendrite", 0.01, 0.05, 10, -69, 0.1, g_ca=10),
            _c("distal_dendrite", 0.01, 0.05, 10, -69, 0.1, g_ca=10),
        ),
    ),
    "thalamic_nonspecific": CellSpec(
        "thalamic_nonspecific",
        (
            _c("soma", 0.08, 0.08, 10, -64, 0.09, 100, 100),
            _c("proximal_dendrite", 0.015, 0.1, 10, -64, 0.1, g_ca=250),
            _c("distal_dendrite", 0.015, 0.1, 10, -64, 0.1, g_ca=250),
        ),
    ),
    "layer4_excitatory": CellSpec(
        "layer4_excitatory",
        (
            _c("soma", 0.05, 0.05, 40, -65, 0.01, 50, 30),
            _c("proximal_dendrite", 0.01, 0.25, 40, -65, 0.01),
        ),
    ),
    "layer4_inhibitory": CellSpec(
        "layer4_inhibitory",
        (
            _c("soma", 0.02, 0.02, 100, -50, 0.01, 50, 30),
            _c("proximal_dendrite", 0.01, 0.05, 100, -50, 0.01),
        ),
    ),
    "layer23_excitatory": CellSpec(
        "layer23_excitatory",
        (
            _c("soma", 0.05, 0.05, 100, -65, 0.05, 50, 30),
            _c("proximal_dendrite", 0.02, 0.225, 100, -65, 0.05),
        ),
    ),
    "layer23_inhibitory": CellSpec(
        "layer23_inhibitory",
        (
            _c("soma", 0.02, 0.02, 60, -49, 0.01, 50, 30),
            _c("proximal_dendrite", 0.01, 0.05, 60, -49, 0.01),
        ),
    ),
    "layer5_excitatory": CellSpec(
        "layer5_excitatory",
        (
            _c("soma", 0.1, 0.15, 5, -72, 0.1, 50, 30),
            _c("proximal_dendrite", 0.06, 0.4, 5, -72, 0.03),
            _c("distal_dendrite", 0.06, 0.5, 5, -72, 0.03, 50, 30),
        ),
    ),
    "layer6i_excitatory": CellSpec(
        "layer6i_excitatory",
        (
            _c("soma", 0.08, 0.1, 80, -70, 0.15, 50, 30),
            _c("proximal_dendrite", 0.05, 0.1, 80, -70, 0.9),
        ),
    ),
    "layer6ii_excitatory": CellSpec(
        "layer6ii_excitatory",
        (
            _c("soma", 0.06, 0.1, 25, -64, 0.1, 50, 30),
            _c("proximal_dendrite", 0.08, 0.1, 25, -64, 0.03),
            _c("distal_dendrite", 0.08, 0.2, 25, -64, 0.03),
        ),
    ),
}


def get_cell_spec(name: str) -> CellSpec:
    try:
        return TABLE3_CELLS[name]
    except KeyError as exc:
        raise ValueError(f"unknown SMART Table 3 cell {name!r}; expected {tuple(TABLE3_CELLS)}") from exc
