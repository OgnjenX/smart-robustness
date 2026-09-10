"""Preregisterable GIF-only parameters for controlled SMART substitutions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GIFParameters:
    """Pozzorini--Mensi generalized integrate-and-fire parameters.

    SMART supplies capacitance, leak conductance, morphology, axial coupling,
    receptor ports, thalamic T-current, and AHP/ACh.  Threshold and reset are
    expressed relative to each SMART cell's somatic leak reversal so the
    literature arm changes the spike generator without replacing passive
    membrane or anatomical parameters.
    """

    threshold_offset_mV: float
    reset_offset_mV: float
    eta_fast_pA: float
    eta_fast_tau_ms: float
    eta_slow_pA: float
    eta_slow_tau_ms: float
    gamma_fast_mV: float
    gamma_fast_tau_ms: float
    gamma_slow_mV: float
    gamma_slow_tau_ms: float
    escape_rate_hz: float
    stochasticity_mV: float
    refractory_ms: float = 4.0
    effective_leak_offset_mV: float = 0.0
    somatic_capacitance_scale: float = 1.0
    somatic_leak_conductance_scale: float = 1.0

    def __post_init__(self) -> None:
        values = asdict(self)
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("all GIF parameters must be finite")
        for name in (
            "eta_fast_tau_ms",
            "eta_slow_tau_ms",
            "gamma_fast_tau_ms",
            "gamma_slow_tau_ms",
            "escape_rate_hz",
            "stochasticity_mV",
            "somatic_capacitance_scale",
            "somatic_leak_conductance_scale",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.refractory_ms < 0:
            raise ValueError("refractory_ms cannot be negative")

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> GIFParameters:
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown GIF parameters: {sorted(unknown)}")
        return cls(**{field: float(value) for field, value in values.items()})


# Means used without network tuning in Setareh et al. (2017), Table 1,
# extracted from Mensi et al. (2012). Passive C, gL, and EL are deliberately
# inherited from SMART. Absolute source thresholds/resets are converted to
# offsets from the source EL before transfer to each SMART cell class.
LITERATURE_EXCITATORY = GIFParameters(
    threshold_offset_mV=27.4,  # -39.6 - (-67.0)
    reset_offset_mV=30.3,  # -36.7 - (-67.0)
    eta_fast_pA=56.7,
    eta_fast_tau_ms=57.8,
    eta_slow_pA=-6.9,
    eta_slow_tau_ms=218.2,
    gamma_fast_mV=11.7,
    gamma_fast_tau_ms=53.8,
    gamma_slow_mV=1.8,
    gamma_slow_tau_ms=640.0,
    escape_rate_hz=10_000.0,
    stochasticity_mV=1.4,
    refractory_ms=4.0,
)

LITERATURE_INHIBITORY = GIFParameters(
    threshold_offset_mV=30.0,  # -41.2 - (-71.2)
    reset_offset_mV=22.8,  # -48.4 - (-71.2)
    eta_fast_pA=31.8,
    eta_fast_tau_ms=11.5,
    eta_slow_pA=1.6,
    eta_slow_tau_ms=500.1,
    gamma_fast_mV=5.6,
    gamma_fast_tau_ms=11.5,
    gamma_slow_mV=0.6,
    gamma_slow_tau_ms=473.7,
    escape_rate_hz=10_000.0,
    stochasticity_mV=0.6,
    refractory_ms=4.0,
)


INHIBITORY_SMART_CELL_CLASSES = frozenset(
    {
        "trn",
        "thalamic_interneuron",
        "layer4_inhibitory_v1",
        "layer23_inhibitory_v1",
    }
)


def literature_gif_parameters(cell_class: str) -> GIFParameters:
    """Return the preregistered excitatory/inhibitory literature mapping."""

    return (
        LITERATURE_INHIBITORY
        if cell_class in INHIBITORY_SMART_CELL_CLASSES
        else LITERATURE_EXCITATORY
    )
