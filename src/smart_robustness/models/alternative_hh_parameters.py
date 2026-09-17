"""Parameters for the morphology-preserving Pospischil-type HH substitution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class AlternativeHHParameters:
    """Somatic conductance densities and kinetic shift from Pospischil et al."""

    threshold_mV: float
    sodium_density_mS_cm2: float
    potassium_density_mS_cm2: float
    m_current_density_mS_cm2: float
    m_current_tau_max_ms: float

    def __post_init__(self) -> None:
        values = asdict(self)
        if not all(isfinite(float(value)) for value in values.values()):
            raise ValueError("alternative-HH parameters must be finite")
        if self.sodium_density_mS_cm2 <= 0:
            raise ValueError("sodium density must be positive")
        if self.potassium_density_mS_cm2 <= 0:
            raise ValueError("potassium density must be positive")
        if self.m_current_density_mS_cm2 < 0:
            raise ValueError("M-current density cannot be negative")
        if self.m_current_tau_max_ms <= 0:
            raise ValueError("M-current tau max must be positive")

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> AlternativeHHParameters:
        expected = set(cls.__dataclass_fields__)
        if set(values) != expected:
            missing = sorted(expected - set(values))
            extra = sorted(set(values) - expected)
            raise ValueError(
                f"alternative-HH parameter keys differ: missing={missing}, extra={extra}"
            )
        return cls(**{name: float(values[name]) for name in expected})


POSPISCHIL_RS_MEAN = AlternativeHHParameters(
    threshold_mV=-61.5,
    sodium_density_mS_cm2=50.0,
    potassium_density_mS_cm2=4.8,
    m_current_density_mS_cm2=0.13,
    m_current_tau_max_ms=1123.5,
)

POSPISCHIL_FS_MEAN = AlternativeHHParameters(
    threshold_mV=-61.84,
    sodium_density_mS_cm2=46.0,
    potassium_density_mS_cm2=5.1,
    m_current_density_mS_cm2=0.07,
    m_current_tau_max_ms=824.5,
)

