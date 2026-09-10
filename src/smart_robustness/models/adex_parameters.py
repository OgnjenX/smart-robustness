"""Preregisterable AdEx-only parameters for controlled SMART substitutions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AdExParameters:
    """AdEx parameters that are not already fixed by the SMART cell library.

    Capacitance, leak conductance, morphology, axial coupling, T-type calcium,
    and SMART AHP/ACh are supplied by the baseline. The optional effective-leak
    offset is zero for strict substitutions; separately labeled calibration
    arms may fit it because standard AdEx includes its own resting-potential
    parameter whereas removing HH Na/K changes the classic resting balance.
    """

    threshold_offset_mV: float
    slope_factor_mV: float
    reset_offset_mV: float
    peak_mV: float
    subthreshold_adaptation_nS: float
    spike_adaptation_pA: float
    adaptation_time_constant_ms: float
    refractory_ms: float = 2.0
    effective_leak_offset_mV: float = 0.0
    somatic_capacitance_scale: float = 1.0
    somatic_leak_conductance_scale: float = 1.0

    def __post_init__(self) -> None:
        values = asdict(self)
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("all AdEx parameters must be finite")
        if self.slope_factor_mV <= 0:
            raise ValueError("slope_factor_mV must be positive")
        if self.adaptation_time_constant_ms <= 0:
            raise ValueError("adaptation_time_constant_ms must be positive")
        if self.subthreshold_adaptation_nS < 0 or self.spike_adaptation_pA < 0:
            raise ValueError("AdEx adaptation parameters cannot be negative")
        if self.refractory_ms < 0:
            raise ValueError("refractory_ms cannot be negative")
        if self.somatic_capacitance_scale <= 0:
            raise ValueError("somatic_capacitance_scale must be positive")
        if self.somatic_leak_conductance_scale <= 0:
            raise ValueError("somatic_leak_conductance_scale must be positive")

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> AdExParameters:
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown AdEx parameters: {sorted(unknown)}")
        return cls(**{field: float(value) for field, value in values.items()})


# Brette--Gerstner regular-spiking AdEx-only constants, with the passive
# membrane terms intentionally inherited from each SMART Table 3 cell.
LITERATURE_REGULAR_SPIKING = AdExParameters(
    threshold_offset_mV=20.2,
    slope_factor_mV=2.0,
    reset_offset_mV=0.0,
    peak_mV=20.0,
    subthreshold_adaptation_nS=4.0,
    spike_adaptation_pA=80.5,
    adaptation_time_constant_ms=144.0,
    refractory_ms=2.0,
    effective_leak_offset_mV=0.0,
    somatic_capacitance_scale=1.0,
    somatic_leak_conductance_scale=1.0,
)
