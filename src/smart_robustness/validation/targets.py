"""Predeclared validation targets from Grossberg and Versace (2008).

This module contains claims and protocol values, not tuning parameters.  A
target whose exact numerical trace is absent from the publication is marked as
qualitative or not identifiable instead of being assigned a fabricated number.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class EvidenceClass(StrEnum):
    """Strength of the primary-source evidence behind an acceptance target."""

    EXACT_SOURCE = "exact-source"
    STRUCTURAL = "structural"
    QUALITATIVE = "qualitative"
    APPROXIMATE_NUMERIC = "approximate-numeric"
    NOT_IDENTIFIABLE = "not-identifiable"


@dataclass(frozen=True, slots=True)
class ValidationTarget:
    """One immutable, source-backed classic-SMART acceptance target."""

    id: str
    figure: str
    outcome: str
    evidence: tuple[EvidenceClass, ...]
    source: str
    protocol: Mapping[str, object]
    numeric_targets: Mapping[str, float]
    unresolved: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id or any(character.isspace() for character in self.id):
            raise ValueError("validation target IDs must be non-empty slugs")
        if not self.evidence:
            raise ValueError(f"{self.id}: at least one evidence class is required")
        object.__setattr__(self, "protocol", MappingProxyType(dict(self.protocol)))
        object.__setattr__(self, "numeric_targets", MappingProxyType(dict(self.numeric_targets)))


def _target(**kwargs: object) -> ValidationTarget:
    return ValidationTarget(**kwargs)  # type: ignore[arg-type]


CLASSIC_SMART_TARGETS: tuple[ValidationTarget, ...] = (
    _target(
        id="fig6_learning",
        figure="Figure 6",
        outcome="Published gated timing-curve family and oriented learned maps.",
        evidence=(EvidenceClass.EXACT_SOURCE, EvidenceClass.QUALITATIVE),
        source="Methods 4.3 and Figure 6",
        protocol={"relative_spike_time_min_ms": -30.0, "relative_spike_time_max_ms": 30.0},
        numeric_targets={},
        unresolved=(
            "Curve amplitudes and exact map values require digitization or source recovery.",
        ),
    ),
    _target(
        id="fig7_match_mismatch_arousal",
        figure="Figure 7",
        outcome="Mismatch disinhibits nonspecific thalamus relative to match.",
        evidence=(EvidenceClass.STRUCTURAL, EvidenceClass.QUALITATIVE),
        source="Figure 7 caption and Results 2.5",
        protocol={
            "match_bottom_up": "horizontal",
            "match_top_down": "horizontal",
            "mismatch_bottom_up": "vertical",
            "mismatch_top_down": "horizontal",
        },
        numeric_targets={},
        unresolved=(
            "The publication gives no numeric nonspecific-thalamus rates or complete time series.",
        ),
    ),
    _target(
        id="fig8_relay_tonic_burst",
        figure="Figure 8",
        outcome="Depolarized relay firing is tonic; hyperpolarization enables a T-current rebound burst.",
        evidence=(EvidenceClass.EXACT_SOURCE, EvidenceClass.QUALITATIVE),
        source="Figure 8 and Methods 4.6",
        protocol={"current_injection_nA": 0.3},
        numeric_targets={},
        unresolved=("Exact burst latency and spike count are graphical only.",),
    ),
    _target(
        id="fig10_reset",
        figure="Figure 10",
        outcome="A mismatch-driven nonspecific burst resets the active layer-4 winner.",
        evidence=(EvidenceClass.STRUCTURAL, EvidenceClass.QUALITATIVE),
        source="Figure 10 and Results 2.7",
        protocol={"reported_dendritic_recording_distance_um": 400.0},
        numeric_targets={},
        unresolved=("Reset latency and winner probability are not tabulated.",),
    ),
    _target(
        id="fig11_depletion",
        figure="Figure 11",
        outcome="Use and depletion strength deepen depletion; faster recovery reduces it.",
        evidence=(EvidenceClass.EXACT_SOURCE, EvidenceClass.QUALITATIVE),
        source="Figure 11 and Methods 4.4",
        protocol={
            "duration_ms": 150.0,
            "conditions": ((0.5, 50.0), (1.0, 50.0), (1.0, 10.0)),
            "firing_rate_pairs_hz": ((2.0, 3.0), (7.0, 12.0)),
        },
        numeric_targets={},
    ),
    _target(
        id="fig12_19_ahp_ach",
        figure="Figures 12 and 19",
        outcome="AHP adapts layer-5 firing; ACh suppresses AHP and increases excitability.",
        evidence=(EvidenceClass.EXACT_SOURCE, EvidenceClass.QUALITATIVE),
        source="Figures 12 and 19; Methods 4.7",
        protocol={
            "driven_rate_hz": 80.0,
            "ach_duration_ms": 100.0,
            "ahp_tau_rise_ms": 80.0,
            "ahp_tau_fall_ms": 100.0,
            "ach_tau_rise_ms": 5.0,
            "ach_tau_fall_ms": 6.0,
            "recovery_ms": 500.0,
        },
        numeric_targets={},
        unresolved=(
            "The exact maximal AHP conductance is not uniquely identifiable from the paper text.",
        ),
    ),
    _target(
        id="fig14_match_mismatch_spectra",
        figure="Figure 14",
        outcome="Match is gamma-dominant; mismatch increases slower activity and reduces gamma.",
        evidence=(EvidenceClass.EXACT_SOURCE, EvidenceClass.QUALITATIVE),
        source="Figure 14 and Methods 4.9–4.10",
        protocol={
            "duration_ms": 1000.0,
            "stimulated_relay_cells": 5,
            "hamming_window_ms": 200.0,
            "gamma_band_hz": (20.0, 70.0),
            "middle_band_caption_hz": (8.0, 20.0),
            "middle_band_methods_hz": (8.0, 10.0),
        },
        numeric_targets={},
        unresolved=("The caption and Methods disagree on the middle frequency band.",),
    ),
    _target(
        id="fig15_local_synchrony",
        figure="Figure 15",
        outcome="Nearby layer-4 cells synchronize in the gamma range.",
        evidence=(EvidenceClass.APPROXIMATE_NUMERIC,),
        source="Figure 15 caption",
        protocol={"local_range_um": 300.0},
        numeric_targets={"simulation_peak_hz": 44.0, "experiment_peak_hz": 50.0},
    ),
    _target(
        id="fig16_long_range_synchrony",
        figure="Figure 16",
        outcome="Inter-area synchrony is stronger at lower frequencies than local gamma.",
        evidence=(EvidenceClass.EXACT_SOURCE, EvidenceClass.QUALITATIVE),
        source="Figure 16 and Methods 4.10–4.11",
        protocol={
            "pre_recording_input_ms": 1000.0,
            "inter_area_delay_ms": 10.0,
            "inter_area_delay_pathway": "V1 layer 2/3 to V2 layer 4",
            "bands_hz": ((2.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 20.0), (20.0, 100.0)),
        },
        numeric_targets={},
        unresolved=(
            "Published correlation amplitudes are graphical only.",
            "SMART.nml serializes 5 ms for the caption-overridden pathway.",
        ),
    ),
)


_TARGETS_BY_ID = MappingProxyType({target.id: target for target in CLASSIC_SMART_TARGETS})


def get_validation_target(target_id: str) -> ValidationTarget:
    """Return a predeclared target, raising a useful error for unknown IDs."""

    try:
        return _TARGETS_BY_ID[target_id]
    except KeyError as error:
        known = ", ".join(sorted(_TARGETS_BY_ID))
        raise KeyError(
            f"unknown validation target {target_id!r}; known targets: {known}"
        ) from error
