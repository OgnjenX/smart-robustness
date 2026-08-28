"""Validated contracts for constrained classic-SMART calibration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from smart_robustness.classic_sector import FirstOrderRuntimeConventions

ALLOWED_STATUSES = {
    "conflicting_official_source",
    "derived_source_ambiguity",
    "source_ambiguity",
    "conflicting_protocol_source",
    "not_identifiable",
}

TRN_STAGE_A_DIMENSIONS = (
    "intrinsic_cell_convention",
    "calcium_kinetics_convention",
    "calcium_density_convention",
    "nak_rate_convention",
    "axial_convention",
    "membrane_initialization_convention",
    "spike_event_rule",
)


@dataclass(frozen=True, slots=True)
class CalibrationDimension:
    name: str
    kind: str
    status: str
    source: str
    values: tuple[str, ...] = ()
    bounds: tuple[float, float] | None = None
    grid: tuple[float, ...] = ()

    @property
    def admissible_values(self) -> tuple[str | float, ...]:
        """Return the finite, predeclared values considered by calibration."""

        return self.values if self.kind == "categorical" else self.grid


@dataclass(frozen=True, slots=True)
class CalibrationContract:
    name: str
    base_tag: str
    training_targets: tuple[str, ...]
    holdout_targets: tuple[str, ...]
    dimensions: tuple[CalibrationDimension, ...]
    forbidden_free_parameters: tuple[str, ...]

    def iter_candidates(
        self, active_dimensions: tuple[str, ...] | None = None
    ) -> Iterator[dict[str, str | float]]:
        """Enumerate deterministic complete candidates over selected dimensions.

        Dimensions outside ``active_dimensions`` are fixed to the first declared
        value. This permits causal stage screening without creating partial
        candidates or changing their fingerprint format.
        """

        known = tuple(dimension.name for dimension in self.dimensions)
        active = known if active_dimensions is None else active_dimensions
        if len(set(active)) != len(active):
            raise ValueError("active dimensions must be unique")
        unknown = sorted(set(active) - set(known))
        if unknown:
            raise ValueError(f"unknown active dimensions: {unknown}")
        active_set = set(active)
        value_axes = [
            dimension.admissible_values
            if dimension.name in active_set
            else dimension.admissible_values[:1]
            for dimension in self.dimensions
        ]
        for values in product(*value_axes):
            yield dict(zip(known, values, strict=True))

    @property
    def fingerprint(self) -> str:
        payload = {
            "name": self.name,
            "base_tag": self.base_tag,
            "training_targets": self.training_targets,
            "holdout_targets": self.holdout_targets,
            "dimensions": [
                {
                    "name": item.name,
                    "kind": item.kind,
                    "status": item.status,
                    "source": item.source,
                    "values": item.values,
                    "bounds": item.bounds,
                    "grid": item.grid,
                }
                for item in self.dimensions
            ],
            "forbidden_free_parameters": self.forbidden_free_parameters,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def candidate_fingerprint(self, values: dict[str, Any]) -> str:
        expected = {dimension.name for dimension in self.dimensions}
        if set(values) != expected:
            missing = sorted(expected - set(values))
            extra = sorted(set(values) - expected)
            raise ValueError(f"candidate dimensions differ: missing={missing}, extra={extra}")
        for dimension in self.dimensions:
            value = values[dimension.name]
            if dimension.kind == "categorical" and value not in dimension.values:
                raise ValueError(f"{dimension.name}: value {value!r} is not admissible")
            if dimension.kind == "numeric":
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError(f"{dimension.name}: numeric value required")
                assert dimension.bounds is not None
                if not dimension.bounds[0] <= float(value) <= dimension.bounds[1]:
                    raise ValueError(f"{dimension.name}: value outside declared bounds")
        encoded = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(f"{self.fingerprint}:{encoded}".encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class TrnStageAResult:
    """Causal promotion result for one isolated-TRN calibration candidate."""

    candidate_fingerprint: str
    runtime_fingerprint: str
    control_finite: bool
    driven_finite: bool
    control_post_drive_spikes: int
    driven_post_drive_spikes: int
    control_soma_range_mV: tuple[float, float]
    driven_soma_range_mV: tuple[float, float]

    @property
    def quiescent_control_pass(self) -> bool:
        return self.control_finite and self.control_post_drive_spikes == 0

    @property
    def driven_recruitment_pass(self) -> bool:
        return self.driven_finite and self.driven_post_drive_spikes >= 1

    @property
    def promoted(self) -> bool:
        return self.quiescent_control_pass and self.driven_recruitment_pass


def runtime_conventions_for_candidate(
    values: dict[str, Any],
    *,
    base: FirstOrderRuntimeConventions | None = None,
) -> FirstOrderRuntimeConventions:
    """Map calibration dimensions onto an executable classic-SMART profile.

    Protocol-only dimensions (relay input and top-down current) intentionally
    remain outside the runtime convention object and are consumed by their
    corresponding validation protocols.
    """

    from smart_robustness.classic_sector import figure6_runtime_conventions

    required = {
        "intrinsic_cell_convention",
        "calcium_kinetics_convention",
        "calcium_density_convention",
        "nak_rate_convention",
        "axial_convention",
        "membrane_initialization_convention",
        "spike_event_rule",
        "gaussian_spread_convention",
        "relay_input_interpretation",
        "top_down_current_pA",
    }
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"candidate is missing runtime dimensions: {missing}")
    calcium_kinetics = str(values["calcium_kinetics_convention"])
    calcium_gate = "reciprocal" if calcium_kinetics == "paper_2008" else "modeldb_112923"
    calcium_density_source = str(values["calcium_density_convention"])
    if calcium_density_source == "cell_specific":
        calcium_density = "table3"
    elif calcium_density_source == "trn_methods_global_250_others_table3":
        calcium_density = calcium_density_source
    else:
        calcium_density = "methods_global_250"
    return replace(
        figure6_runtime_conventions() if base is None else base,
        intrinsic_cell_convention=str(values["intrinsic_cell_convention"]),
        calcium_kinetics_convention=calcium_kinetics,
        calcium_gate_convention=calcium_gate,
        calcium_density_convention=calcium_density,
        nak_rate_convention=str(values["nak_rate_convention"]),
        axial_convention=str(values["axial_convention"]),
        membrane_initialization_convention=str(
            values["membrane_initialization_convention"]
        ),
        spike_event_rule=str(values["spike_event_rule"]),
        trn_spike_event_coordinate=(
            None
            if values.get("trn_spike_event_coordinate") is None
            else str(values["trn_spike_event_coordinate"])
        ),
        trn_spike_event_threshold_mV=(
            None
            if values.get("trn_spike_event_threshold_mV") is None
            else float(values["trn_spike_event_threshold_mV"])
        ),
        trn_spike_event_voltage_offset_mV=(
            None
            if values.get("trn_spike_event_voltage_offset_mV") is None
            else float(values["trn_spike_event_voltage_offset_mV"])
        ),
        trn_spike_event_proximal_blend_fraction=(
            None
            if values.get("trn_spike_event_proximal_blend_fraction") is None
            else float(values["trn_spike_event_proximal_blend_fraction"])
        ),
        trn_potassium_convention=str(
            values.get(
                "trn_potassium_convention",
                (
                    figure6_runtime_conventions() if base is None else base
                ).trn_potassium_convention,
            )
        ),
        trn_calcium_source_convention=str(
            values.get(
                "trn_calcium_source_convention",
                (
                    figure6_runtime_conventions() if base is None else base
                ).trn_calcium_source_convention,
            )
        ),
        trn_dendritic_calcium_density_convention=str(
            values.get(
                "trn_dendritic_calcium_density_convention",
                (
                    figure6_runtime_conventions() if base is None else base
                ).trn_dendritic_calcium_density_convention,
            )
        ),
        trn_dendritic_calcium_density_mS_cm2=(
            None
            if values.get("trn_dendritic_calcium_density_mS_cm2") is None
            else float(values["trn_dendritic_calcium_density_mS_cm2"])
        ),
        trn_soma_proximal_axial_conductance_scale=float(
            values.get(
                "trn_soma_proximal_axial_conductance_scale",
                (
                    figure6_runtime_conventions() if base is None else base
                ).trn_soma_proximal_axial_conductance_scale,
            )
        ),
        gaussian_spread_convention=str(values["gaussian_spread_convention"]),
        gaussian_learning_bounds_convention=str(
            values.get(
                "gaussian_learning_bounds_convention",
                (figure6_runtime_conventions() if base is None else base).gaussian_learning_bounds_convention,
            )
        ),
        postsynaptic_depression_scale_convention=str(
            values.get(
                "postsynaptic_depression_scale_convention",
                (
                    figure6_runtime_conventions() if base is None else base
                ).postsynaptic_depression_scale_convention,
            )
        ),
        postsynaptic_learning_threshold_mV=float(
            values.get(
                "postsynaptic_learning_threshold_mV",
                (
                    figure6_runtime_conventions() if base is None else base
                ).postsynaptic_learning_threshold_mV,
            )
        ),
        postsynaptic_learning_coordinate=str(
            values.get(
                "postsynaptic_learning_coordinate",
                (
                    figure6_runtime_conventions() if base is None else base
                ).postsynaptic_learning_coordinate,
            )
        ),
        top_down_learning_rule_convention=str(
            values.get(
                "top_down_learning_rule_convention",
                (
                    figure6_runtime_conventions() if base is None else base
                ).top_down_learning_rule_convention,
            )
        ),
        postsynaptic_learning_timestamp=str(
            values.get(
                "postsynaptic_learning_timestamp",
                (
                    figure6_runtime_conventions() if base is None else base
                ).postsynaptic_learning_timestamp,
            )
        ),
        postsynaptic_signal_convention=str(
            values.get(
                "postsynaptic_signal_convention",
                (
                    figure6_runtime_conventions() if base is None else base
                ).postsynaptic_signal_convention,
            )
        ),
    )


def run_trn_stage_a_candidate(
    contract: CalibrationContract,
    values: dict[str, Any],
    *,
    protocol=None,
    brian=None,
) -> TrnStageAResult:
    """Run independent control and driven TRN trials for one candidate."""

    from smart_robustness.validation.isolated_cells import run_trn_recruitment_condition

    candidate_fingerprint = contract.candidate_fingerprint(values)
    conventions = runtime_conventions_for_candidate(values)
    control = run_trn_recruitment_condition(
        driven=False, conventions=conventions, protocol=protocol, brian=brian
    )
    driven = run_trn_recruitment_condition(
        driven=True, conventions=conventions, protocol=protocol, brian=brian
    )
    if control.convention_fingerprint != driven.convention_fingerprint:
        raise RuntimeError("control and driven trials used different runtime conventions")
    return TrnStageAResult(
        candidate_fingerprint=candidate_fingerprint,
        runtime_fingerprint=control.convention_fingerprint,
        control_finite=control.finite,
        driven_finite=driven.finite,
        control_post_drive_spikes=control.post_drive_spike_count,
        driven_post_drive_spikes=driven.post_drive_spike_count,
        control_soma_range_mV=control.soma_voltage_range_mV,
        driven_soma_range_mV=driven.soma_voltage_range_mV,
    )


def load_calibration_contract(path: str | Path) -> CalibrationContract:
    raw = yaml.safe_load(Path(path).read_text())
    if raw.get("schema_version") != 1:
        raise ValueError("calibration contract requires schema_version 1")
    training = tuple(raw["training_targets"])
    holdout = tuple(raw["holdout_targets"])
    overlap = sorted(set(training) & set(holdout))
    if overlap:
        raise ValueError(f"training and holdout targets overlap: {overlap}")
    dimensions: list[CalibrationDimension] = []
    for name, spec in raw["dimensions"].items():
        kind = spec["kind"]
        status = spec["status"]
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"{name}: status {status!r} is not calibration-admissible")
        if kind == "categorical":
            values = tuple(spec.get("values", ()))
            if len(values) < 2:
                raise ValueError(f"{name}: categorical dimension needs at least two values")
            bounds = None
            grid = ()
        elif kind == "numeric":
            values = ()
            bounds_raw = tuple(float(value) for value in spec.get("bounds", ()))
            if len(bounds_raw) != 2 or bounds_raw[0] >= bounds_raw[1]:
                raise ValueError(f"{name}: numeric bounds must be increasing")
            bounds = (bounds_raw[0], bounds_raw[1])
            grid = tuple(float(value) for value in spec.get("grid", ()))
            if not grid or any(not bounds[0] <= value <= bounds[1] for value in grid):
                raise ValueError(f"{name}: grid must be nonempty and within bounds")
        else:
            raise ValueError(f"{name}: unknown dimension kind {kind!r}")
        dimensions.append(
            CalibrationDimension(
                name=name,
                kind=kind,
                status=status,
                source=str(spec["source"]),
                values=values,
                bounds=bounds,
                grid=grid,
            )
        )
    forbidden = tuple(raw.get("forbidden_free_parameters", ()))
    if not forbidden:
        raise ValueError("calibration contract must declare forbidden free parameters")
    return CalibrationContract(
        name=str(raw["name"]),
        base_tag=str(raw["base_tag"]),
        training_targets=training,
        holdout_targets=holdout,
        dimensions=tuple(dimensions),
        forbidden_free_parameters=forbidden,
    )
