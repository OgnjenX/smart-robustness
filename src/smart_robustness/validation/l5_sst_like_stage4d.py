"""Pre-outcome contracts for the registered SST-like inter-area holdout.

This module does not authorize execution or observe network outcomes. The
checkpoint runner must additionally verify its separate seal and parent inputs.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .. import classic_sector
from ..models.sst_like_feedback import make_l5_sst_like_sector_builder
from .higher_order import Figure16Protocol

RESULT = "results/l5-sst-like-stage4d-figure16.yaml"
GATES = (
    "exactly_1000_recording_samples",
    "v1_and_v2_fields_finite",
    "strongest_lower_frequency_correlation_strictly_exceeds_20_100_hz",
)
BANDS = [[2.0, 4.0], [4.0, 8.0], [8.0, 12.0], [12.0, 20.0], [20.0, 100.0]]


def validate_protocol(registration: dict, parent_registration: dict) -> Figure16Protocol:
    """Reject drift in either the extension or its inherited Figure-16 assay."""

    stage = registration["stage4d_figure16"]
    parent = parent_registration["figure16_protocol"]
    common = {
        "network": "complete_recovered_v1_pulvinar_v2_catalog",
        "prestimulus_ms": 1000.0,
        "recording_ms": 1000.0,
        "integration_dt_ms": 0.01,
        "recording_sample_ms": 1.0,
        "inter_area_delay_ms": 10.0,
        "frequency_bands_hz": BANDS,
        "backend": "brian2_cpp_standalone",
    }
    if any(stage.get(k) != v or parent.get(k) != v for k, v in common.items()):
        raise ValueError("Stage-4D inherited protocol changed")
    stage_only = {
        "input_learning_state": "corresponding-stage4a-repetition-and-point",
        "intervention_scope": "v1-layer5-route-only",
        "geometry_seeds": [16, 17],
        "gates": list(GATES),
        "result_target": RESULT,
    }
    parent_only = {
        "stimulus": "horizontal_five_cell_bar",
        "inter_area_delay_pathway": "v1_layer23_to_v2_layer4",
        "higher_area_signal": "mean_v2_inferior_300um_electrode_tips",
        "lower_area_signal": "mean_v1_superior_300um_electrode_tips",
        "geometry_seed_v1": 16,
        "geometry_seed_v2": 17,
    }
    if any(stage.get(k) != v for k, v in stage_only.items()) or any(
        parent.get(k) != v for k, v in parent_only.items()
    ):
        raise ValueError("Stage-4D intervention, geometry or signal reduction changed")
    if parent_registration.get("gates") != dict.fromkeys(GATES, True):
        raise ValueError("Stage-4D parent gates changed")
    design = registration["execution_design"]
    required = (
        "exact_repeat_required",
        "execute_all_seven_points_in_every_substage",
        "no_point_dropped_after_a_failed_substage",
        "failed_earlier_substage_cannot_be_compensated_by_later_pass",
        "checkpoint_after_every_repetition",
        "single_process_lock_per_result",
        "separate_implementation_commit_and_pre_outcome_seal_per_substage",
        "assess_complete_substage_before_implementing_next_substage",
    )
    if (
        any(design.get(k) is not True for k in required)
        or design.get("repetitions_per_point_per_substage") != 2
        or design.get("result_guided_parameter_or_analysis_changes") != "forbidden"
    ):
        raise ValueError("Stage-4D execution or no-selection contract changed")
    # These are two area-specific geometries in each run, NOT two seed arms.
    return Figure16Protocol()


def full_network_builder(point: dict):
    """Retain the complete V1/pulvinar/V2 catalog; add only the V1-L5 route."""

    return make_l5_sst_like_sector_builder(
        base_builder=classic_sector.build_full_smart_network,
        total_conductance_nS=float(point["total_conductance_nS"]),
        delay_ms=float(point["delay_ms"]),
    )


def classify_point(outcomes: list[dict[str, Any]], *, earlier: dict[str, str]) -> dict:
    """Keep deterministic repeat, engineering and earlier-failure boundaries."""

    if len(outcomes) != 2 or [o.get("repetition") for o in outcomes] != [0, 1]:
        raise ValueError("Stage-4D requires ordered repetitions zero and one")
    allowed = {
        "stage4a": {"learning_first_order_survival", "failure"},
        "stage4b": {"figure14_survival", "figure14_failure"},
        "stage4c": {"figure15_survival", "figure15_failure"},
    }
    if set(earlier) != set(allowed) or any(earlier[k] not in v for k, v in allowed.items()):
        raise ValueError("Stage-4D requires assessed earlier-stage classifications")
    normalized = []
    for outcome in outcomes:
        if set(outcome.get("gates", {})) != set(GATES) or any(
            type(v) is not bool for v in outcome["gates"].values()
        ):
            raise ValueError("Stage-4D requires all three Boolean gates")
        value = deepcopy(outcome)
        value.pop("repetition")
        normalized.append(value)
    exact = normalized[0] == normalized[1]
    engineering_ok = exact and all(o["gates"][GATES[0]] and o["gates"][GATES[1]] for o in outcomes)
    passed = engineering_ok and all(all(o["gates"].values()) for o in outcomes)
    classification = (
        "engineering_stop"
        if not engineering_ok
        else "figure16_survival"
        if passed
        else "figure16_failure"
    )
    if not engineering_ok:
        joint = "engineering_stop"
    elif earlier["stage4a"] == "failure":
        joint = "learning_or_first_order_failure"
    elif (
        passed
        and earlier["stage4b"] == "figure14_survival"
        and earlier["stage4c"] == "figure15_survival"
    ):
        joint = "complete_progression_survival"
    else:
        joint = "bounded_partial_survival"
    return {
        "exact_repeat": exact,
        "both_figure16_gate_sets_pass": passed,
        "classification": classification,
        "joint_classification": joint,
        "earlier_classifications": dict(earlier),
    }
