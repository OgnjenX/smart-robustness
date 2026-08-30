"""Screen a bounded TRN-to-relay inhibitory-transfer calibration."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import (
    Figure6LearningProtocol,
    assess_figure6_cortical_recruitment,
    assess_figure6_top_down_timing,
    run_figure6_learning,
)

MONITORED_POPULATIONS = (
    "thalamic_relay",
    "layer4_excitatory_v1",
    "layer23_inhibitory_v1",
    "layer23_excitatory_v1",
    "layer5_excitatory_v1",
    "layer6i_excitatory_v1",
    "layer6ii_excitatory_v1",
    "trn",
    "thalamic_nonspecific",
)


def _plain(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    return value


def _relay_counts(indices: tuple[int, ...], expected: tuple[int, ...]) -> dict[int, int]:
    return {index: indices.count(index) for index in expected}


def _pairs(values: tuple[tuple[int, Any], ...]) -> dict[int, Any]:
    return {int(index): value for index, value in values}


def _write(
    *,
    output: Path,
    profile_path: str,
    profile: dict[str, Any],
    base_profile: dict[str, Any],
    stage_1: list[dict[str, Any]],
    stage_2: list[dict[str, Any]],
    complete: bool,
) -> None:
    stage_1_survivors = [item["scale"] for item in stage_1 if item["pass"]]
    stage_2_survivors = [item["scale"] for item in stage_2 if item["pass"]]
    artifact = {
        "schema_version": 1,
        "id": output.stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "complete" if complete else "running",
        "profile": profile_path,
        "base_candidate_fingerprint": base_profile["candidate_fingerprint"],
        "runtime_fingerprint": stage_1[0]["runtime_fingerprint"] if stage_1 else None,
        "holdouts_consulted": False,
        "stage_1_outcomes": stage_1,
        "stage_1_survivor_scales": stage_1_survivors,
        "stage_2_outcomes": stage_2,
        "stage_2_survivor_scales": stage_2_survivors,
        "assessment": {
            "stage_1_completed_count": len(stage_1),
            "registered_scale_count": len(profile["dimension"]["grid"]),
            "stage_2_completed_count": len(stage_2),
            "figure6_survivor_count": len(stage_2_survivors),
            "advance_to_same_network_match": bool(complete and stage_2_survivors),
        },
        "next_gate": profile["next_gate"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="configs/calibration/trn_gaba_transfer_figure6_grid_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    base_profile = yaml.safe_load(Path(profile["base_profile"]).read_text())
    base = runtime_conventions_for_candidate(base_profile["candidate"])
    detector = profile["fixed_detector"]
    conventions = replace(
        base,
        trn_spike_event_coordinate="absolute_physical",
        trn_spike_event_threshold_mV=float(detector["arm_mV"]),
        trn_spike_event_release_mV=float(detector["release_mV"]),
        trn_spike_event_proximal_blend_fraction=None,
    )
    projection_ids = tuple(str(value) for value in profile["scaled_projection_ids"])
    expected_indices = tuple(
        int(value) for value in profile["stage_1_gate"]["relay_active_indices"]
    )
    output = Path(args.output)
    stage_1_outcomes: list[dict[str, Any]] = []
    stage_2_outcomes: list[dict[str, Any]] = []
    for raw_scale in profile["dimension"]["grid"]:
        scale = float(raw_scale)
        values = profile["stage_1_protocol"]
        protocol = Figure6LearningProtocol(
            stimulus_ms=float(values["stimulus_ms"]),
            dt_ms=float(values["dt_ms"]),
            source_value=float(values["source_value"]),
            category_source_value=float(values["category_source_value"]),
            monitored_populations=MONITORED_POPULATIONS,
        )
        run = run_figure6_learning(
            conventions=conventions,
            protocol=protocol,
            projection_weight_scales={key: scale for key in projection_ids},
            brian=brian,
        )
        result = run.result
        relay_indices = (result.population_spike_indices or {})["thalamic_relay"]
        counts = _relay_counts(relay_indices, expected_indices)
        passed = bool(
            set(relay_indices) == set(expected_indices)
            and all(
                count
                >= int(profile["stage_1_gate"]["minimum_events_per_active_relay"])
                for count in counts.values()
            )
            and result.population_spikes["trn"] > 0
        )
        stage_1_outcomes.append(
            {
                "scale": scale,
                "runtime_fingerprint": conventions.fingerprint,
                "population_spikes": result.population_spikes,
                "relay_spike_indices": relay_indices,
                "relay_event_counts_by_index": counts,
                "pass": passed,
            }
        )
        _write(
            output=output,
            profile_path=args.profile,
            profile=profile,
            base_profile=base_profile,
            stage_1=stage_1_outcomes,
            stage_2=stage_2_outcomes,
            complete=False,
        )
        print(
            f"stage1 scale={scale:g}: relay={len(relay_indices)} "
            f"trn={result.population_spikes['trn']} pass={passed}",
            flush=True,
        )

    for stage_1 in (item for item in stage_1_outcomes if item["pass"]):
        scale = float(stage_1["scale"])
        values = profile["stage_2_protocol"]
        protocol = Figure6LearningProtocol(
            stimulus_ms=float(values["stimulus_ms"]),
            dt_ms=float(values["dt_ms"]),
            source_value=float(values["source_value"]),
            category_source_value=float(values["category_source_value"]),
            monitored_populations=MONITORED_POPULATIONS,
        )
        run = run_figure6_learning(
            conventions=conventions,
            protocol=protocol,
            record_relay_detector_diagnostics=True,
            projection_weight_scales={key: scale for key in projection_ids},
            brian=brian,
        )
        result = run.result
        relay_indices = (result.population_spike_indices or {})["thalamic_relay"]
        counts = _relay_counts(relay_indices, expected_indices)
        threshold_cycles = _pairs(result.relay_detector_threshold_upcrossings_by_index)
        arm_cycles = _pairs(result.relay_detector_arm_transitions_by_index)
        release_cycles = _pairs(result.relay_detector_release_transitions_by_index)
        recruitment = assess_figure6_cortical_recruitment(result)
        timing = assess_figure6_top_down_timing(result)
        top_down_contrast = (
            result.top_down_wide.horizontal_orientation_contrast
            + result.top_down_narrow.horizontal_orientation_contrast
        )
        gate = profile["stage_2_gate"]
        gates = {
            "relay_event_count": len(relay_indices) == int(gate["relay_event_count"]),
            "relay_active_indices": set(relay_indices) == set(expected_indices),
            "relay_events_per_active_index": all(
                count == int(gate["relay_events_per_active_index"])
                for count in counts.values()
            ),
            "relay_fresh_detector_cycles_per_active_index": all(
                threshold_cycles[index]
                == arm_cycles[index]
                == release_cycles[index]
                == int(gate["relay_fresh_detector_cycles_per_active_index"])
                for index in expected_indices
            ),
            "feedforward_chain_complete": recruitment.feedforward_chain_complete,
            "causal_pair_in_learning_window": timing.causal_pair_in_learning_window,
            "bottom_up_horizontal_orientation": result.bottom_up_oriented,
            "top_down_horizontal_contrast": (
                top_down_contrast
                >= float(gate["minimum_top_down_horizontal_contrast"])
            ),
        }
        stage_2_outcomes.append(
            {
                "scale": scale,
                "population_spikes": result.population_spikes,
                "relay_spike_indices": relay_indices,
                "relay_event_counts_by_index": counts,
                "relay_detector_threshold_upcrossings_by_index": threshold_cycles,
                "relay_detector_arm_transitions_by_index": arm_cycles,
                "relay_detector_release_transitions_by_index": release_cycles,
                "recruitment": recruitment,
                "top_down_timing": timing,
                "bottom_up_horizontal_orientation_contrast": (
                    result.bottom_up.horizontal_orientation_contrast
                ),
                "top_down_combined_horizontal_orientation_contrast": top_down_contrast,
                "top_down_combined_final_peak": max(result.top_down_combined.after),
                "gates": gates,
                "pass": all(gates.values()),
            }
        )
        _write(
            output=output,
            profile_path=args.profile,
            profile=profile,
            base_profile=base_profile,
            stage_1=stage_1_outcomes,
            stage_2=stage_2_outcomes,
            complete=False,
        )
        print(
            f"stage2 scale={scale:g}: relay={len(relay_indices)} "
            f"trn={result.population_spikes['trn']} "
            f"pass={stage_2_outcomes[-1]['pass']}",
            flush=True,
        )
    _write(
        output=output,
        profile_path=args.profile,
        profile=profile,
        base_profile=base_profile,
        stage_1=stage_1_outcomes,
        stage_2=stage_2_outcomes,
        complete=True,
    )


if __name__ == "__main__":
    main()
