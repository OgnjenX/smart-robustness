"""Apply isolated TRN detector survivors to the Figure 6 prerequisite."""

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


def _pairs(values: tuple[tuple[int, Any], ...]) -> dict[int, Any]:
    return {int(index): value for index, value in values}


def _write_artifact(
    *,
    output: Path,
    profile_path: str,
    profile: dict[str, Any],
    base_profile: dict[str, Any],
    outcomes: list[dict[str, Any]],
    complete: bool,
) -> None:
    survivors = [item["pair"]["label"] for item in outcomes if item["figure6_pass"]]
    artifact = {
        "schema_version": 1,
        "id": output.stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": (
            "figure6-prerequisite-complete"
            if complete
            else "figure6-prerequisite-running"
        ),
        "profile": profile_path,
        "base_candidate_fingerprint": base_profile["candidate_fingerprint"],
        "stage_1_artifact": profile["stage_1_artifact"],
        "holdouts_consulted": False,
        "outcomes": outcomes,
        "figure6_survivor_labels": survivors,
        "assessment": {
            "completed_candidate_count": len(outcomes),
            "registered_candidate_count": len(profile["survivor_order"]),
            "figure6_survivor_count": len(survivors),
            "advance_to_same_network_match": bool(complete and survivors),
        },
        "next_gate": profile["next_gate"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default=(
            "configs/calibration/"
            "trn_detector_hysteresis_figure6_prerequisite_v1.yaml"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    base_profile = yaml.safe_load(Path(profile["base_profile"]).read_text())
    stage_1 = yaml.safe_load(Path(profile["stage_1_artifact"]).read_text())
    registered_labels = [item["label"] for item in profile["survivor_order"]]
    if set(registered_labels) != set(stage_1["stage_1_survivor_labels"]):
        raise ValueError("Figure 6 candidates differ from Artifact 200 survivors")
    base = runtime_conventions_for_candidate(base_profile["candidate"])
    values = profile["protocol"]
    protocol = Figure6LearningProtocol(
        warmup_ms=float(values["warmup_ms"]),
        stimulus_ms=float(values["stimulus_ms"]),
        post_stimulus_ms=float(values["post_stimulus_ms"]),
        dt_ms=float(values["dt_ms"]),
        source_value=float(values["source_value"]),
        category_source_value=float(values["category_source_value"]),
        winning_layer4_index=int(values["winning_layer4_index"]),
        active_category_index=int(values["active_category_index"]),
        layer6ii_ahp_scale=float(values["layer6ii_ahp_scale"]),
        monitored_populations=MONITORED_POPULATIONS,
    )
    gate = profile["source_supported_gate"]
    expected_indices = tuple(int(value) for value in gate["relay_active_indices"])
    output = Path(args.output)
    outcomes: list[dict[str, Any]] = []
    for pair in profile["survivor_order"]:
        conventions = replace(
            base,
            trn_spike_event_coordinate=str(
                profile["fixed_choices"]["trn_spike_event_coordinate"]
            ),
            trn_spike_event_threshold_mV=float(pair["arm_mV"]),
            trn_spike_event_release_mV=float(pair["release_mV"]),
            trn_spike_event_proximal_blend_fraction=None,
        )
        learning = run_figure6_learning(
            conventions=conventions,
            protocol=protocol,
            record_relay_detector_diagnostics=True,
            brian=brian,
        )
        result = learning.result
        relay_indices = (result.population_spike_indices or {})["thalamic_relay"]
        relay_counts = {index: relay_indices.count(index) for index in expected_indices}
        threshold_cycles = _pairs(
            result.relay_detector_threshold_upcrossings_by_index
        )
        arm_cycles = _pairs(result.relay_detector_arm_transitions_by_index)
        release_cycles = _pairs(result.relay_detector_release_transitions_by_index)
        recruitment = assess_figure6_cortical_recruitment(result)
        timing = assess_figure6_top_down_timing(result)
        top_down_contrast = (
            result.top_down_wide.horizontal_orientation_contrast
            + result.top_down_narrow.horizontal_orientation_contrast
        )
        gates = {
            "relay_event_count": len(relay_indices) == int(gate["relay_event_count"]),
            "relay_active_indices": set(relay_indices) == set(expected_indices),
            "relay_events_per_active_index": all(
                count == int(gate["relay_events_per_active_index"])
                for count in relay_counts.values()
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
        outcome = {
            "pair": pair,
            "runtime_fingerprint": conventions.fingerprint,
            "protocol": protocol,
            "population_spikes": result.population_spikes,
            "relay_spike_indices": relay_indices,
            "relay_event_counts_by_index": relay_counts,
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
            "figure6_pass": all(gates.values()),
        }
        outcomes.append(outcome)
        _write_artifact(
            output=output,
            profile_path=args.profile,
            profile=profile,
            base_profile=base_profile,
            outcomes=outcomes,
            complete=False,
        )
        print(
            f"pair={pair['label']}: relay={len(relay_indices)} "
            f"trn={result.population_spikes['trn']} "
            f"chain={recruitment.feedforward_chain_complete} "
            f"pass={outcome['figure6_pass']}",
            flush=True,
        )
    _write_artifact(
        output=output,
        profile_path=args.profile,
        profile=profile,
        base_profile=base_profile,
        outcomes=outcomes,
        complete=True,
    )


if __name__ == "__main__":
    main()
