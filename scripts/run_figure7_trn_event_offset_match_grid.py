"""Screen cue-safe TRN event offsets on simultaneous Figure 7 match."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure7 import run_figure7_condition

EXPECTED_MATCH_RELAY_INDICES = frozenset({38, 39, 40, 41, 42})


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid-profile",
        default="configs/calibration/trn_event_offset_behavior_grid_v1.yaml",
    )
    parser.add_argument(
        "--stage-1-artifact",
        default="docs/validation-results/figure7-trn-event-offset-cue-grid-175.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    grid_profile = yaml.safe_load(Path(args.grid_profile).read_text())
    stage_1 = yaml.safe_load(Path(args.stage_1_artifact).read_text())
    registered = {float(value) for value in grid_profile["dimension"]["grid"]}
    offsets = tuple(
        float(value) for value in stage_1["stage_1_survivor_offsets_mV"]
    )
    if not offsets or not set(offsets) <= registered:
        raise ValueError("Stage-1 survivors must be nonempty registered grid values")
    base_profile_path = str(grid_profile["base_profile"])
    base_profile = yaml.safe_load(Path(base_profile_path).read_text())
    base = runtime_conventions_for_candidate(base_profile["candidate"])
    fixed = grid_profile["fixed_choices"]
    top_down_current_pA = float(base_profile["candidate"]["top_down_current_pA"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for offset in offsets:
        conventions = replace(
            base,
            trn_calcium_source_convention=str(
                fixed["trn_calcium_source_convention"]
            ),
            trn_dendritic_calcium_density_mS_cm2=float(
                fixed["trn_dendritic_calcium_density_mS_cm2"]
            ),
            trn_soma_proximal_axial_conductance_scale=float(
                fixed["trn_soma_proximal_axial_conductance_scale"]
            ),
            trn_spike_event_voltage_offset_mV=offset,
        )
        result = run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=top_down_current_pA,
            use_paper_constrained_reference=True,
            conventions=conventions,
            duration_ms=50.0,
            top_down_cue_lead_ms=0.0,
            equilibration_ms=20.0,
            record_relay_diagnostics=False,
            brian=brian,
        )
        active_relay_indices = frozenset(result.relay_spike_indices)
        stage_2a_pass = bool(
            active_relay_indices == EXPECTED_MATCH_RELAY_INDICES
            and result.trn_spike_times_ms
        )
        outcomes.append(
            {
                "trn_spike_event_voltage_offset_mV": offset,
                "physical_arm_threshold_mV": 30.0 - offset,
                "physical_release_threshold_mV": -offset,
                "runtime_fingerprint": conventions.fingerprint,
                "active_relay_indices": sorted(active_relay_indices),
                "relay_events": len(result.relay_spike_times_ms),
                "trn_events": len(result.trn_spike_times_ms),
                "active_trn_cells": len(set(result.trn_spike_indices)),
                "nonspecific_events": len(result.nonspecific_spike_times_ms),
                "stage_2a_pass": stage_2a_pass,
                "result": result,
            }
        )
        checkpoint = {
            "schema_version": 1,
            "id": output.stem,
            "status": "partial-grid-checkpoint",
            "completed_offsets_mV": [
                item["trn_spike_event_voltage_offset_mV"] for item in outcomes
            ],
            "outcomes": outcomes,
        }
        output.write_text(yaml.safe_dump(_plain(checkpoint), sort_keys=False))
        print(
            f"offset={offset:g} mV: relay={len(result.relay_spike_times_ms)} "
            f"active={sorted(active_relay_indices)} trn={len(result.trn_spike_times_ms)} "
            f"stage_2a={stage_2a_pass}",
            flush=True,
        )

    survivors = [item for item in outcomes if item["stage_2a_pass"]]
    artifact = {
        "schema_version": 1,
        "id": output.stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "stage-2a-survivors-found" if survivors else "no-stage-2a-survivor",
        "grid_profile": args.grid_profile,
        "stage_1_artifact": args.stage_1_artifact,
        "protocol": {
            "condition": "match",
            "duration_ms": 50.0,
            "equilibration_ms": 20.0,
            "top_down_cue_lead_ms": 0.0,
            "timing": "simultaneous bottom-up/top-down onset",
        },
        "stage_2a_survivor_offsets_mV": [
            item["trn_spike_event_voltage_offset_mV"] for item in survivors
        ],
        "outcomes": outcomes,
        "next_gate": "Run mismatch only for Stage-2a survivors.",
    }
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))


if __name__ == "__main__":
    main()
