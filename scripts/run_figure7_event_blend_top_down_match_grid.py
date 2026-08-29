"""Screen cue-safe top-down currents on the 50-ms Figure 7 match gate."""

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
        default="configs/calibration/figure7_event_blend_top_down_grid_v1.yaml",
    )
    parser.add_argument(
        "--stage-1-artifact",
        default="docs/validation-results/figure7-event-blend-current-cue-grid-182.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    grid_profile = yaml.safe_load(Path(args.grid_profile).read_text())
    stage_1 = yaml.safe_load(Path(args.stage_1_artifact).read_text())
    registered = {float(value) for value in grid_profile["dimension"]["grid"]}
    currents = tuple(float(value) for value in stage_1["stage_1_survivor_currents_pA"])
    if not currents or not set(currents) <= registered:
        raise ValueError("Stage-1 survivors must be nonempty registered grid values")
    base_profile = yaml.safe_load(Path(grid_profile["base_profile"]).read_text())
    base = runtime_conventions_for_candidate(base_profile["candidate"])
    fixed = grid_profile["fixed_choices"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for current in currents:
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
            trn_spike_event_proximal_blend_fraction=float(
                fixed["trn_spike_event_proximal_blend_fraction"]
            ),
        )
        result = run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=current,
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
                "top_down_current_pA": current,
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
            "completed_currents_pA": [item["top_down_current_pA"] for item in outcomes],
            "outcomes": outcomes,
        }
        output.write_text(yaml.safe_dump(_plain(checkpoint), sort_keys=False))
        print(
            f"current={current:g} pA: relay={len(result.relay_spike_times_ms)} "
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
        },
        "stage_2a_survivor_currents_pA": [
            item["top_down_current_pA"] for item in survivors
        ],
        "outcomes": outcomes,
        "next_gate": "Run the 300-ms match/mismatch pair only for Stage-2a survivors.",
    }
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))


if __name__ == "__main__":
    main()
