"""Screen the registered local TRN event-transfer grid under arrival alignment."""

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
        default="configs/calibration/trn_event_blend_arrival_interaction_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.grid_profile).read_text())
    blends = tuple(float(value) for value in profile["dimension"]["grid"])
    if blends != (0.4, 0.42, 0.44, 0.46, 0.48, 0.49, 0.5):
        raise ValueError("local event-transfer grid differs from its registration")
    base_profile = yaml.safe_load(Path(profile["base_profile"]).read_text())
    base = runtime_conventions_for_candidate(base_profile["candidate"])
    fixed = profile["fixed_choices"]
    current = float(fixed["top_down_current_pA"])
    lead = float(fixed["top_down_cue_lead_ms"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for blend in blends:
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
            trn_spike_event_proximal_blend_fraction=blend,
            nonspecific_spike_event_proximal_blend_fraction=None,
        )
        result = run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=current,
            use_paper_constrained_reference=True,
            conventions=conventions,
            duration_ms=20.0,
            top_down_cue_lead_ms=lead,
            equilibration_ms=20.0,
            record_relay_diagnostics=False,
            brian=brian,
        )
        cue_gates = {
            "source_category_event_before_bottom_up": any(
                index == 40 for index in result.cue_lead_category_spike_indices
            ),
            "off_source_category_events": not any(
                index != 40 for index in result.cue_lead_category_spike_indices
            ),
            "relay_events_during_lead": not result.cue_lead_relay_spike_times_ms,
            "nonspecific_events_during_lead": not (
                result.cue_lead_nonspecific_spike_times_ms
            ),
        }
        match_gates = {
            "match_relay_subset": frozenset(result.relay_spike_indices)
            == EXPECTED_MATCH_RELAY_INDICES,
            "match_trn_nonzero": bool(result.trn_spike_times_ms),
        }
        stage_1_pass = all(cue_gates.values()) and all(match_gates.values())
        outcomes.append(
            {
                "trn_spike_event_proximal_blend_fraction": blend,
                "runtime_fingerprint": conventions.fingerprint,
                "cue_gates": cue_gates,
                "match_gates": match_gates,
                "relay_events": len(result.relay_spike_times_ms),
                "active_relay_indices": sorted(set(result.relay_spike_indices)),
                "trn_events": len(result.trn_spike_times_ms),
                "stage_1_pass": stage_1_pass,
                "result": result,
            }
        )
        checkpoint = {
            "schema_version": 1,
            "id": output.stem,
            "status": "partial-grid-checkpoint",
            "completed_blend_fractions": [
                item["trn_spike_event_proximal_blend_fraction"] for item in outcomes
            ],
            "outcomes": outcomes,
        }
        output.write_text(yaml.safe_dump(_plain(checkpoint), sort_keys=False))
        print(
            f"blend={blend:g}: relay={len(result.relay_spike_times_ms)} "
            f"active={sorted(set(result.relay_spike_indices))} "
            f"trn={len(result.trn_spike_times_ms)} stage_1={stage_1_pass}",
            flush=True,
        )

    survivors = [item for item in outcomes if item["stage_1_pass"]]
    artifact = {
        "schema_version": 1,
        "id": output.stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "stage-1-survivors-found" if survivors else "no-stage-1-survivor",
        "grid_profile": args.grid_profile,
        "protocol": {
            "condition": "match",
            "duration_ms_after_bottom_up_onset": 20.0,
            "equilibration_ms": 20.0,
            "top_down_cue_lead_ms": lead,
            "top_down_current_pA": current,
        },
        "stage_1_survivor_blend_fractions": [
            item["trn_spike_event_proximal_blend_fraction"] for item in survivors
        ],
        "outcomes": outcomes,
        "next_gate": "Run fresh mismatch only for Stage-1 survivors.",
    }
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))


if __name__ == "__main__":
    main()
