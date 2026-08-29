"""Run mismatch only for arrival-aligned local TRN blend survivors."""

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

ALLOWED_MISMATCH_RELAY_INDICES = frozenset({40})


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
    parser.add_argument(
        "--stage-1-artifact",
        default=(
            "docs/validation-results/"
            "figure7-trn-event-blend-arrival-match-grid-190.yaml"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.grid_profile).read_text())
    stage_1 = yaml.safe_load(Path(args.stage_1_artifact).read_text())
    registered = {float(value) for value in profile["dimension"]["grid"]}
    blends = tuple(
        float(value) for value in stage_1["stage_1_survivor_blend_fractions"]
    )
    if not blends or not set(blends) <= registered:
        raise ValueError("Stage-1 survivors must be registered blend values")
    match_by_blend = {
        float(item["trn_spike_event_proximal_blend_fraction"]): item
        for item in stage_1["outcomes"]
    }
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
        match_item = match_by_blend[blend]
        if match_item["runtime_fingerprint"] != conventions.fingerprint:
            raise ValueError("saved match fingerprint does not match conventions")
        mismatch = run_figure7_condition(
            condition=MatchCondition.MISMATCH,
            top_down_current_pA=current,
            use_paper_constrained_reference=True,
            conventions=conventions,
            duration_ms=20.0,
            top_down_cue_lead_ms=lead,
            equilibration_ms=20.0,
            record_relay_diagnostics=False,
            brian=brian,
        )
        match = match_item["result"]
        gates = {
            "mismatch_relay_suppressed": frozenset(mismatch.relay_spike_indices)
            <= ALLOWED_MISMATCH_RELAY_INDICES,
            "trn_match_greater_than_mismatch": match_item["trn_events"]
            > len(mismatch.trn_spike_times_ms),
        }
        stage_2_pass = all(gates.values())
        outcomes.append(
            {
                "trn_spike_event_proximal_blend_fraction": blend,
                "runtime_fingerprint": conventions.fingerprint,
                "gates": gates,
                "stage_2_pass": stage_2_pass,
                "match": match,
                "mismatch": mismatch,
                "match_source_artifact": args.stage_1_artifact,
            }
        )
        checkpoint = {
            "schema_version": 1,
            "id": output.stem,
            "status": "partial-comparison-checkpoint",
            "completed_blend_fractions": [
                item["trn_spike_event_proximal_blend_fraction"] for item in outcomes
            ],
            "outcomes": outcomes,
        }
        output.write_text(yaml.safe_dump(_plain(checkpoint), sort_keys=False))
        print(
            f"blend={blend:g}: mismatch_relay="
            f"{len(mismatch.relay_spike_times_ms)} "
            f"active={sorted(set(mismatch.relay_spike_indices))} "
            f"trn={match_item['trn_events']}/"
            f"{len(mismatch.trn_spike_times_ms)} stage_2={stage_2_pass}",
            flush=True,
        )

    survivors = [item for item in outcomes if item["stage_2_pass"]]
    artifact = {
        "schema_version": 1,
        "id": output.stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "stage-2-survivors-found" if survivors else "no-stage-2-survivor",
        "grid_profile": args.grid_profile,
        "stage_1_artifact": args.stage_1_artifact,
        "protocol": {
            "conditions": ["match", "mismatch"],
            "duration_ms_after_bottom_up_onset": 20.0,
            "equilibration_ms": 20.0,
            "top_down_cue_lead_ms": lead,
            "top_down_current_pA": current,
            "match_trace_reused_from_stage_1": True,
        },
        "stage_2_survivor_blend_fractions": [
            item["trn_spike_event_proximal_blend_fraction"] for item in survivors
        ],
        "outcomes": outcomes,
        "next_gate": "Run a 100-ms pair only for Stage-2 survivors.",
    }
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))


if __name__ == "__main__":
    main()
