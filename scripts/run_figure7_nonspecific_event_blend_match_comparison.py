"""Compare 100-ms match against registered mismatch traces for blend survivors."""

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
        default=(
            "configs/calibration/nonspecific_soma_proximal_event_blend_v1.yaml"
        ),
    )
    parser.add_argument(
        "--stage-2a-artifact",
        default=(
            "docs/validation-results/"
            "figure7-nonspecific-blend-mismatch-grid-186.yaml"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    grid_profile = yaml.safe_load(Path(args.grid_profile).read_text())
    stage_2a = yaml.safe_load(Path(args.stage_2a_artifact).read_text())
    registered = {float(value) for value in grid_profile["dimension"]["grid"]}
    blends = tuple(
        float(value) for value in stage_2a["stage_2a_survivor_blend_fractions"]
    )
    if not blends or not set(blends) <= registered:
        raise ValueError("Stage-2a survivors must be nonempty registered grid values")
    mismatch_by_blend = {
        float(item["nonspecific_spike_event_proximal_blend_fraction"]): item
        for item in stage_2a["outcomes"]
    }
    base_profile = yaml.safe_load(Path(grid_profile["base_profile"]).read_text())
    base = runtime_conventions_for_candidate(base_profile["candidate"])
    fixed = grid_profile["fixed_choices"]
    current = float(fixed["top_down_current_pA"])
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
            trn_spike_event_proximal_blend_fraction=float(
                fixed["trn_spike_event_proximal_blend_fraction"]
            ),
            nonspecific_spike_event_proximal_blend_fraction=blend,
        )
        mismatch_item = mismatch_by_blend[blend]
        if mismatch_item["runtime_fingerprint"] != conventions.fingerprint:
            raise ValueError("mismatch trace fingerprint does not match conventions")
        match = run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=current,
            use_paper_constrained_reference=True,
            conventions=conventions,
            duration_ms=100.0,
            top_down_cue_lead_ms=0.0,
            equilibration_ms=20.0,
            record_relay_diagnostics=False,
            brian=brian,
        )
        mismatch = mismatch_item["result"]
        gates = {
            "relay_match_greater_than_mismatch": len(match.relay_spike_times_ms)
            > len(mismatch["relay_spike_times_ms"]),
            "trn_match_greater_than_mismatch": len(match.trn_spike_times_ms)
            > len(mismatch["trn_spike_times_ms"]),
            "nonspecific_mismatch_greater_than_match": len(
                mismatch["nonspecific_spike_times_ms"]
            )
            > len(match.nonspecific_spike_times_ms),
        }
        stage_2b_pass = all(gates.values())
        outcomes.append(
            {
                "nonspecific_spike_event_proximal_blend_fraction": blend,
                "runtime_fingerprint": conventions.fingerprint,
                "gates": gates,
                "stage_2b_pass": stage_2b_pass,
                "match": match,
                "mismatch": mismatch,
                "mismatch_source_artifact": args.stage_2a_artifact,
            }
        )
        checkpoint = {
            "schema_version": 1,
            "id": output.stem,
            "status": "partial-comparison-checkpoint",
            "completed_blend_fractions": [
                item["nonspecific_spike_event_proximal_blend_fraction"]
                for item in outcomes
            ],
            "outcomes": outcomes,
        }
        output.write_text(yaml.safe_dump(_plain(checkpoint), sort_keys=False))
        print(
            f"blend={blend:g}: "
            f"relay={len(match.relay_spike_times_ms)}/"
            f"{len(mismatch['relay_spike_times_ms'])} "
            f"trn={len(match.trn_spike_times_ms)}/"
            f"{len(mismatch['trn_spike_times_ms'])} "
            f"nonspecific={len(match.nonspecific_spike_times_ms)}/"
            f"{len(mismatch['nonspecific_spike_times_ms'])} "
            f"stage_2b={stage_2b_pass}",
            flush=True,
        )

    survivors = [item for item in outcomes if item["stage_2b_pass"]]
    artifact = {
        "schema_version": 1,
        "id": output.stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "stage-2b-survivors-found" if survivors else "no-stage-2b-survivor",
        "grid_profile": args.grid_profile,
        "stage_2a_artifact": args.stage_2a_artifact,
        "protocol": {
            "conditions": ["match", "mismatch"],
            "duration_ms": 100.0,
            "equilibration_ms": 20.0,
            "top_down_current_pA": current,
            "mismatch_trace_reused_from_stage_2a": True,
        },
        "stage_2b_survivor_blend_fractions": [
            item["nonspecific_spike_event_proximal_blend_fraction"]
            for item in survivors
        ],
        "outcomes": outcomes,
        "next_gate": "Run 300-ms pair only for Stage-2b survivors.",
    }
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))


if __name__ == "__main__":
    main()
