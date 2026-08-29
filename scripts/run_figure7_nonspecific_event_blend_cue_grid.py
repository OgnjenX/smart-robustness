"""Screen nonspecific detector blends for settled top-down cue safety."""

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


def _conventions(base: Any, fixed: dict[str, Any], blend: float) -> Any:
    return replace(
        base,
        trn_calcium_source_convention=str(fixed["trn_calcium_source_convention"]),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid-profile",
        default=(
            "configs/calibration/nonspecific_soma_proximal_event_blend_v1.yaml"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    grid_profile = yaml.safe_load(Path(args.grid_profile).read_text())
    base_profile = yaml.safe_load(Path(grid_profile["base_profile"]).read_text())
    base = runtime_conventions_for_candidate(base_profile["candidate"])
    fixed = grid_profile["fixed_choices"]
    blends = tuple(float(value) for value in grid_profile["dimension"]["grid"])
    bounds = tuple(float(value) for value in grid_profile["dimension"]["bounds"])
    if not blends or any(
        not np.isfinite(value) or not bounds[0] <= value <= bounds[1]
        for value in blends
    ):
        raise ValueError("nonspecific blend grid must remain within finite bounds")
    equilibration_ms = 20.0
    equilibration_tail_ms = float(
        grid_profile["stage_1_gate"]["quiescent_equilibration_tail_ms"]
    )
    current = float(fixed["top_down_current_pA"])
    outcomes = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for blend in blends:
        conventions = _conventions(base, fixed, blend)
        result = run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=current,
            use_paper_constrained_reference=True,
            conventions=conventions,
            duration_ms=1.0,
            top_down_cue_lead_ms=10.0,
            equilibration_ms=equilibration_ms,
            record_relay_diagnostics=False,
            brian=brian,
        )
        source_events = sum(index == 40 for index in result.cue_lead_category_spike_indices)
        off_source_events = sum(
            index != 40 for index in result.cue_lead_category_spike_indices
        )
        tail_start_ms = equilibration_ms - equilibration_tail_ms
        tail_events = sum(
            sum(value >= tail_start_ms for value in times)
            for times in (
                result.equilibration_nonspecific_spike_times_ms,
                result.equilibration_layer4_spike_times_ms,
                result.equilibration_relay_spike_times_ms,
                result.equilibration_trn_spike_times_ms,
                result.equilibration_category_spike_times_ms,
            )
        )
        stage_1_pass = bool(
            source_events
            and not off_source_events
            and not tail_events
            and not result.cue_lead_relay_spike_times_ms
            and not result.cue_lead_nonspecific_spike_times_ms
        )
        outcomes.append(
            {
                "nonspecific_spike_event_proximal_blend_fraction": blend,
                "runtime_fingerprint": conventions.fingerprint,
                "equilibration_tail_output_events": tail_events,
                "source_category_cue_events": source_events,
                "off_source_category_cue_events": off_source_events,
                "cue_lead_trn_events": len(result.cue_lead_trn_spike_times_ms),
                "cue_lead_relay_events": len(result.cue_lead_relay_spike_times_ms),
                "cue_lead_nonspecific_events": len(
                    result.cue_lead_nonspecific_spike_times_ms
                ),
                "stage_1_pass": stage_1_pass,
                "result": result,
            }
        )
        checkpoint = {
            "schema_version": 1,
            "id": output.stem,
            "status": "partial-grid-checkpoint",
            "completed_blend_fractions": [
                item["nonspecific_spike_event_proximal_blend_fraction"]
                for item in outcomes
            ],
            "outcomes": outcomes,
        }
        output.write_text(yaml.safe_dump(_plain(checkpoint), sort_keys=False))
        print(
            f"blend={blend:g}: cue_nonspecific="
            f"{len(result.cue_lead_nonspecific_spike_times_ms)} "
            f"stage_1={stage_1_pass}",
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
            "equilibration_ms": equilibration_ms,
            "quiescent_equilibration_tail_ms": equilibration_tail_ms,
            "top_down_cue_lead_ms": 10.0,
            "post_lead_bottom_up_ms": 1.0,
            "top_down_current_pA": current,
        },
        "stage_1_survivor_blend_fractions": [
            item["nonspecific_spike_event_proximal_blend_fraction"]
            for item in survivors
        ],
        "outcomes": outcomes,
        "next_gate": "Run 100-ms mismatch only for Stage-1 survivors.",
    }
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))


if __name__ == "__main__":
    main()
