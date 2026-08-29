"""Align the first TRN event or relay inhibition with Figure 7 input onset."""

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


def _trn_available_at_onset(result) -> bool:
    return bool(result.cue_lead_trn_spike_times_ms) or (
        bool(result.trn_spike_times_ms) and result.trn_spike_times_ms[0] <= 0.1
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid-profile",
        default="configs/calibration/figure7_inhibitory_arrival_alignment_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.grid_profile).read_text())
    leads = tuple(float(value) for value in profile["dimension"]["grid"])
    if leads != (13.65, 13.75):
        raise ValueError("inhibitory-arrival grid differs from its registration")
    base_profile = yaml.safe_load(Path(profile["base_profile"]).read_text())
    base = runtime_conventions_for_candidate(base_profile["candidate"])
    fixed = profile["fixed_choices"]
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
        nonspecific_spike_event_proximal_blend_fraction=None,
    )
    current = float(fixed["top_down_current_pA"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for lead in leads:
        match = run_figure7_condition(
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
                index == 40 for index in match.cue_lead_category_spike_indices
            ),
            "off_source_category_events": not any(
                index != 40 for index in match.cue_lead_category_spike_indices
            ),
            "relay_events_during_lead": not match.cue_lead_relay_spike_times_ms,
            "nonspecific_events_during_lead": not (
                match.cue_lead_nonspecific_spike_times_ms
            ),
            "trn_event_at_or_before_bottom_up": _trn_available_at_onset(match),
        }
        match_gate = (
            frozenset(match.relay_spike_indices) == EXPECTED_MATCH_RELAY_INDICES
        )
        stage_1_pass = all(cue_gates.values()) and match_gate
        mismatch = None
        stage_2_gates = None
        stage_2_pass = False
        if stage_1_pass:
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
            stage_2_gates = {
                "relay_match_greater_than_mismatch": len(match.relay_spike_times_ms)
                > len(mismatch.relay_spike_times_ms),
                "trn_match_greater_than_mismatch": len(match.trn_spike_times_ms)
                > len(mismatch.trn_spike_times_ms),
            }
            stage_2_pass = all(stage_2_gates.values())
        outcomes.append(
            {
                "top_down_cue_lead_ms": lead,
                "runtime_fingerprint": conventions.fingerprint,
                "cue_gates": cue_gates,
                "match_relay_subset": match_gate,
                "stage_1_pass": stage_1_pass,
                "stage_2_gates": stage_2_gates,
                "stage_2_pass": stage_2_pass,
                "match": match,
                "mismatch": mismatch,
            }
        )
        checkpoint = {
            "schema_version": 1,
            "id": output.stem,
            "status": "partial-grid-checkpoint",
            "completed_leads_ms": [item["top_down_cue_lead_ms"] for item in outcomes],
            "outcomes": outcomes,
        }
        output.write_text(yaml.safe_dump(_plain(checkpoint), sort_keys=False))
        print(
            f"lead={lead:g}: cue_trn={len(match.cue_lead_trn_spike_times_ms)} "
            f"cue_relay={len(match.cue_lead_relay_spike_times_ms)} "
            f"match_relay={len(match.relay_spike_times_ms)} "
            f"match_trn={len(match.trn_spike_times_ms)} "
            f"stage_1={stage_1_pass} stage_2={stage_2_pass}",
            flush=True,
        )

    survivors = [item for item in outcomes if item["stage_2_pass"]]
    artifact = {
        "schema_version": 1,
        "id": output.stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "stage-2-survivors-found" if survivors else "no-stage-2-survivor",
        "grid_profile": args.grid_profile,
        "protocol": {
            "conditions": ["match", "mismatch_for_stage_1_survivors"],
            "duration_ms_after_bottom_up_onset": 20.0,
            "equilibration_ms": 20.0,
            "top_down_current_pA": current,
        },
        "stage_2_survivor_leads_ms": [
            item["top_down_cue_lead_ms"] for item in survivors
        ],
        "outcomes": outcomes,
        "next_gate": "Run a 100-ms pair only for Stage-2 survivors.",
    }
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))


if __name__ == "__main__":
    main()
