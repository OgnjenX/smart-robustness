"""Factor the official-source TRN calcium conflict at the Figure 7 bottleneck."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from smart_robustness.classic_sector import TrnCalciumSourceConvention
from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure7 import run_figure7_condition
from smart_robustness.validation.isolated_cells import (
    TrnRecruitmentProtocol,
    run_trn_recruitment_condition,
)


def _plain(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    return value


def _sampled_peak(rows: tuple[tuple[int, float, float], ...]) -> float:
    return max(row[2] for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="configs/calibration/figure6_kinness_equation27_transition_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration-ms", type=float, default=50.0)
    parser.add_argument("--equilibration-ms", type=float, default=20.0)
    parser.add_argument("--cue-lead-ms", type=float, default=10.0)
    parser.add_argument("--isolated-only", action="store_true")
    args = parser.parse_args()
    if args.duration_ms <= 45.0 and not args.isolated_only:
        raise ValueError("connected diagnostics require --duration-ms greater than 45")

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    base = runtime_conventions_for_candidate(profile["candidate"])
    top_down_current_pA = float(profile["candidate"]["top_down_current_pA"])
    isolated_protocol = TrnRecruitmentProtocol(pre_drive_ms=args.equilibration_ms)
    outcomes = []
    for calcium in TrnCalciumSourceConvention:
        conventions = replace(base, trn_calcium_source_convention=calcium.value)
        control = run_trn_recruitment_condition(
            driven=False,
            conventions=conventions,
            protocol=isolated_protocol,
            brian=brian,
        )
        driven = run_trn_recruitment_condition(
            driven=True,
            conventions=conventions,
            protocol=isolated_protocol,
            brian=brian,
        )
        control_quiescent = bool(control.finite and not control.post_drive_spike_times_ms)
        isolated_pass = bool(
            control_quiescent and driven.finite and driven.post_drive_spike_times_ms
        )
        outcome: dict[str, Any] = {
            "trn_calcium_source_convention": calcium.value,
            "runtime_fingerprint": conventions.fingerprint,
            "isolated_control": control,
            "isolated_driven": driven,
            "isolated_control_quiescent": control_quiescent,
            "isolated_quiescent_and_recruitable": isolated_pass,
        }
        if not args.isolated_only and control_quiescent:
            connected = run_figure7_condition(
                condition=MatchCondition.MATCH,
                top_down_current_pA=top_down_current_pA,
                use_paper_constrained_reference=True,
                conventions=conventions,
                duration_ms=args.duration_ms,
                top_down_cue_lead_ms=args.cue_lead_ms,
                equilibration_ms=args.equilibration_ms,
                record_relay_diagnostics=True,
                brian=brian,
            )
            source_cue_events = sum(
                index == 40 for index in connected.cue_lead_category_spike_indices
            )
            off_source_cue_events = sum(
                index != 40 for index in connected.cue_lead_category_spike_indices
            )
            connected_pass = bool(
                source_cue_events
                and not off_source_cue_events
                and not connected.cue_lead_trn_spike_times_ms
                and not connected.cue_lead_relay_spike_times_ms
                and connected.trn_spike_times_ms
            )
            outcome.update(
                {
                    "connected_match": connected,
                    "source_category_cue_events": source_cue_events,
                    "off_source_category_cue_events": off_source_cue_events,
                    "post_bottom_up_trn_events": len(connected.trn_spike_times_ms),
                    "post_bottom_up_active_trn_cells": len(
                        set(connected.trn_spike_indices)
                    ),
                    "post_bottom_up_relay_events": len(connected.relay_spike_times_ms),
                    "sampled_trn_soma_peak_mV": _sampled_peak(
                        connected.trn_soma_voltage_range_mV_by_index
                    ),
                    "sampled_trn_proximal_peak_mV": _sampled_peak(
                        connected.trn_proximal_voltage_range_mV_by_index
                    ),
                    "connected_causal_recruitment_pass": connected_pass,
                }
            )
        elif not args.isolated_only:
            outcome["connected_skipped_reason"] = (
                "post-equilibration autonomous isolated TRN output"
            )
        outcomes.append(outcome)
        print(
            f"{calcium.value}: control_quiescent={control_quiescent} "
            f"isolated={isolated_pass} "
            f"connected_trn={outcome.get('post_bottom_up_trn_events', 'skipped')}",
            flush=True,
        )

    connected = [item for item in outcomes if "connected_match" in item]
    survivors = [
        item for item in connected if item["connected_causal_recruitment_pass"]
    ]
    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": (
            "isolated-source-factor-screen"
            if args.isolated_only
            else (
                "connected-causal-survivor-found"
                if survivors
                else "no-connected-causal-survivor"
            )
        ),
        "claim": (
            "Whether the Table 3 versus SMART.nml TRN soma-calcium-channel and "
            "calcium-reversal conflicts explain the missing Figure 7 somatic event"
        ),
        "profile": args.profile,
        "base_candidate_fingerprint": profile["candidate_fingerprint"],
        "source_values": {
            "table3_soma_calcium_density_mS_cm2": None,
            "smart_nml_soma_calcium_density_mS_cm2": 100.0,
            "table3_calcium_reversal_mV": 180.0,
            "smart_nml_calcium_reversal_mV": 120.0,
        },
        "protocol": {
            "isolated_pre_drive_ms": isolated_protocol.pre_drive_ms,
            "isolated_drive_ms": isolated_protocol.drive_ms,
            "connected_condition": "match",
            "duration_ms": args.duration_ms,
            "equilibration_ms": args.equilibration_ms,
            "cue_lead_ms": args.cue_lead_ms,
            "top_down_current_pA": top_down_current_pA,
            "paper_constrained_learned_state": True,
        },
        "promotion_gate": {
            "post_equilibration_isolated_control_events": 0,
            "selective_source_category_cue": True,
            "cue_lead_trn_events": 0,
            "cue_lead_relay_events": 0,
            "post_bottom_up_trn_events": ">0",
            "note": "A survivor still requires a paired mismatch run and all Figure 6 gates.",
        },
        "assessment": {
            "connected_candidates": len(connected),
            "connected_causal_survivors": len(survivors),
            "autonomous_candidates_rejected_before_network": sum(
                not item["isolated_control_quiescent"] for item in outcomes
            ),
        },
        "outcomes": outcomes,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))


if __name__ == "__main__":
    main()
