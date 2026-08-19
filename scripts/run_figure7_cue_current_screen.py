"""Screen the unreported Figure 7 category-current amplitude before BU onset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

import yaml

from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure7 import run_figure7_condition


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="configs/calibration/figure7_trn_kinness_event_coordinate_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--currents-pA",
        type=float,
        nargs="+",
        default=(100.0, 200.0, 300.0, 400.0, 500.0, 600.0),
    )
    parser.add_argument("--cue-lead-ms", type=float, default=10.0)
    parser.add_argument("--equilibration-ms", type=float, default=20.0)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    conventions = runtime_conventions_for_candidate(profile["candidate"])
    outcomes = []
    for current_pA in args.currents_pA:
        result = run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=current_pA,
            use_paper_constrained_reference=True,
            conventions=conventions,
            duration_ms=1.0,
            top_down_cue_lead_ms=args.cue_lead_ms,
            equilibration_ms=args.equilibration_ms,
            record_relay_diagnostics=False,
            brian=brian,
        )
        lead_category = tuple(
            zip(
                result.cue_lead_category_spike_indices,
                result.cue_lead_category_spike_times_ms,
                strict=True,
            )
        )
        lead_trn = result.cue_lead_trn_spike_times_ms
        lead_relay = result.cue_lead_relay_spike_times_ms
        source_recruited = any(index == 40 for index, _ in lead_category)
        off_source_indices = sorted({index for index, _ in lead_category if index != 40})
        outcomes.append(
            {
                "top_down_current_pA": current_pA,
                "source_recruited_during_lead": source_recruited,
                "off_source_category_indices_during_lead": off_source_indices,
                "category_events_during_lead": len(lead_category),
                "trn_events_during_lead": len(lead_trn),
                "relay_events_during_lead": len(lead_relay),
                "result": _plain(asdict(result)),
            }
        )
        print(
            f"{current_pA:g} pA: source={source_recruited} "
            f"off_source={len(off_source_indices)} trn={len(lead_trn)}",
            flush=True,
        )

    selective = [
        item
        for item in outcomes
        if item["source_recruited_during_lead"]
        and not item["off_source_category_indices_during_lead"]
    ]
    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "selective-cue-found" if selective else "no-selective-cue-found",
        "profile": args.profile,
        "candidate_fingerprint": profile["candidate_fingerprint"],
        "runtime_fingerprint": conventions.fingerprint,
        "protocol": {
            "cue_lead_ms": args.cue_lead_ms,
            "equilibration_ms": args.equilibration_ms,
            "post_lead_bottom_up_ms": 1.0,
            "paper_constrained_learned_state": True,
        },
        "selection_gate": {
            "source_index": 40,
            "source_event_during_lead": True,
            "off_source_category_events_during_lead": 0,
        },
        "selective_currents_pA": [item["top_down_current_pA"] for item in selective],
        "outcomes": outcomes,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(artifact, sort_keys=False))


if __name__ == "__main__":
    main()
