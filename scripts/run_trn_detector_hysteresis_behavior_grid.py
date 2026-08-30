"""Screen registered somatic TRN detector arm/release pairs."""

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
from smart_robustness.validation.isolated_cells import (
    TrnRecruitmentProtocol,
    run_trn_detector_cycle_condition,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="configs/calibration/trn_detector_hysteresis_behavior_grid_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    base_profile = yaml.safe_load(Path(profile["base_profile"]).read_text())
    base = runtime_conventions_for_candidate(base_profile["candidate"])
    fixed = profile["fixed_choices"]
    values = profile["stage_1_protocol"]
    base_protocol = TrnRecruitmentProtocol(
        pre_drive_ms=float(values["pre_drive_ms"]),
        drive_ms=float(values["drive_ms"]),
        dt_ms=float(values["dt_ms"]),
        relay_ampa_gate=float(values["relay_ampa_gate"]),
        layer6ii_ampa_gate=float(values["layer6ii_ampa_gate"]),
        layer6ii_nmda_gate=float(values["layer6ii_nmda_gate"]),
    )
    recovery_ms = float(values["post_drive_recovery_ms"])
    multipliers = [float(value) for value in values["drive_multipliers"]]
    pairs = [
        profile["dimension"]["source_pair"],
        *profile["dimension"]["calibrated_pairs"],
    ]
    outcomes = []
    for pair in pairs:
        conventions = replace(
            base,
            trn_potassium_convention=str(fixed["trn_potassium_convention"]),
            trn_calcium_source_convention=str(
                fixed["trn_calcium_source_convention"]
            ),
            trn_dendritic_calcium_density_mS_cm2=float(
                fixed["trn_dendritic_calcium_density_mS_cm2"]
            ),
            trn_spike_event_threshold_mV=float(pair["arm_mV"]),
            trn_spike_event_release_mV=float(pair["release_mV"]),
            trn_spike_event_proximal_blend_fraction=None,
        )
        control = run_trn_detector_cycle_condition(
            driven=False,
            conventions=conventions,
            protocol=base_protocol,
            post_drive_recovery_ms=recovery_ms,
            brian=brian,
        )
        driven_outcomes = []
        for multiplier in multipliers:
            protocol = replace(base_protocol, drive_multiplier=multiplier)
            driven = run_trn_detector_cycle_condition(
                driven=True,
                conventions=conventions,
                protocol=protocol,
                post_drive_recovery_ms=recovery_ms,
                brian=brian,
            )
            driven_outcomes.append(
                {
                    "drive_multiplier": multiplier,
                    "result": driven,
                    "fresh_detector_cycle_pass": driven.fresh_detector_cycle_pass,
                }
            )
            print(
                f"pair={pair['label']} multiplier={multiplier:g}: "
                f"events={len(driven.post_stimulus_spike_times_ms)} "
                f"up={driven.threshold_upcrossings} "
                f"arm={driven.arm_transitions} "
                f"release={driven.release_transitions} "
                f"pass={driven.fresh_detector_cycle_pass}",
                flush=True,
            )
        control_pass = bool(
            control.finite
            and not control.post_stimulus_spike_times_ms
            and control.threshold_upcrossings == 0
            and control.arm_transitions == 0
        )
        stage_1_pass = bool(
            control_pass
            and any(item["fresh_detector_cycle_pass"] for item in driven_outcomes)
        )
        outcomes.append(
            {
                "pair": pair,
                "runtime_fingerprint": conventions.fingerprint,
                "control": control,
                "control_pass": control_pass,
                "driven_outcomes": driven_outcomes,
                "stage_1_pass": stage_1_pass,
            }
        )
    survivors = [item for item in outcomes if item["stage_1_pass"]]
    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "stage-1-survivors-found" if survivors else "no-stage-1-survivor",
        "profile": args.profile,
        "base_candidate_fingerprint": base_profile["candidate_fingerprint"],
        "protocol": base_protocol,
        "post_drive_recovery_ms": recovery_ms,
        "drive_multipliers": multipliers,
        "stage_1_survivor_labels": [item["pair"]["label"] for item in survivors],
        "outcomes": outcomes,
        "assessment": {
            "stage_1_survivor_count": len(survivors),
            "source_pair_survives": outcomes[0]["stage_1_pass"],
            "all_controls_pass": all(item["control_pass"] for item in outcomes),
            "startup_latched_release_counts_as_pass": False,
            "advance_to_connected_prerequisite": bool(survivors),
        },
        "next_gate": profile["next_gate"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))


if __name__ == "__main__":
    main()
