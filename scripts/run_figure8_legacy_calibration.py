"""Calibrate the unresolved version-1 Figure 8 calcium conductance unit."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

from smart_robustness.validation.calibration import load_calibration_contract
from smart_robustness.validation.isolated_cells import Figure8Protocol, run_figure8_source_candidate

# Endpoints are the archived literal mS/cm2 interpretation and the legacy
# microSiemens-to-mS interpretation. Interior values are fixed in advance on a
# sparse logarithmic grid because the version-1 unit conversion is unrecovered.
CALCIUM_DENSITY_GRID_MSIEMENS_CM2 = (
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--contract", default="configs/calibration/classic_uncertainty_space.yaml"
    )
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    contract = load_calibration_contract(args.contract)
    protocol = Figure8Protocol(depolarized_hold_mV=-62.3)
    outcomes = []
    for calcium_density in CALCIUM_DENSITY_GRID_MSIEMENS_CM2:
        candidate = run_figure8_source_candidate(
            leak_density_mS_cm2=0.1,
            specific_capacitance_uF_cm2=1.0,
            calcium_density_mS_cm2=calcium_density,
            protocol=protocol,
            brian=brian,
        )
        outcomes.append(
            {
                "calcium_density_mS_cm2": calcium_density,
                "tonic_spike_times_ms": [
                    float(value) for value in candidate.tonic.spike_times_ms
                ],
                "burst_spike_times_ms": [
                    float(value) for value in candidate.burst.spike_times_ms
                ],
                "assessment": asdict(candidate.assessment),
                "reproduced": candidate.assessment.reproduced,
            }
        )
        print(
            f"gCa={calcium_density:g}: tonic={candidate.assessment.tonic_pass} "
            f"({candidate.assessment.tonic_spike_count}) "
            f"burst={candidate.assessment.burst_pass} "
            f"({candidate.assessment.burst_spike_count})",
            flush=True,
        )

    survivors = [item for item in outcomes if item["reproduced"]]
    artifact = {
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "passing" if survivors else "failed-no-survivor",
        "claim": "An unresolved version-1 calcium unit reproduces Figure 8 tonic and burst",
        "contract": str(args.contract),
        "contract_fingerprint": contract.fingerprint,
        "training_target": "figure8_relay_modes",
        "holdouts_consulted": False,
        "source_ambiguity": {
            "serialized_g_bar": 250,
            "literal_interpretation_mS_cm2": 250.0,
            "legacy_microSiemens_interpretation_mS_cm2": 0.25,
            "unit_mapping": "not recovered from version-1 libkinmaze",
        },
        "fixed_protocol": asdict(protocol),
        "fixed_passive_defaults": {
            "leak_density_mS_cm2": 0.1,
            "specific_capacitance_uF_cm2": 1.0,
        },
        "candidate_count": len(outcomes),
        "survivor_count": len(survivors),
        "survivor_calcium_densities_mS_cm2": [
            item["calcium_density_mS_cm2"] for item in survivors
        ],
        "outcomes": outcomes,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(artifact, sort_keys=False))


if __name__ == "__main__":
    main()
