"""Reproduce the non-TRN isolated-cell gates in the calibration contract."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import yaml

from smart_robustness.classic_sector import figure6_runtime_conventions
from smart_robustness.validation.calibration import load_calibration_contract
from smart_robustness.validation.isolated_cells import (
    Figure8Protocol,
    Figure19Protocol,
    Layer5PropagationProtocol,
    assess_figure19_kernel,
    run_figure8_source_candidate,
    run_figure19_kernel_condition,
    run_layer5_propagation_condition,
)

FIGURE8_LEAK_DENSITIES = (0.01, 0.03, 0.05, 0.1, 0.2)
FIGURE8_CAPACITANCES = (0.5, 1.0, 1.5, 2.0)
AXIAL_CONVENTIONS = (
    "kinness_serialized_edge",
    "kinness_2008",
    "paper_literal",
    "symmetric_cable",
)


def _run_figure19_profile(*, name: str, protocol: Figure19Protocol, brian) -> dict:
    traces = {
        "control": run_figure19_kernel_condition(
            spike_count=0, acetylcholine=False, protocol=protocol, brian=brian
        ),
        "one_spike": run_figure19_kernel_condition(
            spike_count=1, acetylcholine=False, protocol=protocol, brian=brian
        ),
        "two_spike": run_figure19_kernel_condition(
            spike_count=2, acetylcholine=False, protocol=protocol, brian=brian
        ),
        "two_spike_ach": run_figure19_kernel_condition(
            spike_count=2, acetylcholine=True, protocol=protocol, brian=brian
        ),
    }
    assessment = assess_figure19_kernel(
        traces["control"],
        traces["one_spike"],
        traces["two_spike"],
        traces["two_spike_ach"],
    )
    return {
        "name": name,
        "protocol": asdict(protocol),
        "assessment": asdict(assessment),
        "reproduced": assessment.reproduced,
    }


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

    figure8_protocol = Figure8Protocol(depolarized_hold_mV=-62.3)
    figure8_outcomes = []
    for leak_density in FIGURE8_LEAK_DENSITIES:
        for capacitance in FIGURE8_CAPACITANCES:
            candidate = run_figure8_source_candidate(
                leak_density_mS_cm2=leak_density,
                specific_capacitance_uF_cm2=capacitance,
                protocol=figure8_protocol,
                brian=brian,
            )
            figure8_outcomes.append(
                {
                    "leak_density_mS_cm2": leak_density,
                    "specific_capacitance_uF_cm2": capacitance,
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
                f"Figure 8 leak={leak_density:g} C={capacitance:g}: "
                f"tonic={candidate.assessment.tonic_pass} "
                f"burst={candidate.assessment.burst_pass}",
                flush=True,
            )

    figure19_profiles = [
        _run_figure19_profile(name="paper", protocol=Figure19Protocol(), brian=brian),
        _run_figure19_profile(
            name="modeldb_112923",
            protocol=Figure19Protocol(
                ahp_event_weight=4.5, ahp_convention="modeldb_112923"
            ),
            brian=brian,
        ),
    ]

    propagation_protocol = Layer5PropagationProtocol(dt_ms=0.01)
    propagation_outcomes = []
    base = figure6_runtime_conventions()
    for axial_convention in AXIAL_CONVENTIONS:
        result = run_layer5_propagation_condition(
            conventions=replace(base, axial_convention=axial_convention),
            protocol=propagation_protocol,
            brian=brian,
        )
        propagation_outcomes.append(
            {
                "axial_convention": axial_convention,
                "result": asdict(result),
                "propagation_pass": result.finite and bool(result.post_drive_spike_times_ms),
            }
        )
        print(
            f"Layer 5 axial={axial_convention}: "
            f"events={len(result.post_drive_spike_times_ms)} finite={result.finite}",
            flush=True,
        )

    figure8_survivors = [item for item in figure8_outcomes if item["reproduced"]]
    propagation_survivors = [
        item for item in propagation_outcomes if item["propagation_pass"]
    ]
    artifact = {
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "failed-no-complete-cellular-survivor",
        "contract": str(args.contract),
        "contract_fingerprint": contract.fingerprint,
        "holdouts_consulted": False,
        "figure8": {
            "source": "Grossberg and Versace Figure 8 and ModelDB Ca_rebound.xml",
            "protocol": asdict(figure8_protocol),
            "candidate_count": len(figure8_outcomes),
            "survivor_count": len(figure8_survivors),
            "outcomes": figure8_outcomes,
        },
        "figure19": {
            "source": "Grossberg and Versace Figure 19 and Layer_5_and_Maynert_AHP_ACh.nml",
            "profiles": figure19_profiles,
        },
        "layer5_propagation": {
            "source": "Grossberg and Versace Figure 10b and Equation 2",
            "protocol": asdict(propagation_protocol),
            "survivor_count": len(propagation_survivors),
            "outcomes": propagation_outcomes,
        },
        "promotion": {
            "allowed": False,
            "reason": "the independently required TRN Stage A gate has no survivor",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(artifact, sort_keys=False))


if __name__ == "__main__":
    main()
