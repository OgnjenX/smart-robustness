"""Reassess Figure 6 using only source-supported qualitative/numeric gates."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-artifact",
        default=(
            "docs/validation-results/figure6-kinness-equation27-transition-153.yaml"
        ),
    )
    parser.add_argument(
        "--feasibility-artifact",
        default=(
            "docs/validation-results/figure6-published-amplitude-feasibility-155.yaml"
        ),
    )
    parser.add_argument(
        "--diagnostic-artifact",
        default=(
            "docs/validation-results/"
            "figure6-kinness-equation27-figure7-diagnostic-156.yaml"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    training = yaml.safe_load(Path(args.training_artifact).read_text())
    feasibility = yaml.safe_load(Path(args.feasibility_artifact).read_text())
    diagnostic = yaml.safe_load(Path(args.diagnostic_artifact).read_text())
    figure6 = training["figure6"]
    diagnostic_figure6 = diagnostic["figure6"]
    if diagnostic["candidate_fingerprint"] != training["candidate_fingerprint"]:
        raise ValueError("diagnostic and training candidate fingerprints differ")
    if diagnostic["runtime_fingerprint"] != training["runtime_fingerprint"]:
        raise ValueError("diagnostic and training runtime fingerprints differ")
    if diagnostic_figure6["population_spikes"] != figure6["population_spikes"]:
        raise ValueError("diagnostic and training Figure 6 spike counts differ")
    relay_indices = diagnostic["figure7"]["conditions"]["match"][
        "relay_spike_indices"
    ]
    relay_confined = (
        len(relay_indices) == 20
        and set(relay_indices) == {38, 39, 40, 41, 42}
        and all(relay_indices.count(index) == 4 for index in (38, 39, 40, 41, 42))
    )
    combined_contrast = (
        figure6["top_down_wide"]["horizontal_orientation_contrast"]
        + figure6["top_down_narrow"]["horizontal_orientation_contrast"]
    )
    gates = {
        "relay_confined_to_five_horizontal_cells_at_40_hz": relay_confined,
        "feedforward_chain_complete": figure6["feedforward_chain_complete"],
        "causal_pair_in_learning_window": figure6["causal_pair_in_learning_window"],
        "bottom_up_horizontal_orientation": figure6["bottom_up_oriented"],
        "top_down_horizontal_contrast_at_least_0_01": combined_contrast >= 0.01,
    }
    qualitative_pass = all(gates.values())
    adaptive_bound = feasibility["derivation"][
        "two_adaptive_components_bound"
    ]
    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": (
            "qualitative-figure6-reproduced"
            if qualitative_pass
            else "qualitative-figure6-failed"
        ),
        "training_artifact": args.training_artifact,
        "feasibility_artifact": args.feasibility_artifact,
        "diagnostic_artifact": args.diagnostic_artifact,
        "candidate_fingerprint": training["candidate_fingerprint"],
        "runtime_fingerprint": training["runtime_fingerprint"],
        "source_strength_correction": {
            "absolute_map_amplitude": "not-identifiable",
            "reason": (
                "The paper publishes a raster/colorbar but no numeric matrix; "
                "the printed learning law's proven two-component bound is "
                f"{adaptive_bound}."
            ),
            "historical_2_0_peak_gate": "retracted-unsupported",
            "normalized_spatial_shape": "verifiable",
        },
        "observed": {
            "combined_adaptive_final_peak": feasibility[
                "corrected_kinness_candidate"
            ]["combined_adaptive_final_peak_approximate"],
            "combined_top_down_horizontal_contrast": combined_contrast,
        },
        "gates": gates,
        "assessment": {
            "qualitative_figure6_reproduced": qualitative_pass,
            "exact_absolute_amplitude_reproduced": None,
            "figure7_eligible_as_source_strength_prerequisite": qualitative_pass,
            "candidate_promoted_for_downstream_calibration": qualitative_pass,
        },
        "scope": (
            "This promotion validates the published relay rate/confinement, "
            "cortical recruitment, causal timing, and horizontal map shapes. "
            "It does not claim recovery of an unpublished numeric weight matrix."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(artifact, sort_keys=False))


if __name__ == "__main__":
    main()
