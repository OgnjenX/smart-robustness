"""Run the preregistered isolated projection-025 conductance bracket."""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import yaml

from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.layer6i_replay import run_layer6i_replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()

    registration_path = Path(args.registration)
    registration = yaml.safe_load(registration_path.read_text())
    trace_path = Path(registration["trace"]["path"])
    if sha256(trace_path.read_bytes()).hexdigest() != registration["trace"]["sha256"]:
        raise ValueError("replay trace differs from preregistered Artifact 638 trace")

    profile = yaml.safe_load(Path(registration["profile"]).read_text())
    training_profile = yaml.safe_load(Path(profile["training_profile"]).read_text())
    base_profile = yaml.safe_load(Path(training_profile["base_profile"]).read_text())
    conventions = replace(
        runtime_conventions_for_candidate(base_profile["candidate"]),
        **training_profile["runtime_overrides"],
        **profile["runtime_overrides"],
    )
    if conventions.fingerprint != registration["runtime_fingerprint"]:
        raise ValueError("registration runtime differs from executable profile")

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    scale_results: list[dict[str, object]] = []
    for scale in registration["projection025_conductance_scales"]:
        result = run_layer6i_replay(
            trace_path,
            conventions=conventions,
            projection025_conductance_scale=float(scale),
            brian=brian,
        )
        spike_times = list(result.replay_spike_times_ms)
        scale_results.append(
            {
                "scale": float(scale),
                "finite": result.finite,
                "event_count": len(spike_times),
                "spike_times_ms": spike_times,
                "first_event_ms": spike_times[0] if spike_times else None,
                "soma_peak_mV": result.soma_peak_mV,
                "proximal_peak_mV": result.proximal_peak_mV,
                "exact_source_event_train": result.exact_spike_train,
                "maximum_state_error_from_source": max(
                    error for _, error in result.max_abs_error_by_variable
                ),
            }
        )

    scale_one = scale_results[0]
    gate = registration["scale_one_identity_gate"]
    scale_one_pass = (
        scale_one["scale"] == 1.0
        and scale_one["exact_source_event_train"]
        and scale_one["maximum_state_error_from_source"]
        <= float(gate["maximum_state_error"])
    )
    first_event_scale = next(
        (item["scale"] for item in scale_results if item["event_count"] > 0),
        None,
    )
    artifact = {
        "schema_version": 1,
        "id": registration["result_id"],
        "date": registration["date"],
        "status": (
            "isolated-first-event-bracket-located"
            if scale_one_pass and first_event_scale is not None
            else "isolated-first-event-bracket-open"
        ),
        "classification": registration["classification"],
        "registration": str(registration_path),
        "trace": registration["trace"],
        "runtime_fingerprint": conventions.fingerprint,
        "scale_one_identity_gate_pass": scale_one_pass,
        "all_trials_finite": all(item["finite"] for item in scale_results),
        "scale_results": scale_results,
        "first_event_scale": first_event_scale,
        "connected_network_changed": False,
        "parameter_selected": False,
        "original_smart_reproduced": False,
        "baseline_promoted": False,
        "interpretation_boundary": registration["boundary"],
    }
    print(yaml.safe_dump(artifact, sort_keys=False), end="")


if __name__ == "__main__":
    main()
