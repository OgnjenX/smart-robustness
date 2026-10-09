"""Execute the separately sealed, checkpoint-resumable SST inter-area holdout."""

from __future__ import annotations

import argparse
import fcntl
import platform
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import brian2 as brian
import numpy as np
import run_l5_sst_like_stage4c as parent
import yaml
from verify_l5_sst_like_stage4c import verify_result as verify_stage4c

from smart_robustness.validation import l5_sst_like_stage4d as contract
from smart_robustness.validation.active_apical_recording import file_sha256
from smart_robustness.validation.higher_order import (
    assess_figure16_candidate,
    run_figure16_candidate,
)

SEAL = Path("docs/validation-results/post2008-l5-sst-like-stage4d-seal-1017.yaml")
ASSESSMENT = Path("docs/validation-results/post2008-l5-sst-like-stage4c-assessment-1016.yaml")
REGISTRATION = parent.REGISTRATION
PARENT_PROTOCOL = Path("docs/validation-results/calibrated-figure16-holdout-registration-813.yaml")
RESULT = Path(contract.RESULT)


def verify_seal(path: Path) -> dict:
    seal = yaml.safe_load(path.read_text())
    required = {
        "scripts/run_l5_sst_like_stage4d.py",
        "scripts/verify_l5_sst_like_stage4d.py",
        "tests/test_l5_sst_like_stage4d.py",
        "tests/test_l5_sst_like_stage4d_runner.py",
        "src/smart_robustness/validation/l5_sst_like_stage4d.py",
        "src/smart_robustness/validation/higher_order.py",
        "src/smart_robustness/analysis/cross_correlation.py",
        "src/smart_robustness/analysis/lfp.py",
        "src/smart_robustness/models/sst_like_feedback.py",
        "src/smart_robustness/classic_sector.py",
        str(REGISTRATION),
        str(PARENT_PROTOCOL),
        str(ASSESSMENT),
        str(parent.SEAL),
    }
    if (
        seal.get("status") != "sealed-before-stage4d-network-outcomes"
        or seal.get("execution_authorized") is not True
        or seal.get("network_outcome_observed_before_seal") is not False
        or seal.get("result") != str(RESULT)
        or not required.issubset(seal.get("files", {}))
    ):
        raise ValueError("Stage-4D seal incomplete or unauthorized")
    for filename, digest in seal["files"].items():
        if file_sha256(Path(filename)) != digest:
            raise ValueError(f"Stage-4D sealed file changed: {filename}")
    if seal["environment"] != {
        "python": platform.python_version(),
        "brian2": brian.__version__,
        "platform": platform.platform(),
    }:
        raise ValueError("Stage-4D execution environment changed")
    return seal


def load_inputs(seal: dict):
    assessment = yaml.safe_load(ASSESSMENT.read_text())
    auth = assessment["stage4d_authorization"]
    if (
        auth.get("authorized") is not True
        or auth.get("all_seven_points_required") is not True
        or auth.get("scope") != "implement-and-seal-stage4d-figure16-only"
        or auth.get("geometry_seeds") != [16, 17]
        or auth.get("repetitions_per_point") != 2
    ):
        raise ValueError("Stage-4C assessment does not authorize this implementation")
    raw = Path(assessment["raw_result"])
    if file_sha256(raw) != assessment["raw_result_sha256"]:
        raise ValueError("Stage-4C raw result changed")
    verify_stage4c(raw)
    synchrony = yaml.safe_load(raw.read_text())
    learning, spectra, baseline, profile, _ = parent.load_inputs(parent.verify_seal(parent.SEAL))
    if (
        baseline.manifest_fingerprint != seal["baseline_manifest_fingerprint"]
        or baseline.runtime_fingerprint != seal["runtime_fingerprint"]
    ):
        raise ValueError("Stage-4D frozen baseline changed")
    reg = yaml.safe_load(REGISTRATION.read_text())
    stage = reg["stage4d_figure16"]
    if str(PARENT_PROTOCOL) != stage["parent_protocol"] or (
        file_sha256(PARENT_PROTOCOL) != stage["parent_protocol_sha256"]
    ):
        raise ValueError("Stage-4D parent protocol changed")
    protocol = contract.validate_protocol(reg, yaml.safe_load(PARENT_PROTOCOL.read_text()))
    return learning, spectra, synchrony, baseline, profile, protocol


def field_payload(field) -> dict:
    names = (
        "potential_uV",
        "current_source_density_uV_per_um",
        "inferior_300um_tip_depth_um",
        "inferior_300um_potential_uV",
        "superior_300um_tip_depth_um",
        "superior_300um_potential_uV",
    )
    return {
        "seed": field.seed,
        "fingerprint": field.fingerprint,
        **{name: np.asarray(getattr(field, name), dtype=float).tolist() for name in names},
    }


def run_repetition(*, point, repetition, learned_weights, conventions, scales, protocol):
    # Unique directories avoid reusing partial standalone builds after a crash.
    directory = tempfile.mkdtemp(prefix="smart-sst-stage4d-", dir="/private/tmp")
    candidate = run_figure16_candidate(
        learned_weights=learned_weights,
        persistent_projection_weight_scales=scales,
        projection036_variance_topology=True,
        protocol=protocol,
        geometry_seed=16,
        conventions=conventions,
        cpp_standalone_directory=directory,
        network_builder=contract.full_network_builder(point),
        brian=brian,
    )
    fields = {"v1": field_payload(candidate.v1_field), "v2": field_payload(candidate.v2_field)}
    finite = all(
        np.all(np.isfinite(np.asarray(value)))
        for field in fields.values()
        for key, value in field.items()
        if key not in {"seed", "fingerprint"}
    )
    if not finite or len(candidate.sample_times_ms) != 1000:
        # Do not feed invalid fields to the correlation estimator or call them
        # a scientific failure. The checkpoint executor retains this evidence.
        correlations = []
        spectral_pass = False
    else:
        assessment = assess_figure16_candidate(candidate)
        correlations = [
            {
                "band_hz": list(c.band_hz),
                "lag_ms": c.lag_ms.tolist(),
                "raw": c.raw.tolist(),
                "normalized": c.normalized.tolist(),
                "peak_absolute_normalized": c.peak_absolute_normalized,
            }
            for c in assessment.cross_correlations
        ]
        spectral_pass = assessment.frequency_assessment.lower_frequency_stronger_than_gamma
    gates = dict(
        zip(
            contract.GATES,
            [len(candidate.sample_times_ms) == 1000, bool(finite), bool(spectral_pass)],
            strict=True,
        )
    )
    return {
        "repetition": repetition,
        "learned_state_provenance": candidate.learned_state_provenance,
        "sample_times_ms": list(candidate.sample_times_ms),
        "fields": fields,
        "cross_correlations": correlations,
        "gates": gates,
    }


def earlier_for(index: int, learning: dict, spectra: dict, synchrony: dict) -> dict:
    return {
        "stage4a": learning["points"][index]["classification"],
        "stage4b": spectra["points"][index]["classification"],
        "stage4c": synchrony["points"][index]["classification"],
    }


def identity_for(seal_path: Path, seal: dict, protocol) -> dict:
    return {
        "seal_sha256": file_sha256(seal_path),
        "files": seal["files"],
        "protocol": {**asdict(protocol), "frequency_bands_hz": contract.BANDS},
        "geometry_seed_v1": 16,
        "geometry_seed_v2": 17,
        "environment": seal["environment"],
        "registered_points": parent.parent.stage4a.registered_points(),
        "baseline_manifest_fingerprint": seal["baseline_manifest_fingerprint"],
        "runtime_fingerprint": seal["runtime_fingerprint"],
    }


def validate_payload(payload, identity, learning, spectra, synchrony):
    if (
        payload.get("schema_version") != 1
        or payload.get("identity") != identity
        or payload.get("status") not in {"running-stage4d", "completed-stage4d"}
        or payload.get("frozen_baseline_modified") is not False
        or payload.get("network_execution") is not True
    ):
        raise ValueError("Stage-4D checkpoint identity or status changed")
    parent.parent.validate_checkpoint_points(payload["points"])
    from verify_l5_sst_like_stage4d import check_outcome

    for index, point in enumerate(payload["points"]):
        earlier = earlier_for(index, learning, spectra, synchrony)
        if point.get("earlier_classifications") != earlier:
            raise ValueError("Stage-4D earlier classifications changed")
        for outcome in point["outcomes"]:
            check_outcome(outcome)
        if point.get("classification") is not None:
            expected = contract.classify_point(point["outcomes"], earlier=earlier)
            if any(point.get(k) != v for k, v in expected.items()):
                raise ValueError("Stage-4D joint classification changed")
            if point["classification"] == "engineering_stop":
                raise RuntimeError("Stage-4D engineering stop; do not resume scientific execution")
    complete = len(payload["points"]) == 7 and all(
        p.get("classification") for p in payload["points"]
    )
    if payload["status"] == "completed-stage4d" and (
        not complete
        or payload.get("all_points_reported") is not True
        or payload.get("joint_classification_counts")
        != dict(sorted(Counter(p["joint_classification"] for p in payload["points"]).items()))
    ):
        raise ValueError("Stage-4D terminal inventory changed")


def execute(output: Path, seal_path: Path):
    seal = verify_seal(seal_path)
    if output.resolve() != RESULT.resolve():
        raise ValueError("Stage-4D output differs from registration")
    learning, spectra, synchrony, baseline, _profile, protocol = load_inputs(seal)
    identity = identity_for(seal_path, seal, protocol)
    payload = (
        yaml.safe_load(output.read_text())
        if output.exists()
        else {
            "schema_version": 1,
            "status": "running-stage4d",
            "identity": identity,
            "points": [],
            "all_points_reported": False,
            "joint_classification_counts": {},
            "network_execution": True,
            "frozen_baseline_modified": False,
        }
    )
    validate_payload(payload, identity, learning, spectra, synchrony)
    if payload["status"] == "completed-stage4d":
        return
    from verify_l5_sst_like_stage4d import check_outcome

    for index, registered in enumerate(parent.parent.stage4a.registered_points()):
        if index == len(payload["points"]):
            payload["points"].append(
                {
                    **registered,
                    "outcomes": [],
                    "classification": None,
                    "earlier_classifications": earlier_for(index, learning, spectra, synchrony),
                }
            )
        point = payload["points"][index]
        for repetition in range(len(point["outcomes"]), 2):
            outcome = run_repetition(
                point=point,
                repetition=repetition,
                learned_weights=parent.parent.learned_weights_for_repetition(
                    learning, index, repetition
                ),
                conventions=baseline.runtime_conventions(),
                scales=dict(baseline.projection_weight_scales),
                protocol=protocol,
            )
            # Preserve a numerical-stop outcome before the verifier rejects it.
            point["outcomes"].append(outcome)
            parent.parent.checkpoint(output, payload)
            check_outcome(outcome)
        point.update(
            contract.classify_point(point["outcomes"], earlier=point["earlier_classifications"])
        )
        parent.parent.checkpoint(output, payload)
        if point["classification"] == "engineering_stop":
            raise RuntimeError("Stage-4D engineering stop")
    payload.update(
        status="completed-stage4d",
        all_points_reported=True,
        joint_classification_counts=dict(
            sorted(Counter(p["joint_classification"] for p in payload["points"]).items())
        ),
    )
    validate_payload(payload, identity, learning, spectra, synchrony)
    parent.parent.checkpoint(output, payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seal", type=Path, default=SEAL)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(output.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(output, args.seal.resolve())


if __name__ == "__main__":
    main()
