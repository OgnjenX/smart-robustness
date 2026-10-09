"""Explicitly sealed Stage-4D continuation preserving the failed source artifact."""

from __future__ import annotations

import argparse
import fcntl
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path

import run_l5_sst_like_stage4d as original
import yaml
from verify_l5_sst_like_stage4d_continuation import check_outcome

from smart_robustness.validation import l5_sst_like_stage4d as contract
from smart_robustness.validation.active_apical_recording import file_sha256

REGISTRATION = Path(
    "docs/validation-results/post2008-l5-sst-like-stage4d-continuation-registration-1022.yaml"
)
SEAL = Path("docs/validation-results/post2008-l5-sst-like-stage4d-continuation-seal-1023.yaml")
RESULT = Path("results/l5-sst-like-stage4d-continuation-1024.yaml")
SOURCE = original.RESULT


def verify_seal(path: Path) -> dict:
    seal = yaml.safe_load(path.read_text())
    required = {
        str(REGISTRATION),
        "scripts/run_l5_sst_like_stage4d_continuation.py",
        "scripts/verify_l5_sst_like_stage4d_continuation.py",
        "tests/test_l5_sst_like_stage4d_continuation.py",
        "src/smart_robustness/validation/stage4d_precision.py",
        str(original.SEAL),
        "docs/validation-results/post2008-l5-sst-like-stage4d-numerical-audit-assessment-1021.yaml",
    }
    if (
        seal.get("status") != "sealed-before-stage4d-continuation-replay-and-execution"
        or seal.get("execution_authorized") is not True
        or seal.get("checkpoint_replayed_before_seal") is not False
        or seal.get("new_network_outcomes_before_seal") is not False
        or seal.get("result") != str(RESULT)
        or not required.issubset(seal.get("files", {}))
    ):
        raise ValueError("Continuation seal incomplete or unauthorized")
    for filename, digest in seal["files"].items():
        if file_sha256(Path(filename)) != digest:
            raise ValueError(f"Continuation sealed file changed: {filename}")
    old_seal = original.verify_seal(original.SEAL)
    if seal["environment"] != old_seal["environment"]:
        raise ValueError("Continuation environment changed")
    return seal


def load_inputs(seal: dict) -> dict:
    reg = yaml.safe_load(REGISTRATION.read_text())
    for key in ("authorization",):
        if file_sha256(Path(reg[key])) != reg[key + "_sha256"]:
            raise ValueError("Continuation authorization changed")
    auth = yaml.safe_load(Path(reg["authorization"]).read_text())[
        "continuation_design_authorization"
    ]
    if auth.get("authorized") is not True:
        raise ValueError("Synthetic assessment did not authorize continuation design")
    lineage = reg["preserved_lineage"]
    for key in ("original_registration", "original_seal", "original_stop", "original_checkpoint"):
        if file_sha256(Path(lineage[key])) != lineage[key + "_sha256"]:
            raise ValueError(f"Continuation lineage changed: {key}")
    if Path(lineage["original_checkpoint"]).resolve() != SOURCE.resolve():
        raise ValueError("Continuation source path changed")
    if str(RESULT) != reg["result"] or str(SEAL) != reg["execution_seal"]:
        raise ValueError("Continuation registered output or seal changed")
    amendment = reg["numerical_amendment"]
    for key in ("synthetic_registration", "checker"):
        if file_sha256(Path(amendment[key])) != amendment[key + "_sha256"]:
            raise ValueError("Continuation numerical amendment changed")
    old_seal = original.verify_seal(original.SEAL)
    learning, spectra, synchrony, baseline, _profile, protocol = original.load_inputs(old_seal)
    if reg["execution"]["environment"] != seal["environment"]:
        raise ValueError("Registered continuation environment changed")
    for key, actual in (
        ("baseline_manifest_fingerprint", baseline.manifest_fingerprint),
        ("runtime_fingerprint", baseline.runtime_fingerprint),
    ):
        if reg["execution"][key] != actual:
            raise ValueError("Continuation frozen baseline changed")
    source = yaml.safe_load(SOURCE.read_text())
    if source["identity"] != original.identity_for(original.SEAL, old_seal, protocol):
        raise ValueError("Original checkpoint scientific identity changed")
    if (
        source.get("status") != "running-stage4d"
        or len(source["points"]) != 1
        or len(source["points"][0]["outcomes"]) != 1
        or source["points"][0].get("classification") is not None
        or source.get("frozen_baseline_modified") is not False
        or source.get("network_execution") is not True
    ):
        raise ValueError("Original stopped checkpoint inventory changed")
    original.parent.parent.validate_checkpoint_points(source["points"])
    if source["points"][0]["earlier_classifications"] != original.earlier_for(
        0, learning, spectra, synchrony
    ):
        raise ValueError("Original earlier classifications changed")
    if SOURCE.stat().st_size != lineage["original_checkpoint_bytes"]:
        raise ValueError("Original checkpoint size changed")
    order = [
        {"stage4a_index": i, "point_id": p["point_id"], "repetition": r}
        for i, p in enumerate(original.parent.parent.stage4a.registered_points())
        for r in range(2)
        if not (i == 0 and r == 0)
    ]
    if reg["execution"]["fixed_remaining_order"] != order:
        raise ValueError("Continuation remaining order changed")
    check_outcome(source["points"][0]["outcomes"][0])
    return {
        "registration": reg,
        "source": source,
        "learning": learning,
        "spectra": spectra,
        "synchrony": synchrony,
        "baseline": baseline,
        "protocol": protocol,
    }


def identity_for(seal_path: Path, seal: dict, inputs: dict) -> dict:
    return {
        "registration_sha256": file_sha256(REGISTRATION),
        "seal_sha256": file_sha256(seal_path),
        "original_checkpoint_sha256": inputs["registration"]["preserved_lineage"][
            "original_checkpoint_sha256"
        ],
        "source_identity": inputs["source"]["identity"],
        "files": seal["files"],
        "environment": seal["environment"],
    }


def diagnostics_for(points):
    return [
        {"stage4a_index": i, "repetition": r, **check_outcome(o)}
        for i, p in enumerate(points)
        for r, o in enumerate(p["outcomes"])
    ]


def validate_payload(payload, identity, inputs):
    if (
        payload.get("schema_version") != 1
        or payload.get("identity") != identity
        or payload.get("status")
        not in {"running-stage4d-continuation", "completed-stage4d-continuation"}
        or payload.get("original_failure_retained") is not True
        or payload.get("frozen_baseline_modified") is not False
        or payload.get("network_execution") is not True
    ):
        raise ValueError("Continuation checkpoint identity or status changed")
    points = payload["points"]
    if not points or points[0]["outcomes"][:1] != inputs["source"]["points"][0]["outcomes"]:
        raise ValueError("Preserved original outcome changed")
    original.parent.parent.validate_checkpoint_points(points)
    for i, p in enumerate(points):
        expected_earlier = original.earlier_for(
            i, inputs["learning"], inputs["spectra"], inputs["synchrony"]
        )
        if p["earlier_classifications"] != expected_earlier:
            raise ValueError("Continuation earlier classifications changed")
        for outcome in p["outcomes"]:
            check_outcome(outcome)
        if p.get("classification") is not None:
            classified = contract.classify_point(p["outcomes"], earlier=expected_earlier)
            if any(p.get(k) != v for k, v in classified.items()):
                raise ValueError("Continuation joint classification changed")
            if classified["classification"] == "engineering_stop":
                raise RuntimeError("Continuation engineering stop")
    if payload["status"] != "completed-stage4d-continuation":
        return
    if (
        len(points) != 7
        or any(p.get("classification") is None for p in points)
        or payload.get("all_points_reported") is not True
        or payload.get("joint_classification_counts")
        != dict(sorted(Counter(p["joint_classification"] for p in points).items()))
        or payload.get("numerical_diagnostics") != diagnostics_for(points)
    ):
        raise ValueError("Continuation terminal inventory or diagnostics changed")


def execute(output, seal_path):
    seal = verify_seal(seal_path)
    if output.resolve() != RESULT.resolve():
        raise ValueError("Continuation output differs from registration")
    inputs = load_inputs(seal)
    identity = identity_for(seal_path, seal, inputs)
    payload = (
        yaml.safe_load(output.read_text())
        if output.exists()
        else {
            "schema_version": 1,
            "status": "running-stage4d-continuation",
            "identity": identity,
            "points": deepcopy(inputs["source"]["points"]),
            "original_failure_retained": True,
            "frozen_baseline_modified": False,
            "network_execution": True,
            "all_points_reported": False,
            "joint_classification_counts": {},
        }
    )
    validate_payload(payload, identity, inputs)
    if payload["status"] == "completed-stage4d-continuation":
        return
    original.parent.parent.checkpoint(output, payload)
    for i, registered in enumerate(original.parent.parent.stage4a.registered_points()):
        if i == len(payload["points"]):
            payload["points"].append(
                {
                    **registered,
                    "outcomes": [],
                    "classification": None,
                    "earlier_classifications": original.earlier_for(
                        i, inputs["learning"], inputs["spectra"], inputs["synchrony"]
                    ),
                }
            )
        point = payload["points"][i]
        for repetition in range(len(point["outcomes"]), 2):
            outcome = original.run_repetition(
                point=point,
                repetition=repetition,
                learned_weights=original.parent.parent.learned_weights_for_repetition(
                    inputs["learning"], i, repetition
                ),
                conventions=inputs["baseline"].runtime_conventions(),
                scales=dict(inputs["baseline"].projection_weight_scales),
                protocol=inputs["protocol"],
            )
            point["outcomes"].append(outcome)
            original.parent.parent.checkpoint(output, payload)
            check_outcome(outcome)
        point.update(
            contract.classify_point(point["outcomes"], earlier=point["earlier_classifications"])
        )
        original.parent.parent.checkpoint(output, payload)
        if point["classification"] == "engineering_stop":
            raise RuntimeError("Continuation engineering stop")
    payload.update(
        status="completed-stage4d-continuation",
        all_points_reported=True,
        joint_classification_counts=dict(
            sorted(Counter(p["joint_classification"] for p in payload["points"]).items())
        ),
        numerical_diagnostics=diagnostics_for(payload["points"]),
    )
    validate_payload(payload, identity, inputs)
    original.parent.parent.checkpoint(output, payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seal", type=Path, default=SEAL)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for path in (SOURCE, output):
            lock = stack.enter_context(path.with_suffix(path.suffix + ".lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(output, args.seal.resolve())


if __name__ == "__main__":
    main()
