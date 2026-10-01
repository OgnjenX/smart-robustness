"""Complete sealed Stage 4B with explicit provenance across the OS patch update."""

from __future__ import annotations

import argparse
import fcntl
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path

import brian2 as brian
import run_l5_sst_like_stage4b as parent
import run_l5_sst_like_stage4b_os_bridge as bridge
import yaml
from verify_l5_sst_like_stage4b import _check_outcome
from verify_l5_sst_like_stage4b_os_bridge import verify_result as verify_bridge

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.validation.active_apical_recording import file_sha256

REGISTRATION = Path(
    "docs/validation-results/post2008-l5-sst-like-stage4b-continuation-registration-1012.yaml"
)
SEAL = Path("docs/validation-results/post2008-l5-sst-like-stage4b-continuation-seal-1013.yaml")


def verify_seal(path: Path) -> None:
    seal = yaml.safe_load(path.read_text())
    if seal.get("status") != "sealed-before-stage4b-continuation":
        raise ValueError("continuation is not sealed")
    required = {
        str(REGISTRATION),
        "scripts/run_l5_sst_like_stage4b_continuation.py",
        "scripts/verify_l5_sst_like_stage4b_continuation.py",
    }
    if not required.issubset(seal["files"]):
        raise ValueError("continuation seal lacks required files")
    for filename, digest in seal["files"].items():
        if file_sha256(Path(filename)) != digest:
            raise ValueError(f"continuation sealed file changed: {filename}")


def load_inputs():
    registration = yaml.safe_load(REGISTRATION.read_text())
    for name in ("bridge_assessment", "original_checkpoint", "bridge_result"):
        if file_sha256(Path(registration[name])) != registration[name + "_sha256"]:
            raise ValueError(f"continuation input changed: {name}")
    assessment = yaml.safe_load(Path(registration["bridge_assessment"]).read_text())
    if (
        assessment["status"] != "completed-exact-os-replay"
        or assessment["authorization"]["derived_continuation_registration_and_implementation"]
        is not True
    ):
        raise ValueError("bridge assessment does not authorize continuation design")
    if verify_bridge(Path(registration["bridge_result"]))["exact_replay"] is not True:
        raise ValueError("bridge did not replay exactly")
    source = yaml.safe_load(Path(registration["original_checkpoint"]).read_text())
    parent_seal = parent.verify_seal(Path(registration["parent_seal"]))
    baseline = load_frozen_classic_baseline(parent_seal["baseline_manifest"])
    learning = parent.stage4a_input(parent_seal)
    stage_registration = yaml.safe_load(Path(parent_seal["stage4_registration"]).read_text())
    spectral_registration = yaml.safe_load(
        Path(parent_seal["parent_figure14_registration"]).read_text()
    )
    protocol = parent.validate_protocol(stage_registration, spectral_registration)
    expected_identity = {
        "seal_sha256": file_sha256(Path(registration["parent_seal"])),
        **{
            key: parent_seal[key]
            for key in (
                "stage4_registration_sha256",
                "parent_figure14_registration_sha256",
                "stage4a_assessment_sha256",
                "stage4a_result_sha256",
            )
        },
        "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
        "runtime_fingerprint": baseline.runtime_fingerprint,
        "registered_points": parent.stage4a.registered_points(),
        "protocol": protocol,
        **registration["old_environment"],
    }
    if source["identity"] != expected_identity:
        raise ValueError("original scientific or environment identity changed")
    if len(source["points"]) != 5 or any(len(p["outcomes"]) != 2 for p in source["points"]):
        raise ValueError("original ten-repetition inventory changed")
    order = [
        {"point_index": index, "point_id": point["point_id"], "repetition": repetition}
        for index, point in enumerate(parent.stage4a.registered_points())
        if index >= 5
        for repetition in range(2)
    ]
    if registration["fixed_remaining_order"] != order:
        raise ValueError("remaining execution order changed")
    if registration["preserved_repetitions"] != 10 or registration["new_repetitions"] != 4:
        raise ValueError("registered repetition counts changed")
    parent.validate_checkpoint_points(source["points"])
    for point in source["points"]:
        if any(point.get(k) != v for k, v in parent.classify_point(point["outcomes"]).items()):
            raise ValueError("original classification changed")
        if point["classification"] == "engineering_stop":
            raise ValueError("original exact repeat failed")
        for outcome in point["outcomes"]:
            _check_outcome(outcome, protocol)
    manifest = yaml.safe_load(Path(parent_seal["baseline_manifest"]).read_text())
    profile = yaml.safe_load(
        (baseline.repository_root / manifest["implementation"]["profile"]["path"]).read_text()
    )
    return registration, source, baseline, learning, profile


def environments(points: list[dict], registration: dict) -> list[dict]:
    return [
        {
            "point_index": index,
            "point_id": point["point_id"],
            "repetition": outcome["repetition"],
            "environment": registration["old_environment" if index < 5 else "new_environment"],
        }
        for index, point in enumerate(points)
        for outcome in point["outcomes"]
    ]


def continuation_identity(registration: dict, seal_path: Path) -> dict:
    return {
        "registration_sha256": file_sha256(REGISTRATION),
        "execution_seal_sha256": file_sha256(seal_path),
        "original_checkpoint_sha256": registration["original_checkpoint_sha256"],
        "bridge_result_sha256": registration["bridge_result_sha256"],
        "bridge_assessment_sha256": registration["bridge_assessment_sha256"],
        "new_environment": registration["new_environment"],
    }


def validate_payload(payload: dict, source: dict, registration: dict, seal_path: Path) -> None:
    if payload.get("source_identity") != source["identity"]:
        raise ValueError("source identity changed")
    if payload.get("continuation_identity") != continuation_identity(registration, seal_path):
        raise ValueError("continuation identity changed")
    if payload.get("frozen_baseline_modified") is not False:
        raise ValueError("frozen baseline preservation missing")
    points = payload["points"]
    if points[:5] != source["points"] or len(points) < 5:
        raise ValueError("original outcomes changed")
    parent.validate_checkpoint_points(points)
    if payload.get("execution_environments") != environments(points, registration):
        raise ValueError("per-repetition environment provenance changed")
    for point in points:
        for outcome in point["outcomes"]:
            _check_outcome(outcome, source["identity"]["protocol"])
        if (
            len(point["outcomes"]) == 2
            and point.get("classification") is not None
            and any(point.get(k) != v for k, v in parent.classify_point(point["outcomes"]).items())
        ):
            raise ValueError("continuation classification changed")
        if point.get("classification") == "engineering_stop":
            raise ValueError("continuation exact-repeat failure")


def execute(output: Path, seal_path: Path) -> None:
    verify_seal(seal_path)
    registration, source, baseline, learning, profile = load_inputs()
    if bridge.environment() != registration["new_environment"]:
        raise ValueError("continuation environment changed")
    if output.resolve() != Path(registration["result"]).resolve():
        raise ValueError("continuation result path changed")
    payload = (
        yaml.safe_load(output.read_text())
        if output.exists()
        else {
            "schema_version": 1,
            "status": "running-stage4b-continuation",
            "source_identity": deepcopy(source["identity"]),
            "continuation_identity": continuation_identity(registration, seal_path),
            "points": deepcopy(source["points"]),
            "execution_environments": environments(source["points"], registration),
            "all_points_reported": False,
            "classification_counts": {},
            "frozen_baseline_modified": False,
        }
    )
    validate_payload(payload, source, registration, seal_path)
    if payload["status"] == "completed-stage4b-continuation":
        if len(payload["points"]) != 7 or any(len(p["outcomes"]) != 2 for p in payload["points"]):
            raise ValueError("incomplete result marked complete")
        return
    if payload["status"] != "running-stage4b-continuation":
        raise ValueError("continuation status does not permit resume")
    brian.prefs.codegen.target = "numpy"
    points = payload["points"]
    for index in range(5, 7):
        registered = parent.stage4a.registered_points()[index]
        if index == len(points):
            points.append(
                {
                    **registered,
                    "stage4a_classification": learning["points"][index]["classification"],
                    "outcomes": [],
                    "exact_repeat": None,
                    "both_figure14_gate_sets_pass": None,
                    "classification": None,
                }
            )
        point = points[index]
        if point["stage4a_classification"] != learning["points"][index]["classification"]:
            raise ValueError("learning classification changed")
        for repetition in range(len(point["outcomes"]), 2):
            point["outcomes"].append(
                parent.run_registered_repetition(
                    point=point,
                    repetition=repetition,
                    learned_weights=parent.learned_weights_for_repetition(
                        learning, index, repetition
                    ),
                    conventions=baseline.runtime_conventions(),
                    scales=dict(baseline.projection_weight_scales),
                    profile=profile,
                    protocol=source["identity"]["protocol"],
                    brian=brian,
                )
            )
            payload["execution_environments"] = environments(points, registration)
            parent.checkpoint(output, payload)
        point.update(parent.classify_point(point["outcomes"]))
        parent.checkpoint(output, payload)
        if point["classification"] == "engineering_stop":
            raise RuntimeError("continuation exact-repeat requirement failed")
    payload.update(
        status="completed-stage4b-continuation",
        all_points_reported=True,
        classification_counts=dict(sorted(Counter(p["classification"] for p in points).items())),
    )
    parent.checkpoint(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seal", type=Path, default=SEAL)
    args = parser.parse_args()
    registration = yaml.safe_load(REGISTRATION.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for path in (Path(registration["original_checkpoint"]), args.output):
            lock = stack.enter_context(path.with_suffix(path.suffix + ".lock").open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("original or continuation runner is already active") from error
        execute(args.output, args.seal)


if __name__ == "__main__":
    main()
