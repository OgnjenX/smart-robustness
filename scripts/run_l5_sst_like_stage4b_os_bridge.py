"""Replay a fixed Stage-4B repetition after a preregistered OS patch change."""

from __future__ import annotations

import argparse
import fcntl
import platform
from contextlib import ExitStack
from pathlib import Path

import brian2 as brian
import run_l5_sst_like_stage4b as parent
import yaml
from verify_l5_sst_like_stage4b import _check_outcome

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.validation.active_apical_recording import file_sha256

REGISTRATION = Path(
    "docs/validation-results/post2008-l5-sst-like-stage4b-os-bridge-registration-1009.yaml"
)
SEAL = Path("docs/validation-results/post2008-l5-sst-like-stage4b-os-bridge-seal-1010.yaml")


def environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "brian2": brian.__version__,
        "platform": platform.platform(),
    }


def validate_registration(registration: dict, current: dict) -> None:
    if current != registration["new_environment"]:
        raise ValueError("new environment differs from registration")
    if registration["point_index"] != 0 or registration["repetition"] != 0:
        raise ValueError("replay selection changed")
    if registration["point_id"] != parent.stage4a.registered_points()[0]["point_id"]:
        raise ValueError("replay point changed")
    if registration["conditions"] != ["match", "mismatch"]:
        raise ValueError("replay conditions changed")


def verify_bridge_seal(seal_path: Path) -> None:
    seal = yaml.safe_load(seal_path.read_text())
    if seal.get("status") != "sealed-before-os-bridge-replay":
        raise ValueError("bridge execution is not sealed")
    required = {str(REGISTRATION), "scripts/run_l5_sst_like_stage4b_os_bridge.py"}
    if not required.issubset(seal["files"]):
        raise ValueError("bridge seal lacks required files")
    for filename, digest in seal["files"].items():
        if file_sha256(Path(filename)) != digest:
            raise ValueError(f"bridge sealed file changed: {filename}")


def exact_replay(expected: dict, actual: dict, protocol: dict) -> bool:
    _check_outcome(expected, protocol)
    _check_outcome(actual, protocol)
    return expected == actual


def execute(output: Path, seal_path: Path) -> None:
    verify_bridge_seal(seal_path)
    registration = yaml.safe_load(REGISTRATION.read_text())
    validate_registration(registration, environment())
    if output.resolve() != Path(registration["result"]).resolve():
        raise ValueError("bridge output path changed")
    if output.exists():
        raise ValueError("bridge result already exists; assess it before any new execution")
    for name in ("parent_seal", "checkpoint"):
        if file_sha256(Path(registration[name])) != registration[name + "_sha256"]:
            raise ValueError(f"bridge input changed: {name}")
    parent_seal = parent.verify_seal(Path(registration["parent_seal"]))
    saved = yaml.safe_load(Path(registration["checkpoint"]).read_text())
    parent.validate_checkpoint_points(saved["points"])
    identity = saved["identity"]
    if {name: identity[name] for name in environment()} != registration["old_environment"]:
        raise ValueError("old checkpoint environment differs from registration")
    baseline = load_frozen_classic_baseline(parent_seal["baseline_manifest"])
    stage4a_result = parent.stage4a_input(parent_seal)
    stage_registration = yaml.safe_load(Path(parent_seal["stage4_registration"]).read_text())
    parent_registration = yaml.safe_load(
        Path(parent_seal["parent_figure14_registration"]).read_text()
    )
    protocol = parent.validate_protocol(stage_registration, parent_registration)
    expected_identity = {
        "seal_sha256": registration["parent_seal_sha256"],
        **{
            name: parent_seal[name]
            for name in (
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
    if identity != expected_identity:
        raise ValueError("old checkpoint identity changed")
    manifest = yaml.safe_load(Path(parent_seal["baseline_manifest"]).read_text())
    profile = yaml.safe_load(
        (baseline.repository_root / manifest["implementation"]["profile"]["path"]).read_text()
    )
    point = parent.stage4a.registered_points()[0]
    expected = saved["points"][0]["outcomes"][0]
    _check_outcome(expected, protocol)
    brian.prefs.codegen.target = "numpy"
    actual = parent.run_registered_repetition(
        point=point,
        repetition=0,
        learned_weights=parent.learned_weights_for_repetition(stage4a_result, 0, 0),
        conventions=baseline.runtime_conventions(),
        scales=dict(baseline.projection_weight_scales),
        profile=profile,
        protocol=protocol,
        brian=brian,
    )
    passed = exact_replay(expected, actual, protocol)
    if file_sha256(Path(registration["checkpoint"])) != registration["checkpoint_sha256"]:
        raise ValueError("original checkpoint changed during replay")
    parent.checkpoint(
        output,
        {
            "schema_version": 1,
            "status": "completed-os-bridge-replay"
            if passed
            else "engineering-stop-replay-disagreement",
            "registration_sha256": file_sha256(REGISTRATION),
            "execution_seal_sha256": file_sha256(seal_path),
            "original_checkpoint_sha256": registration["checkpoint_sha256"],
            "old_environment": registration["old_environment"],
            "new_environment": environment(),
            "point_id": point["point_id"],
            "protocol": protocol,
            "outcome": actual,
            "exact_replay": passed,
            "continuation_authorized": False,
            "frozen_baseline_modified": False,
        },
    )
    if not passed:
        raise RuntimeError("OS bridge exact replay failed; assess preserved result")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seal", type=Path, default=SEAL)
    args = parser.parse_args()
    registration = yaml.safe_load(REGISTRATION.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for path in (Path(registration["checkpoint"]), args.output):
            lock = stack.enter_context(path.with_suffix(path.suffix + ".lock").open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("original or bridge runner is already active") from error
        execute(args.output, args.seal)


if __name__ == "__main__":
    main()
