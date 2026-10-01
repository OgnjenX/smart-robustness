"""Check OS replay provenance and recompute spectra from retained spike times."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_l5_sst_like_stage4b as parent
import run_l5_sst_like_stage4b_os_bridge as bridge
import yaml
from verify_l5_sst_like_stage4b import _check_outcome

from smart_robustness.validation.active_apical_recording import file_sha256


def compare_outcomes(result: dict, expected: dict, protocol: dict) -> dict:
    """Derive replay agreement without trusting the execution runner's flag."""
    if result.get("continuation_authorized") is not False:
        raise ValueError("replay cannot authorize its own continuation")
    if result.get("frozen_baseline_modified") is not False:
        raise ValueError("replay does not preserve frozen baseline")
    if result.get("protocol") != protocol:
        raise ValueError("replay protocol changed")
    expected_summary = _check_outcome(expected, protocol)
    actual_summary = _check_outcome(result["outcome"], protocol)
    exact = expected == result["outcome"]
    status = "completed-os-bridge-replay" if exact else "engineering-stop-replay-disagreement"
    if result.get("exact_replay") is not exact or result.get("status") != status:
        raise ValueError("replay status contradicts independently compared outcomes")
    return {
        "exact_replay": exact,
        "old_summary": expected_summary,
        "new_summary": actual_summary,
    }


def verify_result(path: Path, seal_path: Path = bridge.SEAL) -> dict:
    bridge.verify_bridge_seal(seal_path)
    registration = yaml.safe_load(bridge.REGISTRATION.read_text())
    if path.resolve() != Path(registration["result"]).resolve():
        raise ValueError("replay result path differs from registration")
    for name in ("parent_seal", "checkpoint"):
        if file_sha256(Path(registration[name])) != registration[name + "_sha256"]:
            raise ValueError(f"replay input changed: {name}")
    parent.verify_seal(Path(registration["parent_seal"]))
    saved = yaml.safe_load(Path(registration["checkpoint"]).read_text())
    parent.validate_checkpoint_points(saved["points"])
    bridge.validate_registration(registration, registration["new_environment"])
    result = yaml.safe_load(path.read_text())
    expected_metadata = {
        "schema_version": 1,
        "registration_sha256": file_sha256(bridge.REGISTRATION),
        "execution_seal_sha256": file_sha256(seal_path),
        "original_checkpoint_sha256": registration["checkpoint_sha256"],
        "old_environment": registration["old_environment"],
        "new_environment": registration["new_environment"],
        "point_id": registration["point_id"],
    }
    for name, expected in expected_metadata.items():
        if result.get(name) != expected:
            raise ValueError(f"replay provenance mismatch: {name}")
    if {name: saved["identity"][name] for name in registration["old_environment"]} != (
        registration["old_environment"]
    ):
        raise ValueError("checkpoint environment mismatch")
    index = registration["point_index"]
    if saved["points"][index]["point_id"] != registration["point_id"]:
        raise ValueError("checkpoint replay point mismatch")
    expected = saved["points"][index]["outcomes"][registration["repetition"]]
    comparison = compare_outcomes(result, expected, saved["identity"]["protocol"])
    return {
        "raw_result_sha256": file_sha256(path),
        "original_checkpoint_sha256": registration["checkpoint_sha256"],
        "source_and_provenance_verified": True,
        **comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_result(args.result), indent=2))


if __name__ == "__main__":
    main()
