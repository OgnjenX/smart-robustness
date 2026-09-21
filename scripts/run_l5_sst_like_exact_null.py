"""Run the sealed exact-null L5 SST-like functional-route prerequisite."""

from __future__ import annotations

import argparse
import fcntl
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import run_inhibitory_routing_null as legacy_runner
import yaml
from run_layer5_distal_nak_stage3 import _without_repetition

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.models.sst_like_feedback import make_l5_sst_like_sector_builder
from smart_robustness.validation.active_apical_recording import file_sha256

REPETITIONS = 2
ZERO_RESOURCE_NS = 0.0
CANONICAL_DELAY_MS = 3.0
REGISTERED_DELAYS_MS = (1.0, 3.0, 7.0)
DEFAULT_SEAL = Path(
    "docs/validation-results/post2008-l5-sst-like-exact-null-seal-986.yaml"
)


def checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".l5-sst-null-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            yaml.safe_dump(payload, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def verify_seal(path: Path) -> dict[str, Any]:
    seal = yaml.safe_load(path.read_text())
    if seal["status"] != "sealed-before-null-network-outcomes":
        raise ValueError("an exact-null SST-like execution seal is required")
    for filename, digest in seal["files"].items():
        if file_sha256(Path(filename)) != digest:
            raise ValueError(f"sealed file changed: {filename}")
    if "scripts/run_l5_sst_like_exact_null.py" not in seal["files"]:
        raise ValueError("runner is missing from execution seal")
    return seal


def structural_noop_proof() -> dict[str, Any]:
    sentinel = object()
    markers = [object() for _ in REGISTERED_DELAYS_MS]
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def base_builder(*args: Any, **kwargs: Any):
        calls.append((args, kwargs))
        return sentinel

    returned = []
    for marker, delay_ms in zip(markers, REGISTERED_DELAYS_MS, strict=True):
        builder = make_l5_sst_like_sector_builder(
            base_builder=base_builder,
            total_conductance_nS=ZERO_RESOURCE_NS,
            delay_ms=delay_ms,
        )
        returned.append(builder(marker, registered=True) is sentinel)
    args_preserved = all(
        args == (marker,)
        for (args, _), marker in zip(calls, markers, strict=True)
    )
    kwargs_preserved = all(
        kwargs == {"registered": True} for _, kwargs in calls
    )
    no_factory_injected = all("population_factory" not in kwargs for _, kwargs in calls)
    return {
        "resource_nS": ZERO_RESOURCE_NS,
        "registered_delay_labels_ms": list(REGISTERED_DELAYS_MS),
        "base_builder_calls": len(calls),
        "same_return_identity": all(returned),
        "args_preserved": args_preserved,
        "kwargs_preserved": kwargs_preserved,
        "population_factory_not_injected": no_factory_injected,
        "port_created": False,
        "synapse_created": False,
        "all_gates_pass": bool(
            len(calls) == len(REGISTERED_DELAYS_MS)
            and all(returned)
            and args_preserved
            and kwargs_preserved
            and no_factory_injected
        ),
    }


def load_legacy_reference(path: Path, expected_sha256: str) -> dict[str, Any]:
    if file_sha256(path) != expected_sha256:
        raise ValueError("archived legacy result hash differs")
    payload = yaml.safe_load(path.read_text())
    if payload["status"] != "completed-inhibitory-routing-null":
        raise ValueError("legacy reference is not complete")
    if payload["cross_arm_exact"] is not True:
        raise ValueError("legacy wrapped and unwrapped controls are not exact")
    outcomes = [
        outcome
        for arm in ("unwrapped_control", "wrapped_legacy_aggregate")
        for outcome in payload["arms"][arm]["outcomes"]
    ]
    normalized = [_without_repetition(outcome) for outcome in outcomes]
    if len(normalized) != 4 or any(item != normalized[0] for item in normalized[1:]):
        raise ValueError("legacy reference does not contain four exact outcomes")
    if not all(outcome["behavioral_pass"] for outcome in outcomes):
        raise ValueError("legacy reference contains a behavioral failure")
    return normalized[0]


def run_zero_repetition(**kwargs: Any) -> dict[str, object]:
    original_builder_selector = legacy_runner._builder

    def zero_builder_selector(*, wrapped: bool, base_builder: Any):
        if not wrapped:
            raise ValueError("the SST-like null runner requires its wrapper path")
        return make_l5_sst_like_sector_builder(
            base_builder=base_builder,
            total_conductance_nS=ZERO_RESOURCE_NS,
            delay_ms=CANONICAL_DELAY_MS,
        )

    try:
        legacy_runner._builder = zero_builder_selector
        return legacy_runner._run_repetition(wrapped=True, **kwargs)
    finally:
        legacy_runner._builder = original_builder_selector


def execute(output: Path, seal_path: Path) -> None:
    import brian2 as brian

    seal = verify_seal(seal_path)
    structural = structural_noop_proof()
    if not structural["all_gates_pass"]:
        raise RuntimeError("zero-resource wrapper failed its structural no-op proof")
    baseline = load_frozen_classic_baseline(seal["baseline_manifest"])
    legacy_reference = load_legacy_reference(
        Path(seal["legacy_result"]), seal["legacy_result_sha256"]
    )
    figure6_path = Path(seal["figure6_result"])
    figure6 = yaml.safe_load(figure6_path.read_text())
    source_trials = figure6["arms"]["ratio_0"]["trials"]
    keys = ("bottom_up_weights", "top_down_wide_weights", "top_down_narrow_weights")
    if len(source_trials) != 2 or any(
        source_trials[0][key] != source_trials[1][key] for key in keys
    ):
        raise ValueError("sealed ratio-zero Figure-6 weights are not exact repeats")
    learned_weights = {
        "modeldb112923.projection.035": source_trials[0]["bottom_up_weights"],
        "modeldb112923.projection.005": source_trials[0]["top_down_wide_weights"],
        "modeldb112923.projection.007": source_trials[0]["top_down_narrow_weights"],
    }
    manifest = yaml.safe_load(Path(seal["baseline_manifest"]).read_text())
    profile = yaml.safe_load(
        (baseline.repository_root / manifest["implementation"]["profile"]["path"]).read_text()
    )
    identity = {
        "seal_sha256": file_sha256(seal_path),
        "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
        "runtime_fingerprint": baseline.runtime_fingerprint,
        "legacy_result_sha256": file_sha256(Path(seal["legacy_result"])),
        "figure6_result_sha256": file_sha256(figure6_path),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "brian2": brian.__version__,
    }
    payload = {
        "schema_version": 1,
        "status": "running-l5-sst-like-exact-null",
        "identity": identity,
        "structural_noop_proof": structural,
        "outcomes": [],
        "exact_repeat": None,
        "legacy_exact": None,
        "all_behavioral_gates_pass": None,
        "classification": None,
        "nonzero_execution_authorized": False,
        "frozen_baseline_modified": False,
    }
    if output.exists():
        payload = yaml.safe_load(output.read_text())
        if payload["identity"] != identity or payload["structural_noop_proof"] != structural:
            raise ValueError("checkpoint identity or structural proof mismatch")
    outcomes = list(payload["outcomes"])
    if len(outcomes) > REPETITIONS:
        raise ValueError("checkpoint has too many repetitions")
    if payload["status"] == "completed-l5-sst-like-exact-null":
        if len(outcomes) != REPETITIONS or payload["classification"] is None:
            raise ValueError("incomplete checkpoint marked complete")
        return

    brian.prefs.codegen.target = "numpy"
    for repetition in range(len(outcomes), REPETITIONS):
        outcomes.append(
            run_zero_repetition(
                learned_weights=learned_weights,
                conventions=baseline.runtime_conventions(),
                scales=dict(baseline.projection_weight_scales),
                profile=profile,
                repetition=repetition,
                brian=brian,
            )
        )
        payload["outcomes"] = outcomes
        checkpoint(output, payload)
    normalized = [_without_repetition(outcome) for outcome in outcomes]
    payload["exact_repeat"] = normalized[0] == normalized[1]
    payload["legacy_exact"] = all(outcome == legacy_reference for outcome in normalized)
    payload["all_behavioral_gates_pass"] = all(
        outcome["behavioral_pass"] for outcome in outcomes
    )
    passed = bool(
        payload["exact_repeat"]
        and payload["legacy_exact"]
        and payload["all_behavioral_gates_pass"]
    )
    payload["classification"] = "exact_null_pass" if passed else "failure"
    payload["status"] = "completed-l5-sst-like-exact-null"
    checkpoint(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seal", type=Path, default=DEFAULT_SEAL)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(output.suffix + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("an identical L5 SST-like null runner is active") from error
        execute(output, args.seal.resolve())


if __name__ == "__main__":
    main()
