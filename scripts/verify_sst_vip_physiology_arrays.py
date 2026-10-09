"""Independent immutable-blob and parity reconstruction; never imports simulators."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import yaml
from verify_sst_vip_glif_parity import FIELDS, reconstruct, require

ROOT = Path("results/sst-vip-physiology-validation-1072")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read_array(directory, receipt):
    key = receipt["content_sha256"]
    require(re.fullmatch(r"[0-9a-f]{64}", key) is not None, "unsafe content key")
    path = directory / (key + ".npz")
    require(digest(path) == receipt["file_sha256"], "blob file digest differs")
    with np.load(path, allow_pickle=False) as archive:
        require(archive.files == ["values"], "blob field coverage differs")
        array = archive["values"]
    require(array.dtype.kind in "biufc", "non-numeric blob")
    meta = {"dtype": array.dtype.str, "shape": list(array.shape)}
    require(meta == receipt["metadata"], "blob dtype/shape differs")
    h = hashlib.sha256(json.dumps(meta, sort_keys=True).encode() + b"\0")
    data = memoryview(np.ascontiguousarray(array).reshape(-1).view(np.uint8))
    for start in range(0, len(data), 1048576):
        h.update(data[start : start + 1048576])
    require(h.hexdigest() == key, "blob content digest differs")
    require(array.nbytes == receipt["uncompressed_bytes"], "blob size differs")
    require(path.stat().st_size == receipt["stored_bytes"], "archive size differs")
    return array


def verify_case(directory, path, context):
    result = json.loads(path.read_text())
    require(result["context"] == context, "case execution context differs")
    recording = result["recording"]
    require(recording["identity"]["context"] == context, "recording context differs")
    require(
        digest(recording["identity"]["path"]) == recording["identity"]["source_sha256"],
        "source recording changed",
    )
    if recording["status"] != "complete":
        require(result["parity_passed"] is False, "failed recording promoted")
        return False
    stimulus = read_array(directory, recording["arrays"]["stimulus"])
    response = read_array(directory, recording["arrays"]["response"])
    require(
        stimulus.shape == response.shape
        and np.isfinite(stimulus).all()
        and np.isfinite(response).all(),
        "invalid recording arrays",
    )
    require(
        hashlib.sha256(stimulus.tobytes()).hexdigest() == recording["metadata"]["stimulus_sha256"],
        "stimulus identity differs",
    )
    arrays = {}
    failed = False
    for label in ("native0", "native1", "candidate0", "candidate1"):
        attempt = result["attempts"][label]
        saved = json.loads((directory / (path.stem + "-" + label + ".json")).read_text())
        require(attempt == saved, "attempt receipt differs from case")
        require(attempt["identity"]["context"] == context, "attempt context differs")
        if attempt["status"] == "simulation-exception":
            failed = True
            continue
        require(
            attempt["status"] == "complete" and set(attempt["arrays"]) == set(FIELDS),
            "attempt status or fields differ",
        )
        for field in FIELDS:
            arrays[label + "_" + field] = read_array(directory, attempt["arrays"][field])
    if failed:
        require(result["parity_passed"] is False, "exception promoted")
        return False
    comparison, repeats = reconstruct(arrays)
    require(comparison == result["comparison"], "raw-array parity gates differ")
    for kind, repeat in repeats.items():
        require(repeat == result[kind + "_exact_repeat"], "exact-repeat gate differs")
    bad = any(a["state"]["bad_reset_stop"] for a in result["attempts"].values())
    require(bad == result["bad_reset_stop"], "recorded reset flags differ")
    passed = comparison["passed"] and all(repeats.values()) and not bad
    require(passed == result["parity_passed"], "combined parity gate differs")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        action="store_true",
        help="verify completed cases only, never claim terminal completeness",
    )
    args = parser.parse_args()
    registration = Path(
        "docs/validation-results/post2008-sst-vip-physiology-protocol-registration-1068.yaml"
    )
    seal_path = Path("docs/validation-results/post2008-sst-vip-physiology-execution-seal-1073.yaml")
    seal = yaml.safe_load(seal_path.read_text())
    require(seal["registration_sha256"] == digest(registration), "registration changed")
    for path, expected in seal["implementation_sha256"].items():
        require(digest(path) == expected, "sealed implementation changed")
    context = {"registration_sha256": digest(registration), "seal_sha256": digest(seal_path)}
    paths = sorted(p for p in ROOT.glob("case-*.json") if re.fullmatch(r"case-\d{4}", p.stem))
    terminal = ROOT / "manifest.json"
    if not args.checkpoints:
        manifest = json.loads(terminal.read_text())
        require(
            manifest["context"] == context and len(paths) == len(manifest["records"]) == 509,
            "terminal case coverage incomplete",
        )
        for path, record in zip(paths, manifest["records"], strict=True):
            require(json.loads(path.read_text()) == record, "terminal case record differs")
    passed = sum(verify_case(ROOT, path, context) for path in paths)
    print(
        json.dumps(
            {
                "completed_cases_verified": len(paths),
                "parity_passed_cases": passed,
                "partial_checkpoint_mode": args.checkpoints,
                "physiological_scoring_independently_verified": False,
                "recorded_stop_flags_not_independently_reexecuted": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
