"""Immutable per-attempt execution receipts; no native imports or trace access."""

from __future__ import annotations

import copy
import json
import re

from smart_robustness.validation.native_glif_parity import FIELDS
from smart_robustness.validation.physiology_storage import array_identity, require_space


def stored_attempt(store, name, parameters, stimulus, function, context):
    if re.fullmatch(r"[a-z0-9-]+", name) is None:
        raise ValueError("unsafe attempt name")
    _, _, stimulus_hash = array_identity(stimulus)
    identity = {
        "parameters": parameters,
        "stimulus_content_sha256": stimulus_hash,
        "context": context,
    }
    # Reject non-serializable/nonfinite context before starting a potentially expensive run.
    json.dumps(identity, allow_nan=False)
    path = store.directory / (name + ".json")
    if path.exists():
        result = json.loads(path.read_text())
        if result["identity"] != identity:
            raise ValueError("attempt identity or execution context changed")
        if result["status"] == "complete":
            if set(result["arrays"]) != set(FIELDS):
                raise ValueError("checkpoint field coverage changed")
            for receipt in result["arrays"].values():
                store.get(receipt)
        elif result["status"] != "simulation-exception":
            raise ValueError("unknown attempt checkpoint status")
        return result
    started = store.directory / (name + "-started.json")
    if started.exists():
        raise ValueError("interrupted attempt requires assessment, not automatic rerun")
    # Maximum full-length voltage/threshold/ASC plus four spike-valued arrays and indices.
    # No deterministic repeat deduplication or compression benefit is assumed here.
    estimate = (7 + len(parameters["asc_tau_array"])) * len(stimulus) * 8
    require_space(store.directory, estimate + estimate // 100 + 1048576, store.reserve_bytes)
    store.checkpoint(name + "-started", {"identity": identity, "status": "started"})
    result = {"identity": identity}
    try:
        output, state = function(copy.deepcopy(parameters), stimulus.copy())
    except Exception as exc:  # noqa: BLE001 - every simulator exception is retained as evidence
        result.update(status="simulation-exception", error_type=type(exc).__name__, error=str(exc))
    else:
        if set(output) != set(FIELDS):
            raise ValueError("simulator output field coverage changed")
        result.update(
            status="complete", state=state, arrays={key: store.put(output[key]) for key in FIELDS}
        )
    store.checkpoint(name, result)
    return result


def attempt_outputs(store, result):
    if result["status"] != "complete":
        raise ValueError("failed attempts have no complete output")
    return {key: store.get(result["arrays"][key]) for key in FIELDS}
