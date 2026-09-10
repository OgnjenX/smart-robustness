"""Replay the legacy relay with every voltage gate on KInNeSS's integrated V."""

from __future__ import annotations

import run_legacy_thalamus_2004_internal_coordinate_replay as base

_BASE_MODEL_PARAMS = base._model_params


def _model_params(method: str) -> dict[str, object]:
    params = _BASE_MODEL_PARAMS(method)
    params["calcium_voltage_coordinate"] = "integrated_voltage"
    return params


def main() -> None:
    # The preregistered base harness resolves this function in its own module
    # when constructing each arm. Replacing only that resolver preserves every
    # protocol, metric, gate, and output rule while crossing the one audited
    # calcium-coordinate semantic.
    base._model_params = _model_params
    base.main()


if __name__ == "__main__":
    main()
