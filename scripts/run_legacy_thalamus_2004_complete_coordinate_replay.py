"""Replay the relay benchmark with coordinate-complete reversals and detector."""

from __future__ import annotations

import run_legacy_thalamus_2004_integrated_gate_replay as base

_BASE_MODEL_PARAMS = base._model_params


def _model_params(method: str) -> dict[str, object]:
    params = _BASE_MODEL_PARAMS(method)
    params.update(
        {
            "e_na_mV": 120.0,
            "e_k_mV": -20.0,
            "e_ca_mV": 250.0,
            "spike_event_voltage_offset_mV": -70.0,
        }
    )
    return params


def main() -> None:
    # Preserve the registered harness and replace only its parameter resolver
    # with the dimensionally equivalent internal-coordinate representation.
    base._model_params = _model_params
    base.main()


if __name__ == "__main__":
    main()
