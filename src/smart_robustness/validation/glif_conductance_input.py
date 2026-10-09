"""Validated SI-unit inputs for the registered step-held GLIF extension."""

from __future__ import annotations

import numpy as np


def validate_inputs(parameters, bias, excitation, inhibition, *, e_exc, e_inh):
    """Reject malformed inputs before either backend mutates its state."""
    dt = parameters["dt"]
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("timestep must be finite and positive")
    if not all(np.isfinite(v) for v in (parameters["El_reference"], e_exc, e_inh)):
        raise ValueError("voltage coordinate and reversals must be finite")
    arrays = tuple(np.asarray(a, dtype=float) for a in (bias, excitation, inhibition))
    if any(a.ndim != 1 for a in arrays) or len({a.shape for a in arrays}) != 1:
        raise ValueError("inputs must have identical one-dimensional shapes")
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("inputs must be finite")
    if any(np.any(a < 0) for a in arrays[1:]):
        raise ValueError("conductances must be nonnegative")
    return arrays


def effective_current(bias, excitation, inhibition, absolute_voltage, e_exc, e_inh):
    """Current held through voltage and adaptive-threshold updates at one step."""
    if excitation == 0 and inhibition == 0:
        return bias  # Preserve the current-only null arithmetic exactly.
    return bias + excitation * (e_exc - absolute_voltage) + inhibition * (e_inh - absolute_voltage)


def run_extended_native(neuron, parameters, bias, excitation, inhibition, *, e_exc, e_inh):
    """Wrap a fresh pinned native neuron; restore observers even if it raises.

    Only the effective injected current is changed. The original run loop owns
    spike cuts and resets. No native numerical source is edited. The supplied
    neuron must be freshly initialized from these parameters for each attempt.
    """
    bias, excitation, inhibition = validate_inputs(
        parameters, bias, excitation, inhibition, e_exc=e_exc, e_inh=e_inh
    )
    original_dynamics, original_reset = neuron.dynamics, neuron.reset
    flags = []

    def dynamics(voltage, threshold, currents, injection, time_step, spike_steps):
        absolute_voltage = voltage + parameters["El_reference"]
        # Deliberately independent arithmetic from the candidate helper.
        if excitation[time_step] == 0 and inhibition[time_step] == 0:
            held_input = injection
        else:
            held_input = (
                injection + excitation[time_step] * (e_exc - absolute_voltage)
                + inhibition[time_step] * (e_inh - absolute_voltage)
            )
        return original_dynamics(
            voltage, threshold, currents, held_input, time_step, spike_steps
        )

    def reset(*args):
        output = original_reset(*args)
        flags.append(bool(output[-1]))
        return output

    neuron.dynamics, neuron.reset = dynamics, reset
    try:
        output = neuron.run(bias.copy())
    finally:
        neuron.dynamics, neuron.reset = original_dynamics, original_reset
    return output, {"bad_reset_stop": any(flags), "native_code_called": True}
