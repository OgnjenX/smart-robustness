"""Lossless, unit-labeled relay monitor export for mechanistic diagnostics."""

from hashlib import sha256
from pathlib import Path

import numpy as np


def write_relay_trace(monitor, path, *, stimulus_start_ms, condition, fingerprint, brian):
    """Preserve cue and trial samples; never overwrite an existing trace.

    Negative relative times belong to the pre-stimulus recording period.
    Traces are state observations, not a causal test of calcium contribution.
    """
    output = Path(path)
    variables = tuple(monitor.record_variables)
    values = {}
    units = []
    for name in variables:
        if name.startswith("v_"):
            unit, label = brian.mV, "mV"
        elif name.startswith("i_"):
            unit, label = brian.pA, "pA"
        else:
            unit, label = 1, "dimensionless"
        values[name] = np.asarray(getattr(monitor, name) / unit).copy()
        units.append(label)
    values.update(
        schema_version=np.asarray(1),
        time_ms=np.asarray(monitor.t / brian.ms) - stimulus_start_ms,
        cell_indices=np.asarray(monitor.record, dtype=int),
        variable_names=np.asarray(variables),
        variable_units=np.asarray(units),
        stimulus_start_ms=np.asarray(stimulus_start_ms),
        condition=np.asarray(str(condition)),
        runtime_fingerprint=np.asarray(fingerprint),
        monitor_when=np.asarray(monitor.when),
        monitor_order=np.asarray(monitor.order),
    )
    # Exclusive creation also protects against a file appearing while a long
    # simulation is running. Let missing-parent errors remain explicit.
    with output.open("xb") as stream:
        np.savez_compressed(stream, **values)
    return sha256(output.read_bytes()).hexdigest()
