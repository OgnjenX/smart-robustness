"""Raw, unit-labelled recording for the isolated apical assay.

Only the sealed runner may call ``simulate``. Archive helpers and zero-time
construction tests can be exercised independently before any outcomes exist.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np

from ..standalone import build_and_run_cpp_standalone
from .active_apical_isolated import build_isolated_apical_assay


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_trace(path: Path, arrays: dict[str, np.ndarray]) -> str:
    """Publish a complete compressed archive without overwriting prior data.

    Use a same-directory temporary file and an exclusive hard link. A crash
    before publication leaves no partial checkpoint at the destination.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("object arrays are forbidden in raw traces")
    descriptor, temporary = tempfile.mkstemp(prefix=".apical-trace-", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return file_sha256(path)


def load_trace(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    path = Path(path)
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"raw trace fingerprint mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def exact_trace_repeat(first: dict, second: dict) -> bool:
    """Require identical named arrays including dtype, shape, and all samples."""
    return first.keys() == second.keys() and all(
        np.asarray(first[key]).dtype == np.asarray(second[key]).dtype
        and np.array_equal(first[key], second[key])
        for key in first
    )


def rest_assessment(arrays: dict[str, np.ndarray], dt_ms: float) -> dict:
    t = arrays["time_ms"]
    mask = (t >= 900.0 - dt_ms / 10) & (t < 1000.0 - dt_ms / 10)
    expected = 900.0 + np.arange(round(100.0 / dt_ms)) * dt_ms
    if t[mask].shape != expected.shape or not np.allclose(t[mask], expected, atol=1e-8, rtol=0):
        raise ValueError("incomplete resting-state window")
    distal = arrays["v_distal_dendrite_mV"][:, mask]
    finite = bool(all(np.all(np.isfinite(a)) for a in arrays.values()))
    spikes = arrays["spike_time_ms"]
    count = int(np.count_nonzero((spikes >= 900.0) & (spikes < 1000.0)))
    spread = float(np.max(np.ptp(distal, axis=1)))
    # Cells are identical before stimulation. Require this rather than average
    # away an unexpected condition-dependent resting-state difference.
    same_cells = bool(np.all(distal == distal[0]))
    return {
        "all_recorded_states_finite": finite,
        "maximum_distal_peak_to_peak_mV": spread,
        "terminal_soma_spike_count": count,
        "all_cells_identical_at_rest": same_cells,
        "resting_distal_mV": float(np.mean(distal[0])),
        "pass": finite and spread <= 0.1 and count == 0 and same_cells,
    }


def simulate(
    *, baseline, arm: str, resting_distal_mV: float, dt_ms: float, rest_only: bool = False
) -> dict[str, np.ndarray]:
    """Execute one independent, fresh C++ assay and retain physical-unit arrays."""
    import brian2 as brian

    with tempfile.TemporaryDirectory(prefix="smart-apical-isolated-", dir="/private/tmp") as temp:
        try:
            brian.set_device("cpp_standalone", directory=temp, build_on_run=False)
            brian.start_scope()
            assay = build_isolated_apical_assay(
                baseline=baseline,
                arm=arm,
                resting_distal_mV=resting_distal_mV,
                dt_ms=dt_ms,
                brian=brian,
            )
            assay.network.run(900 * brian.ms)
            assay.state_monitor.active = True
            assay.network.run((100 if rest_only else 500) * brian.ms)
            build_and_run_cpp_standalone(brian, temp)
            state = assay.state_monitor
            arrays = {
                "time_ms": np.array(state.t / brian.ms),
                "spike_time_ms": np.array(assay.spike_monitor.t / brian.ms),
                "spike_cell_index": np.array(assay.spike_monitor.i),
                "dt_ms": np.array(dt_ms),
            }
            for name in state.record_variables:
                value = getattr(state, name)
                if name.startswith("v_"):
                    key, value = name + "_mV", value / brian.mV
                elif name.startswith("i_"):
                    key, value = name + "_pA", value / brian.pA
                else:
                    key = name
                arrays[key] = np.array(value)
            return arrays
        finally:
            brian.device.reinit()
            brian.set_device("runtime")
