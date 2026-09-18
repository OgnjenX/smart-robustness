"""Fixed receptor components for design registration 949; no network fitting.

This module does not modify the classic equations or attach an intervention to
any network. The isolated runner must supply measured rest and shared arrival
history. Peak-density matching is not integrated-current matching.
"""

from __future__ import annotations

import math

import numpy as np

from ..projections import Receptor
from .currents import biexponential_normalization
from .ports import SynapticPortSpec

ARMS = ("classic_ampa", "mixed_rest_block", "mixed_active_block")
DRIVE_FACTORS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
TOTAL_DENSITY = 0.09
NMDA_FRACTION = 0.2


def nmda_block(voltage_mV):
    """Classic SMART block at absolute physical voltage, overflow-safe."""
    voltage = np.asarray(voltage_mV, dtype=float)
    if not np.all(np.isfinite(voltage)):
        raise ValueError("voltage must be finite")
    return np.exp(-np.logaddexp(0.0, math.log(0.33) - voltage / 16.7))


def receptor_kernel(elapsed_ms, *, rise_ms: float, fall_ms: float):
    """Unit-peak causal alpha or biexponential kernel in milliseconds."""
    elapsed = np.asarray(elapsed_ms, dtype=float)
    if not np.all(np.isfinite(elapsed)):
        raise ValueError("elapsed time must be finite")
    if not (math.isfinite(rise_ms) and math.isfinite(fall_ms)):
        raise ValueError("kernel times must be finite")
    if not 0 < rise_ms <= fall_ms:
        raise ValueError("require 0 < rise <= fall")
    t = np.maximum(elapsed, 0.0)
    if rise_ms == fall_ms:
        ratio = t / rise_ms
        result = np.e * ratio * np.exp(-ratio)
    else:
        scale = biexponential_normalization(rise_ms, fall_ms)
        result = scale * (np.exp(-t / fall_ms) - np.exp(-t / rise_ms))
    return np.where(elapsed >= 0, result, 0.0)


def arrival_gate(time_ms, arrivals_ms, *, rise_ms: float, fall_ms: float):
    """KInNeSS two-most-recent-event union for unit-resource assay events.

    Both receptors use the same arrivals and drive factor. This is not an
    all-history sum: older events are discarded as in the classic connector.
    A drive factor belongs outside this union, as a fixed projection weight.
    """
    times = np.asarray(time_ms, dtype=float)
    arrivals = np.asarray(arrivals_ms, dtype=float)
    if times.ndim != 1 or arrivals.ndim != 1:
        raise ValueError("times and arrivals must be one-dimensional")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(arrivals)):
        raise ValueError("times and arrivals must be finite")
    if np.any(np.diff(arrivals) <= 0):
        raise ValueError("arrivals must be strictly increasing")
    # Also validate kinetics for an empty event train.
    receptor_kernel(np.array([0.0]), rise_ms=rise_ms, fall_ms=fall_ms)
    recent = np.searchsorted(arrivals, times, side="right") - 1
    waves = []
    for age in (0, 1):
        index = recent - age
        wave = np.zeros_like(times)
        valid = index >= 0
        wave[valid] = receptor_kernel(
            times[valid] - arrivals[index[valid]], rise_ms=rise_ms, fall_ms=fall_ms
        )
        waves.append(wave)
    return waves[0] + waves[1] - waves[0] * waves[1]


def isolated_receptor_ports(
    arm: str,
    *,
    resting_distal_mV: float,
    distal_area_cm2: float,
    proximal_area_cm2: float,
) -> tuple[SynapticPortSpec, ...]:
    """Create assay-only distal components and nS-matched proximal probe.

    Existing baseline ports remain present, silent, and unchanged. The runner
    drives these dedicated ports directly; none is a new network projection.
    The fixed-block factor is folded into NMDA density because the existing
    compiler already supports voltage-independent ports without any new state.
    """
    if arm not in ARMS:
        raise ValueError(f"unregistered receptor arm: {arm}")
    values = (resting_distal_mV, distal_area_cm2, proximal_area_cm2)
    if not all(math.isfinite(x) for x in values):
        raise ValueError("rest and areas must be finite")
    if min(distal_area_cm2, proximal_area_cm2) <= 0:
        raise ValueError("compartment areas must be positive")

    def port(name, compartment, receptor, density, rise, fall, block=False):
        return SynapticPortSpec(
            name=name,
            record_id=f"assay949.{name}",
            compartment=compartment,
            receptor=receptor,
            reversal_mV=0.0,
            conductance_density_mS_cm2=density,
            rise_ms=rise,
            fall_ms=fall,
            normalization=biexponential_normalization(rise, fall) if rise != fall else 1.0,
            voltage_block=block,
        )

    fast_density = TOTAL_DENSITY if arm == "classic_ampa" else TOTAL_DENSITY * 0.8
    ports = [port("assay_fast", "distal_dendrite", Receptor.AMPA, fast_density, 2.0, 2.0)]
    if arm != "classic_ampa":
        density = TOTAL_DENSITY * NMDA_FRACTION
        if arm == "mixed_rest_block":
            density *= float(nmda_block(resting_distal_mV))
        ports.append(
            port(
                "assay_slow",
                "distal_dendrite",
                Receptor.NMDA,
                density,
                0.7,
                80.0,
                block=arm == "mixed_active_block",
            )
        )
    ports.append(
        port(
            "assay_proximal",
            "proximal_dendrite",
            Receptor.AMPA,
            TOTAL_DENSITY * distal_area_cm2 / proximal_area_cm2,
            2.0,
            2.0,
        )
    )
    return tuple(ports)
