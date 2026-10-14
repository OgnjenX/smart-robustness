"""Sealed search and replay controls; no source-file or held-out data access."""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np
from patchseq_isolated_hh import Parameters, simulate
from patchseq_response_features import characterize
from scipy.stats import qmc

DIMENSIONS = tuple(Parameters.__dataclass_fields__)


def candidates(search):
    count, bounds = search['candidates_per_stratum'], search['bounds']
    if type(count) is not int or count < 2 or count & (count-1) or count > 4096:
        raise ValueError('candidate count must be bounded power of two')
    if set(bounds) != set(DIMENSIONS):
        raise ValueError('candidate dimensions differ')
    lower, upper = [], []
    for name in DIMENSIONS:
        pair = bounds[name]
        if len(pair) != 2 or not all(math.isfinite(v) for v in pair) or pair[0] >= pair[1]:
            raise ValueError('invalid parameter bounds')
        lower.append(pair[0]); upper.append(pair[1])
    points = qmc.Sobol(len(DIMENSIONS), scramble=False).random_base2(count.bit_length()-1)
    return [Parameters(**dict(zip(DIMENSIONS, row, strict=True)))
            for row in qmc.scale(points, lower, upper)]


def replay(parameters, source_command_pA, rate_hz, holding_pA, warmup_s=1., refinement=1):
    """Initialize from model leak, warm up using nominal baseline, then full command."""
    command = np.asarray(source_command_pA, dtype=float)
    if (command.ndim != 1 or len(command) < 2 or not np.isfinite(command).all()
        or not math.isfinite(holding_pA) or not math.isfinite(warmup_s) or warmup_s <= 0
        or type(refinement) is not int or refinement not in (1, 2)):
        raise ValueError('invalid replay protocol')
    rate = rate_hz*refinement
    if not math.isfinite(rate) or rate <= 20000 or rate > 1000000:
        raise ValueError('unsupported replay rate')
    duration = warmup_s*rate
    warm_samples = round(duration)
    if abs(duration-warm_samples) > 1e-8:
        raise ValueError('warmup must be exact sample count')
    if not parameters:
        raise ValueError('no model parameters')
    nominal = command+holding_pA
    samples = np.r_[np.full(warm_samples, nominal[0]), np.repeat(nominal, refinement)]
    current = np.broadcast_to(samples[:, None], (len(samples), len(parameters)))
    voltage = simulate(parameters, current, rate, [p.leak_reversal_mV for p in parameters])
    # Compare/refilter on the original sample grid, never the observed voltage.
    return voltage[:, warm_samples::refinement]


def selected_candidate(rows, expected_count):
    """No outcome-dependent expansion or omitted failures; deterministic tie order."""
    if set(rows) != set(range(expected_count)) or len(rows) != expected_count:
        raise ValueError('incomplete candidate table')
    successful = [(row['loss'], tuple(row['parameters'][name] for name in DIMENSIONS), index)
                  for index, row in rows.items() if row['state'] == 'scored'
                  and math.isfinite(row['loss'])]
    if not successful:
        return None
    return min(successful)[2]


def candidate_dict(parameters):
    return asdict(parameters)


def numerical_controls(parameter, command, rate, holding, epoch, warmup=1.):
    """Selected model only; failures never trigger candidate replacement or refit."""
    original = replay([parameter], command, rate, holding, warmup)[0]
    repeated = replay([parameter], command, rate, holding, warmup)[0]
    refined = replay([parameter], command, rate, holding, warmup, refinement=2)[0]
    longer = replay([parameter], command, rate, holding, warmup*2)[0]
    features = [characterize(v, command, rate, epoch, holding)
                for v in (original, refined, longer)]
    error = float(np.max(np.abs(original-refined)))
    baseline = abs(features[0]['baseline_mean_mV']-features[2]['baseline_mean_mV'])
    gates = {'exact_repeat': bool(np.array_equal(original, repeated)),
             'refinement_voltage': error <= .1,
             'refinement_event_count': features[0]['detected_event_count'] == features[1]['detected_event_count'],
             'warmup_baseline': baseline <= .1,
             'warmup_event_count': features[0]['detected_event_count'] == features[2]['detected_event_count']}
    return {'gates': gates, 'maximum_refinement_voltage_error_mV': error,
            'warmup_baseline_difference_mV': baseline, 'passed': all(gates.values())}
