"""Independent unit-explicit Na/K/M RK4 equations; no Brian or fitting imports."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import exprel

FIELDS = ('capacitance_pF', 'leak_nS', 'leak_reversal_mV', 'sodium_nS', 'potassium_nS',
          'm_current_nS', 'threshold_shift_mV', 'm_tau_max_ms')


def parameters(rows):
    if not rows or any(set(row) != set(FIELDS) for row in rows):
        raise ValueError('independent parameter scope differs')
    p = np.array([[row[k] for k in FIELDS] for row in rows], dtype=float)
    if (not np.isfinite(p).all() or np.any(p[:, [0, 1, 7]] <= 0)
        or np.any(p[:, [3, 4, 5]] < 0)):
        raise ValueError('invalid independent parameter values')
    return p


def kinetics(v_mV, shift_mV, tau_ms):
    """Rates in inverse milliseconds, time constants in milliseconds."""
    u = v_mV-shift_mV
    # Source vtrap(x,y) = y/exprel(x/y); exact analytic zero-argument limits.
    opening = np.array([.32*4/exprel((13-u)/4), .128*np.exp((17-u)/18),
                        .032*5/exprel((15-u)/5)])
    closing = np.array([.28*5/exprel((u-40)/5), 4/(1+np.exp((40-u)/5)),
                        .5*np.exp((10-u)/40)])
    slow_inf = 1/(1+np.exp(-(v_mV+35)/10))
    slow_tau = tau_ms/(3.3*np.exp((v_mV+35)/20)+np.exp(-(v_mV+35)/20))
    return opening, closing, slow_inf, slow_tau


def derivative(state, injected_pA, p):
    v, m, h, n, slow = state
    a, b, slow_inf, slow_tau = kinetics(v, p[:, 6], p[:, 7])
    # nS*mV = pA; pA/pF = mV/ms. No hidden area or density conversion.
    inward_pA = (p[:, 1]*(p[:, 2]-v)+p[:, 3]*m**3*h*(50-v)
                  +p[:, 4]*n**4*(-90-v)+p[:, 5]*slow*(-90-v)+injected_pA)
    change = np.empty_like(state)
    change[0] = inward_pA/p[:, 0]
    change[1:4] = a*(1-state[1:4])-b*state[1:4]
    change[4] = (slow_inf-slow)/slow_tau
    return change


def integrate(rows, sampled_current_pA, rate_hz, initial_mV):
    """Independent sampled-time RK4, returning only original-grid observed voltage."""
    p = parameters(rows)
    current = np.asarray(sampled_current_pA, dtype=float)
    initial = np.asarray(initial_mV, dtype=float)
    if (current.ndim != 2 or current.shape[1] != len(rows) or len(current) < 2
        or len(current) > 10_000_000 or current.size > 20_000_000
        or not np.isfinite(current).all() or initial.shape != (len(rows),)
        or not np.isfinite(initial).all() or not math.isfinite(rate_hz)
        or not 20000 < rate_hz <= 1_000_000):
        raise ValueError('invalid independent sampled inputs')
    step_ms = 1000/rate_hz
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        a, b, slow, _ = kinetics(initial, p[:, 6], p[:, 7])
        state = np.vstack([initial, a/(a+b), slow])
        voltage = np.empty(current.T.shape)
        for index in range(len(current)):
            if (not np.isfinite(state).all() or np.any(state[1:] < -1e-8)
                or np.any(state[1:] > 1+1e-8)):
                raise ValueError('independent observed-state failure')
            voltage[:, index] = state[0]
            if index == len(current)-1:
                break
            # All four stages use the same held current, not the next sample.
            injected = current[index]
            k1 = derivative(state, injected, p)
            k2 = derivative(state+step_ms*k1/2, injected, p)
            k3 = derivative(state+step_ms*k2/2, injected, p)
            k4 = derivative(state+step_ms*k3, injected, p)
            state = state+step_ms*(k1+2*k2+2*k3+k4)/6
    return voltage


def replay(rows, command_pA, rate_hz, holding_pA, warmup_s=1., refinement=1):
    """Independent baseline warmup and zero-order refinement without voltage targets."""
    command = np.asarray(command_pA, dtype=float)
    if (command.ndim != 1 or len(command) < 2 or not np.isfinite(command).all()
        or not math.isfinite(holding_pA) or not math.isfinite(warmup_s) or warmup_s <= 0
        or type(refinement) is not int or refinement not in (1, 2)):
        raise ValueError('invalid independent replay inputs')
    rate = rate_hz*refinement
    if not math.isfinite(rate) or not 20000 < rate <= 1000000:
        raise ValueError('invalid independent replay rate')
    count = round(warmup_s*rate)
    if abs(count-warmup_s*rate) > 1e-8:
        raise ValueError('nonintegral independent warmup')
    parameters(rows)
    nominal = command+holding_pA
    signal = np.concatenate((np.full(count, nominal[0]), np.repeat(nominal, refinement)))
    current = np.broadcast_to(signal[:, None], (len(signal), len(rows)))
    initial = [row['leak_reversal_mV'] for row in rows]
    voltage = integrate(rows, current, rate, initial)
    return voltage[:, count::refinement]
