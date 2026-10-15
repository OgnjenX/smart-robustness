"""Diagnostic-only adaptive Na/K/M reference, exact sample-held command edges."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
from scipy.integrate import solve_ivp

FIELDS = ('capacitance_pF', 'leak_nS', 'leak_reversal_mV', 'sodium_nS', 'potassium_nS',
          'm_current_nS', 'threshold_shift_mV', 'm_tau_max_ms')


def rates(v, p):
    u = v-p['threshold_shift_mV']
    def trap(x):
        return 1. if x == 0. else x/math.expm1(x)
    opening = (1.28*trap((13-u)/4), .128*math.exp((17-u)/18), .16*trap((15-u)/5))
    closing = (1.4*trap((u-40)/5), 4/(1+math.exp((40-u)/5)), .5*math.exp((10-u)/40))
    slow_inf = 1/(1+math.exp(-(v+35)/10))
    slow_tau = p['m_tau_max_ms']/(3.3*math.exp((v+35)/20)+math.exp(-(v+35)/20))
    return opening, closing, slow_inf, slow_tau


def derivative(y, injected, p):
    v, m, h, n, slow = y
    a, b, inf, tau = rates(v, p)
    inward = (p['leak_nS']*(p['leak_reversal_mV']-v)
              +p['sodium_nS']*m**3*h*(50-v)
              +p['potassium_nS']*n**4*(-90-v)+p['m_current_nS']*slow*(-90-v)+injected)
    return np.array([inward/p['capacitance_pF'], a[0]*(1-m)-b[0]*m,
                     a[1]*(1-h)-b[1]*h, a[2]*(1-n)-b[2]*n, (inf-slow)/tau])


def integrate(p, current, rate_hz, method, rtol, voltage_atol, gate_atol,
              max_step_ms=.1, maximum_evaluations=5000000):
    """Time in ms, voltage mV, pA/pF=mV/ms. Never interpolate across input edges."""
    current = np.asarray(current, dtype=float)
    if (set(p) != set(FIELDS) or not all(math.isfinite(v) for v in p.values())
        or min(p[k] for k in ('capacitance_pF', 'leak_nS', 'm_tau_max_ms')) <= 0
        or min(p[k] for k in ('sodium_nS', 'potassium_nS', 'm_current_nS')) < 0
        or current.ndim != 1 or not 2 <= len(current) <= 10000000
        or not np.isfinite(current).all() or not math.isfinite(rate_hz)
        or not 20000 < rate_hz <= 1000000 or method not in ('DOP853', 'Radau')
        or not all(math.isfinite(v) and v > 0 for v in (rtol, voltage_atol, gate_atol, max_step_ms))
        or rtol < 1e-13 or type(maximum_evaluations) is not int or maximum_evaluations < 1):
        raise ValueError('invalid diagnostic reference inputs')
    v = p['leak_reversal_mV']
    a, b, slow, _ = rates(v, p)
    state = np.array([v, *(x/(x+y) for x, y in zip(a, b, strict=True)), slow])
    dt = 1000/rate_hz
    changes = np.flatnonzero(np.diff(current) != 0)+1
    edges = np.r_[0, changes[changes < len(current)-1], len(current)-1]
    states = np.empty((5, len(current)))
    evaluations, segments = 0, []
    for lo, hi in pairwise(edges):
        value = float(current[lo])
        def function(t, y, value=value):
            nonlocal evaluations
            evaluations += 1
            if evaluations > maximum_evaluations:
                raise RuntimeError('fixed reference evaluation budget exceeded')
            return derivative(y, value, p)
        grid = np.arange(lo, hi+1)*dt
        solved = solve_ivp(function, (float(grid[0]), float(grid[-1])), state,
                           method=method, t_eval=grid, rtol=rtol,
                           atol=np.array([voltage_atol]+[gate_atol]*4), max_step=max_step_ms)
        if (not solved.success or solved.y.shape != (5, hi-lo+1)
            or not np.isfinite(solved.y).all() or np.any(solved.y[1:] < -1e-8)
            or np.any(solved.y[1:] > 1+1e-8)):
            raise ValueError('reference integration failed or observed gate invalid: '+str(solved.message))
        states[:, lo:hi+1] = solved.y
        state = solved.y[:, -1].copy()
        segments.append({'first_index': int(lo), 'last_index': int(hi),
                         'constant_current_pA': value, 'nfev': int(solved.nfev)})
    return states[0], {'segments': segments, 'function_evaluations': evaluations,
                       'method': method, 'rtol': rtol, 'voltage_atol_mV': voltage_atol,
                       'gate_atol': gate_atol, 'max_step_ms': max_step_ms}


def replay(p, command, rate_hz, holding, *, warmup_s=1., **options):
    command = np.asarray(command, dtype=float)
    if (command.ndim != 1 or len(command) < 2 or not np.isfinite(command).all()
        or not math.isfinite(holding) or not math.isfinite(warmup_s) or warmup_s <= 0
        or not math.isfinite(rate_hz) or not 20000 < rate_hz <= 1000000):
        raise ValueError('invalid diagnostic replay command or warmup')
    warm = round(warmup_s*rate_hz)
    if abs(warm-warmup_s*rate_hz) > 1e-8:
        raise ValueError('nonintegral diagnostic warmup')
    nominal = command+holding
    current = np.r_[np.full(warm, nominal[0]), nominal]
    voltage, receipt = integrate(p, current, rate_hz, **options)
    return voltage[warm:], receipt
