"""Source response observables; initial-event subset, not full IPFX replay."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
from scipy import signal


def events(v, rate, start, stop):
    if not math.isfinite(rate) or rate <= 20000:
        raise ValueError('source-rate-unsupported-for-fixed-10kHz-filter')
    b, a = signal.bessel(4, 20000/rate, 'low')
    filtered = signal.filtfilt(b, a, v, axis=0)
    derivative = np.diff(filtered)*rate/1000
    crossing = np.flatnonzero(np.diff((derivative[start:stop-1] >= 20.).astype('int8')) == 1)+start
    if len(crossing) > 1:
        crossing = np.array([crossing[0]]+[candidate for prior, candidate in pairwise(crossing)
                                          if np.any(derivative[prior:candidate] < 0)], dtype=int)
    if not len(crossing):
        return [], []
    ends = np.append(crossing[1:], stop-1)
    peaks = np.array([candidate+int(np.argmax(v[candidate:end]))
                      for candidate, end in zip(crossing, ends, strict=True)], dtype=int)
    separation = [bool(np.any(derivative[peak:candidate] < 0))
                  for peak, candidate in zip(peaks[:-1], crossing[1:], strict=True)]
    peaks = peaks[np.array(separation+[True])]
    crossing = crossing[np.array([True]+separation)]
    accepted = (v[peaks] >= -30.) & ((v[peaks]-v[crossing]) >= 2.)
    return crossing[accepted].astype(int).tolist(), peaks[accepted].astype(int).tolist()


def characterize(v, i, rate, epoch, holding):
    if (v.ndim != 1 or i.shape != v.shape or not np.isfinite(v).all() or not np.isfinite(i).all()
        or not math.isfinite(holding) or not math.isfinite(rate) or rate <= 20000):
        raise ValueError('invalid-source-response-arrays-or-holding')
    start, stop = epoch['step_start_index'], epoch['step_stop_index']
    pre_a, pre_b = epoch['pre_window']
    width, early = int(.1*rate), int(.01*rate)
    if not (0 <= pre_a < pre_b < start < stop < len(v)) or min(width, early) < 1 or width > stop-start:
        raise ValueError('unsupported-source-response-window')
    if not np.all(i[start:stop] == i[start]) or i[start] == i[0]:
        raise ValueError('source-step-current-disagreement')
    baseline, steady = float(np.mean(v[pre_a:pre_b])), float(np.mean(v[stop-width:stop]))
    delta = float(i[start]-i[0])
    candidates, peaks = events(v, rate, start, stop)
    times = [(index-start)/rate for index in candidates]
    intervals = np.diff(times)
    resistance = 1000*(steady-baseline)/delta if not candidates else None
    resistance_state = 'no-detected-events-apparent-proxy' if resistance is not None else 'event-bearing-not-passive-proxy'
    if resistance is not None and (not math.isfinite(resistance) or resistance <= 0):
        resistance, resistance_state = None, 'nonpositive-or-nonfinite-response-proxy'
    return {'baseline_mean_mV': baseline, 'early_step_mean_mV': float(np.mean(v[start:start+early])),
            'steady_step_mean_mV': steady, 'source_command_baseline_pA': float(i[0]),
            'source_command_step_pA': float(i[start]), 'source_command_delta_pA': delta,
            'nominal_holding_offset_pA': holding, 'nominal_input_baseline_pA': float(i[0]+holding),
            'nominal_input_step_pA': float(i[start]+holding), 'step_duration_s': (stop-start)/rate,
            'event_candidate_indices': candidates, 'event_peak_indices': peaks,
            'event_latencies_s': times, 'event_peak_voltages_mV': [float(v[p]) for p in peaks],
            'detected_event_count': len(candidates), 'detected_event_rate_hz': len(candidates)/((stop-start)/rate),
            'first_event_latency_s': times[0] if times else None,
            'last_to_first_interval_ratio': float(intervals[-1]/intervals[0]) if len(intervals) >= 2 else None,
            'apparent_steady_resistance_MOhm': resistance, 'resistance_proxy_state': resistance_state}
