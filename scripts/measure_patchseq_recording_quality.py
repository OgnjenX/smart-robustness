"""Sealed source-index VC measurements; missing evidence never becomes zero."""

from __future__ import annotations

import math
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml
from run_patchseq_waveform_QC import read_array
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-recording-quality-registration-1364.yaml')


def measure(protocol, v, i, rate):
    if (v.ndim != 1 or i.shape != v.shape or len(v) < 2 or not math.isfinite(rate) or rate <= 0
        or not np.isfinite(v).all() or not np.isfinite(i).all()):
        return {'status': 'unknown', 'reason': 'invalid-array-or-clock'}
    if protocol == 'bath':
        count = int(.005*rate)
        if count < 1 or count > len(i):
            return {'status': 'unknown', 'reason': 'incomplete-electrode-zero-window'}
        return {'status': 'measured', 'electrode_zero_pA': float(np.mean(i[:count])*1e12), 'window': [0, count]}
    if protocol not in ('seal', 'breakin'):
        return {'status': 'unknown', 'reason': 'unregistered-measurement-protocol'}
    delta = np.diff(v)
    ups, downs = np.flatnonzero(delta > 0)[1:], np.flatnonzero(delta < 0)[1:]
    if not len(ups) or len(ups) != len(downs) or np.any(ups >= downs):
        return {'status': 'unknown', 'reason': 'unsupported-source-positive-pulse-edges'}
    width = int(.001/(1/rate))
    if width < 1:
        return {'status': 'unknown', 'reason': 'empty-one-ms-window'}
    pulses = []
    for up, down in zip(ups, downs, strict=True):
        up, down = int(up), int(down)
        base, steady = [up-1-width, up-1], [down-1-width, down-1]
        if base[0] < 0 or steady[0] <= up or down-1 <= up:
            return {'status': 'unknown', 'reason': 'incomplete-source-pulse-windows'}
        vb, ib = float(np.mean(v[slice(*base)])), float(np.mean(i[slice(*base)]))
        vs, ist = float(np.mean(v[slice(*steady)])), float(np.mean(i[slice(*steady)]))
        peak = up+int(np.argmax(i[up:down-1]))
        di, dip = ist-ib, float(i[peak])-ib
        if di == 0 or (protocol == 'breakin' and dip == 0):
            return {'status': 'unknown', 'reason': 'zero-response-current-difference'}
        stable = (vs-vb)/di
        access = (float(v[peak])-vb)/dip if dip != 0 else None
        if not math.isfinite(stable) or stable <= 0 or (protocol == 'breakin' and (not math.isfinite(access) or access <= 0)):
            return {'status': 'unknown', 'reason': 'nonfinite-or-nonpositive-resistance'}
        pulses.append({'up_index': up, 'down_index': down, 'baseline_window': base, 'steady_window': steady,
            'peak_index': peak if protocol == 'breakin' else None, 'stable_Ohm': stable,
            'access_Ohm': access if protocol == 'breakin' else None})
    result = {'status': 'measured', 'pulses': pulses}
    if protocol == 'seal':
        result['seal_GOhm'] = float(np.mean([p['stable_Ohm'] for p in pulses])*1e-9)
    else:
        result['input_MOhm'] = float(np.mean([p['stable_Ohm'] for p in pulses])*1e-6)
        result['access_MOhm'] = float(np.mean([p['access_Ohm'] for p in pulses])*1e-6)
    return result


def score(trials):
    selected = {}
    for name in ('bath', 'seal', 'breakin'):
        available = [t for t in trials if t['protocol'] == name]
        if available:
            latest = max(t['sweep_number'] for t in available)
            chosen = [t for t in available if t['sweep_number'] == latest]
            if len(chosen) == 1:
                selected[name] = chosen[0]
    result = {'mandatory_measurement_gate': 'unknown', 'selected_sweeps': {k: v['sweep_number'] for k, v in selected.items()},
              'blowout_state': 'unassessed-no-recognized-pair', 'complete_cell_health_established': False}
    if len(selected) != 3 or any(t['measurement']['status'] != 'measured' for t in selected.values()):
        result['reason'] = 'missing-or-invalid-required-measurement'
        return result
    e0 = selected['bath']['measurement']['electrode_zero_pA']
    seal = selected['seal']['measurement']['seal_GOhm']
    ri, ra = (selected['breakin']['measurement'][k] for k in ('input_MOhm', 'access_MOhm'))
    if not all(math.isfinite(v) for v in (e0, seal, ri, ra)) or ri <= 0:
        return {**result, 'reason': 'nonfinite-or-invalid-required-metric'}
    failures = []
    if abs(e0) > 200:
        failures.append('electrode-zero-outside-200-pA')
    if seal < 1:
        failures.append('seal-below-1-GOhm')
    if not 1 <= ra <= 20:
        failures.append('access-outside-1-through-20-MOhm')
    if ra/ri > .15:
        failures.append('access-input-ratio-above-0.15')
    return {**result, 'mandatory_measurement_gate': 'fail' if failures else 'pass',
            'access_input_ratio': ra/ri, 'failures': failures}


def main():
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: digest(REG), Path(__file__): digest(__file__)}
    for name in ('protocols', 'independent_protocols', 'acquisition'):
        pins[Path(reg[name])] = reg[name+'_sha256']
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('registered input changed')
    independent = yaml.load(Path(reg['independent_protocols']).read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(p): h for p, h in independent['source_hashes'].items()})
    # Explicitly pin the reused bounded array decoder before outcome execution.
    pins[Path('scripts/run_patchseq_waveform_QC.py')] = '63617dbe160137c99094c502cf5abeed9940be8d8e85cd3b2649291cc475ea47'
    if any(digest(p) != h for p, h in pins.items()) or version('h5py') != reg['scope']['parser'].split('==')[1]:
        raise ValueError('source or parser differs')
    protocols = yaml.load(Path(reg['protocols']).read_text(), Loader=yaml.CSafeLoader)['records']
    acquired = yaml.safe_load(Path(reg['acquisition']).read_text())
    receipts = {x['slot']: yaml.safe_load(Path(x['path']).read_text()) for x in acquired['receipts']}
    if len(protocols) != len(receipts) or len(protocols) != 48:
        raise ValueError('fixed cohort differs')
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    import h5py
    output = Path(reg['output'])
    output.parent.mkdir()
    records, budget = [], [0]
    for case in protocols:
        body = Path(receipts[case['slot']]['complete_body_path'])
        before = digest(body)
        if before != receipts[case['slot']]['complete_file_SHA256']:
            raise ValueError('recording changed')
        series = {s['path']: s for s in case['health_series']}
        trials = []
        with h5py.File(body, 'r') as file:
            for pair in case['health_pairs']:
                if not pair['metadata_pair_pass'] or pair['protocol'] not in ('bath', 'seal', 'breakin'):
                    raise ValueError('unregistered health trial')
                roles = {series[p]['role']: series[p] for p in pair['series_paths']}
                try:
                    values = {}
                    for role, item in roles.items():
                        raw = read_array(file, {'path': item['path'], 'data_shape': item['shape']}, h5py,
                                         budget, reg['scope']['maximum_total_decoded_float64_bytes'])
                        values[role] = raw*item['conversion']
                    outcome = measure(pair['protocol'], values['stimulus'], values['response'], roles['response']['rate'])
                except (ValueError, TypeError, OSError) as exc:
                    outcome = {'status': 'unknown', 'reason': 'bounded-array-read-error', 'error': str(exc)}
                trials.append({'protocol': pair['protocol'], 'sweep_number': pair['sweep_number'], 'measurement': outcome})
        if digest(body) != before:
            raise ValueError('recording changed during health measurement')
        entry = {'slot': case['slot'], 'source_cell_specimen_id': case['source_cell_specimen_id'],
                 'recording_sha256': before, 'trials': trials, 'quality': score(trials), 'specimen_eligibility_established': False}
        with (output.parent/f'slot-{case["slot"]:03d}.yaml').open('x') as handle:
            yaml.safe_dump(entry, handle, sort_keys=False)
        records.append(entry)
    if any(digest(p) != h for p, h in pins.items()) or len({r['slot'] for r in records}) != 48:
        raise ValueError('source or scope changed')
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'protocol-specific-health-measurement-outcomes-retained-independent-check-pending',
            'source_hashes': {str(p): h for p, h in pins.items()}, 'records': records,
            'charged_float64_bytes': budget[0], 'model_fitting_performed': False,
            'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('Retained protocol-specific VC measurements and bounded quality outcomes for all 48 files')


if __name__ == '__main__':
    main()
