"""Independent training response features; no collector or held-out data access."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml
from scipy import signal
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-fitting-target-verification-registration-1431.yaml')


def require(condition):
    if not condition:
        raise ValueError('independent training response or access evidence differs')


def compare(saved, actual):
    require(type(saved) is type(actual))
    if isinstance(actual, dict):
        require(saved.keys() == actual.keys())
        for key in actual:
            compare(saved[key], actual[key])
    elif isinstance(actual, list):
        require(len(saved) == len(actual))
        for a, b in zip(saved, actual, strict=True):
            compare(a, b)
    elif type(actual) is float:
        require(math.isfinite(saved) and math.isfinite(actual) and abs(saved-actual) <= 1e-9+1e-10*abs(actual))
    else:
        require(saved == actual)


def reconstruct(v, i, rate, epoch, holding):
    require(v.ndim == 1 and v.shape == i.shape and np.isfinite(v).all() and np.isfinite(i).all())
    require(math.isfinite(rate) and rate > 20000 and math.isfinite(holding))
    start, stop = epoch['step_start_index'], epoch['step_stop_index']
    pre_a, pre_b = epoch['pre_window']
    width, early = int(rate/10), int(rate/100)
    require(0 <= pre_a < pre_b < start < stop < len(v) and min(width, early) >= 1 and width <= stop-start)
    require(np.all(i[start:stop] == i[start]) and i[start] != i[0])
    b, a = signal.bessel(4, 20000/rate, 'low')
    filtered = signal.filtfilt(b, a, v, axis=0)
    derivative = np.diff(filtered)*rate/1000
    raw_candidates = [k-1 for k in range(start+1, stop-1) if derivative[k] >= 20. and derivative[k-1] < 20.]
    candidates = []
    for index, candidate in enumerate(raw_candidates):
        if index == 0 or np.any(derivative[raw_candidates[index-1]:candidate] < 0):
            candidates.append(candidate)
    peaks = []
    for index, candidate in enumerate(candidates):
        end = candidates[index+1] if index+1 < len(candidates) else stop-1
        peaks.append(candidate+int(np.argmax(v[candidate:end])))
    separators = [bool(np.any(derivative[peaks[k]:candidates[k+1]] < 0)) for k in range(max(0, len(peaks)-1))]
    if candidates:
        retained_peaks = [p for k, p in enumerate(peaks) if k == len(peaks)-1 or separators[k]]
        retained_candidates = [c for k, c in enumerate(candidates) if k == 0 or separators[k-1]]
        accepted = [(c, p) for c, p in zip(retained_candidates, retained_peaks, strict=True)
                    if v[p] >= -30. and v[p]-v[c] >= 2.]
    else:
        accepted = []
    candidates, peaks = [c for c, _ in accepted], [p for _, p in accepted]
    def mean(values):
        return math.fsum(map(float, values))/len(values)
    baseline, steady = mean(v[pre_a:pre_b]), mean(v[stop-width:stop])
    delta = float(i[start]-i[0])
    times = [(index-start)/rate for index in candidates]
    resistance = 1000*(steady-baseline)/delta if not candidates else None
    state = 'no-detected-events-apparent-proxy' if resistance is not None else 'event-bearing-not-passive-proxy'
    if resistance is not None and (not math.isfinite(resistance) or resistance <= 0):
        resistance, state = None, 'nonpositive-or-nonfinite-response-proxy'
    ratio = (times[-1]-times[-2])/(times[1]-times[0]) if len(times) >= 3 else None
    return {'baseline_mean_mV': baseline, 'early_step_mean_mV': mean(v[start:start+early]),
            'steady_step_mean_mV': steady, 'source_command_baseline_pA': float(i[0]),
            'source_command_step_pA': float(i[start]), 'source_command_delta_pA': delta,
            'nominal_holding_offset_pA': holding, 'nominal_input_baseline_pA': float(i[0]+holding),
            'nominal_input_step_pA': float(i[start]+holding), 'step_duration_s': (stop-start)/rate,
            'event_candidate_indices': candidates, 'event_peak_indices': peaks,
            'event_latencies_s': times, 'event_peak_voltages_mV': [float(v[p]) for p in peaks],
            'detected_event_count': len(candidates), 'detected_event_rate_hz': len(candidates)/((stop-start)/rate),
            'first_event_latency_s': times[0] if times else None, 'last_to_first_interval_ratio': ratio,
            'apparent_steady_resistance_MOhm': resistance, 'resistance_proxy_state': state}


def read(file, descriptor, h5py):
    node = file
    for part in (descriptor['path']+'/data').strip('/').split('/'):
        require(isinstance(node, h5py.Group) and type(node.get(part, getlink=True)) is h5py.HardLink)
        node = node[part]
    require(isinstance(node, h5py.Dataset) and node.ndim == 1 and list(node.shape) == descriptor['data_shape']
            and node.dtype.kind in 'iuf' and node.size <= 10000000 and not node.is_virtual and not node.external)
    properties = node.id.get_create_plist()
    require(all(properties.get_filter(k)[0] in (1, 2, 3) for k in range(properties.get_nfilters())))
    values = np.array(node[()], dtype='float64')
    values *= descriptor['conversion']*(1000 if descriptor['role'] == 'response' else 1e12)
    return values, node.size*8


def main():
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: digest(REG), Path(__file__): digest(__file__)}
    for key in ('result', 'collector', 'feature_code'):
        pins[Path(reg[key])] = reg[key+'_sha256']
    require(all(digest(p) == h for p, h in pins.items()))
    report = yaml.load(Path(reg['result']).read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(p): h for p, h in report['source_hashes'].items()})
    require(all(digest(p) == h for p, h in pins.items()))
    protocol = yaml.safe_load(Path(reg['original_protocol']).read_text())
    nominal = yaml.load(Path(protocol['nominal']).read_text(), Loader=yaml.CSafeLoader)['records']
    contexts = {(r['source_batch'], r['slot']): r for r in nominal}
    saved = {(r['source_batch'], r['slot']): r for r in report['records']}
    require(len(saved) == len(report['records']) == 96 and len(contexts) == 96)
    import os
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    import h5py
    summaries, opened, charged = [], [], 0
    for batch in protocol['batches']:
        data = {k: yaml.load(Path(batch[k]).read_text(), Loader=yaml.CSafeLoader)
                for k in ('selection', 'pairing', 'waveform_QC', 'independent_quality', 'acquisition')}
        pairs = {r['slot']: r for r in data['pairing']['records']}
        waves = {r['slot']: r for r in data['waveform_QC']['records']}
        health = {r['slot']: r for r in data['independent_quality']['records']}
        receipts = {r['slot']: r for r in data['acquisition']['receipts']}
        require(len(data['selection']['slots']) == len(pairs) == len(waves) == len(health) == len(receipts) == 48)
        for slot in data['selection']['slots']:
            key = batch['name'], slot['slot']
            outcome, pair_case, wave = saved[key], pairs[slot['slot']], waves[slot['slot']]
            for name in ('slot', 'source_group', 'layer_label', 'partition', 'frozen_group_id', 'source_cell_specimen_id'):
                require(outcome[name] == slot[name])
            require(outcome['source_batch'] == batch['name'] and len(outcome['sweeps']) == len(pair_case['sweeps']))
            if slot['partition'] == 'held-out':
                require(outcome['sweeps'] == [{'sweep_number': p['sweep_number'], 'state': 'held-out-not-characterized'}
                                            for p in pair_case['sweeps']])
                summaries.append({'source_batch': key[0], 'slot': key[1], 'partition': 'held-out', 'verified_training_features': 0})
                continue  # No receipt/body open or response feature access.
            require(slot['partition'] == 'fitting')
            context = {r['sweep_number']: r for r in contexts[key]['sweeps']}
            epochs = {r['sweep_number']: r for r in wave['sweeps']}
            source_series = {s['path']: s for s in pair_case['series']}
            expected_states = []
            for pair in pair_case['sweeps']:
                number = pair['sweep_number']
                state = ('recording-quality-not-passing' if health[slot['slot']]['mandatory_gate'] != 'pass'
                         else 'waveform-QC-not-passing' if epochs[number]['sweep_QC_pass'] is not True
                         else 'nominal-context-unsupported' if context[number]['nominal_context_supported'] is not True
                         else 'eligible')
                expected_states.append(state)
            verified_count = 0
            eligible = any(state == 'eligible' for state in expected_states)
            if eligible:
                item = receipts[slot['slot']]
                require(digest(item['path']) == item['sha256'])
                receipt = yaml.safe_load(Path(item['path']).read_text())
                body = Path(receipt['complete_body_path'])
                require(digest(body) == receipt['complete_file_SHA256'] == slot['expected_SHA256'])
                opened.append({'source_batch': key[0], 'slot': key[1], 'partition': 'fitting'})
                with h5py.File(body, 'r') as file:
                    for pair, recorded, state in zip(pair_case['sweeps'], outcome['sweeps'], expected_states, strict=True):
                        number = pair['sweep_number']
                        require(recorded['sweep_number'] == number)
                        if state != 'eligible':
                            require(recorded == {'sweep_number': number, 'state': state})
                            continue
                        require(recorded['state'] == 'fitting-response-characterized')
                        roles = {source_series[p]['role']: source_series[p] for p in pair['series_paths']}
                        values = {}
                        for role in ('response', 'stimulus'):
                            cost = roles[role]['data_shape'][0]*8
                            charged += cost
                            require(charged <= protocol['access']['maximum_total_float64_bytes'])
                            values[role], actual_cost = read(file, roles[role], h5py)
                            require(cost == actual_cost)
                        actual = reconstruct(values['response'], values['stimulus'], roles['response']['rate'],
                                             epochs[number], context[number]['nominal_holding_offset_pA'])
                        compare(recorded['features'], actual)
                        verified_count += 1
                require(digest(body) == receipt['complete_file_SHA256'])
            else:
                require(outcome['sweeps'] == [{'sweep_number': p['sweep_number'], 'state': state}
                                            for p, state in zip(pair_case['sweeps'], expected_states, strict=True)])
            summaries.append({'source_batch': key[0], 'slot': key[1], 'partition': 'fitting',
                              'verified_training_features': verified_count})
    require(opened == report['opened_response_files'] and charged == report['charged_float64_bytes'])
    require(len({(r['source_batch'], r['slot']) for r in summaries}) == 96)
    require(report['held_out_response_features_read'] is False and report['model_fitting_performed'] is False
            and report['frozen_baseline_modified'] is False)
    require(all(digest(p) == h for p, h in pins.items()))
    output = Path(reg['output'])
    output.parent.mkdir()
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'all-training-response-features-and-held-out-no-access-boundaries-independently-reconstructed',
                       'source_hashes': {str(p): h for p, h in pins.items()}, 'records': summaries,
                       'opened_response_files': opened, 'charged_float64_bytes': charged,
                       'held_out_response_features_read': False, 'collector_imported': False,
                       'model_fitting_performed': False, 'frozen_baseline_modified': False}, handle, sort_keys=False)
    print('Training response features independently reconstructed; all held-out boundaries verified')


if __name__ == '__main__':
    main()
