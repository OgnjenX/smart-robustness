"""Independent source-array, fsum and deterministic quality-gate reconstruction."""

from __future__ import annotations

import math
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-append-quality-verification-registration-1401.yaml')


def require(ok):
    if not ok:
        raise ValueError('independent recording-quality evidence or result differs')


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


def mean(values):
    return math.fsum(map(float, values))/len(values)


def reconstruct(protocol, v, i, rate):
    if (v.ndim != 1 or i.shape != v.shape or len(v) < 2 or not math.isfinite(rate) or rate <= 0
        or not np.isfinite(v).all() or not np.isfinite(i).all()):
        return {'status': 'unknown', 'reason': 'invalid-array-or-clock'}
    if protocol == 'bath':
        end = int(rate*.005)
        if end < 1 or end > len(i):
            return {'status': 'unknown', 'reason': 'incomplete-electrode-zero-window'}
        return {'status': 'measured', 'electrode_zero_pA': mean(i[:end])*1e12, 'window': [0, end]}
    require(protocol in ('seal', 'breakin'))
    ups = np.nonzero(v[1:] > v[:-1])[0][1:]
    downs = np.nonzero(v[1:] < v[:-1])[0][1:]
    if len(ups) == 0 or len(ups) != len(downs) or any(a >= b for a, b in zip(ups, downs, strict=True)):
        return {'status': 'unknown', 'reason': 'unsupported-source-positive-pulse-edges'}
    width = int(.001/(1/rate))
    if width < 1:
        return {'status': 'unknown', 'reason': 'empty-one-ms-window'}
    pulses = []
    for a, b in zip(ups, downs, strict=True):
        up, down = int(a), int(b)
        base, steady = [up-1-width, up-1], [down-1-width, down-1]
        if base[0] < 0 or steady[0] <= up or down-1 <= up:
            return {'status': 'unknown', 'reason': 'incomplete-source-pulse-windows'}
        vb, ib = mean(v[slice(*base)]), mean(i[slice(*base)])
        vs, ist = mean(v[slice(*steady)]), mean(i[slice(*steady)])
        segment = i[up:down-1]
        peak = up+int(np.flatnonzero(segment == max(segment))[0])
        ds, dp = ist-ib, float(i[peak])-ib
        if ds == 0 or (protocol == 'breakin' and dp == 0):
            return {'status': 'unknown', 'reason': 'zero-response-current-difference'}
        rs = (vs-vb)/ds
        ra = (float(v[peak])-vb)/dp if dp else None
        if not math.isfinite(rs) or rs <= 0 or (protocol == 'breakin' and (not math.isfinite(ra) or ra <= 0)):
            return {'status': 'unknown', 'reason': 'nonfinite-or-nonpositive-resistance'}
        pulses.append({'up_index': up, 'down_index': down, 'baseline_window': base, 'steady_window': steady,
            'peak_index': peak if protocol == 'breakin' else None, 'stable_Ohm': rs, 'access_Ohm': ra if protocol == 'breakin' else None})
    result = {'status': 'measured', 'pulses': pulses}
    if protocol == 'seal':
        result['seal_GOhm'] = mean([p['stable_Ohm'] for p in pulses])*1e-9
    else:
        result['input_MOhm'] = mean([p['stable_Ohm'] for p in pulses])*1e-6
        result['access_MOhm'] = mean([p['access_Ohm'] for p in pulses])*1e-6
    return result


def gate(trials):
    selected = {}
    for name in ('bath', 'seal', 'breakin'):
        items = sorted((t for t in trials if t['protocol'] == name), key=lambda t: t['sweep_number'])
        if items and sum(t['sweep_number'] == items[-1]['sweep_number'] for t in items) == 1:
            selected[name] = items[-1]
    out = {'mandatory_measurement_gate': 'unknown', 'selected_sweeps': {name: t['sweep_number'] for name, t in selected.items()},
           'blowout_state': 'unassessed-no-recognized-pair', 'complete_cell_health_established': False}
    if len(selected) != 3 or any(t['measurement']['status'] != 'measured' for t in selected.values()):
        return {**out, 'reason': 'missing-or-invalid-required-measurement'}
    e0, seal = selected['bath']['measurement']['electrode_zero_pA'], selected['seal']['measurement']['seal_GOhm']
    ri, ra = selected['breakin']['measurement']['input_MOhm'], selected['breakin']['measurement']['access_MOhm']
    if not all(math.isfinite(x) for x in (e0, seal, ri, ra)) or ri <= 0:
        return {**out, 'reason': 'nonfinite-or-invalid-required-metric'}
    conditions = [(abs(e0) <= 200, 'electrode-zero-outside-200-pA'), (seal >= 1, 'seal-below-1-GOhm'),
        (1 <= ra <= 20, 'access-outside-1-through-20-MOhm'), (ra/ri <= .15, 'access-input-ratio-above-0.15')]
    failures = [label for ok, label in conditions if not ok]
    return {**out, 'mandatory_measurement_gate': 'fail' if failures else 'pass', 'access_input_ratio': ra/ri, 'failures': failures}


def read(file, item, h5py):
    node = file
    for part in (item['path']+'/data').strip('/').split('/'):
        require(isinstance(node, h5py.Group) and type(node.get(part, getlink=True)) is h5py.HardLink)
        node = node[part]
    require(isinstance(node, h5py.Dataset) and not node.is_virtual and not node.external
            and node.ndim == 1 and list(node.shape) == item['shape'] and node.size <= 10000000 and node.dtype.kind in 'iuf')
    props = node.id.get_create_plist()
    require(all(props.get_filter(k)[0] in (1, 2, 3) for k in range(props.get_nfilters())))
    return np.array(node[()], dtype=np.float64)*item['conversion'], node.size*8


def main():
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: digest(REG), Path(__file__): digest(__file__), Path(reg['result']): reg['result_sha256']}
    require(all(digest(p) == h for p, h in pins.items()))
    report = yaml.load(Path(reg['result']).read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(p): h for p, h in report['source_hashes'].items()})
    require(all(digest(p) == h for p, h in pins.items()))
    protocol = yaml.safe_load(Path(reg['original_protocol']).read_text())
    require(version('h5py') == protocol['scope']['parser'].split('==')[1])
    cases = yaml.load(Path(protocol['protocols']).read_text(), Loader=yaml.CSafeLoader)['records']
    acquired = yaml.safe_load(Path(protocol['acquisition']).read_text())
    receipts = {c['slot']: yaml.safe_load(Path(c['path']).read_text()) for c in acquired['receipts']}
    require(len(cases) == len(report['records']) == len(receipts) == 48)
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    import h5py
    summaries, charged = [], 0
    for source, saved in zip(cases, report['records'], strict=True):
        require(source['slot'] == saved['slot'] and source['source_cell_specimen_id'] == saved['source_cell_specimen_id'])
        body = Path(receipts[source['slot']]['complete_body_path'])
        require(digest(body) == saved['recording_sha256'] == receipts[source['slot']]['complete_file_SHA256'])
        local = yaml.safe_load((Path(reg['result']).parent/f'slot-{source["slot"]:03d}.yaml').read_text())
        require(local == saved)
        by_path = {s['path']: s for s in source['health_series']}
        require(len(source['health_pairs']) == len(saved['trials']))
        trials = []
        with h5py.File(body, 'r') as file:
            for pair, prior in zip(source['health_pairs'], saved['trials'], strict=True):
                require(pair['protocol'] in ('bath', 'seal', 'breakin'))
                if not pair['metadata_pair_pass']:
                    actual = {'protocol': pair['protocol'], 'sweep_number': pair['sweep_number'],
                              'measurement': {'status': 'unknown', 'reason': 'unsupported-health-pair-metadata',
                                              'failures': list(pair['failures'])}}
                    compare(prior, actual)
                    trials.append(actual)
                    continue
                data, rate = {}, None
                for path in pair['series_paths']:
                    item = by_path[path]
                    # Charge before reading, not after a potentially oversized read.
                    cost = item['shape'][0]*8
                    charged += cost
                    require(charged <= protocol['scope']['maximum_total_decoded_float64_bytes'])
                    data[item['role']], actual_cost = read(file, item, h5py)
                    require(actual_cost == cost)
                    rate = item['rate']
                actual = {'protocol': pair['protocol'], 'sweep_number': pair['sweep_number'],
                    'measurement': reconstruct(pair['protocol'], data['stimulus'], data['response'], rate)}
                compare(prior, actual)
                trials.append(actual)
        compare(saved['quality'], gate(trials))
        require(saved['specimen_eligibility_established'] is False and digest(body) == saved['recording_sha256'])
        summaries.append({'slot': source['slot'], 'verified_trials': len(trials), 'mandatory_gate': gate(trials)['mandatory_measurement_gate']})
    require(len({s['slot'] for s in summaries}) == 48 and charged == report['charged_float64_bytes'])
    require(all(digest(p) == h for p, h in pins.items()) and report['model_fitting_performed'] is False
            and report['specimen_eligibility_established'] is False)
    output = Path(reg['output'])
    output.parent.mkdir()
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'all-recording-quality-arrays-measurements-and-mandatory-gates-independently-reconstructed',
            'verifier_sha256': digest(__file__), 'source_hashes': {str(p): h for p, h in pins.items()},
            'records': summaries, 'charged_float64_bytes': charged, 'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('Independent reconstruction passed for all 162 retained health trials and 48 mandatory-gate outcomes')


if __name__ == '__main__':
    main()
