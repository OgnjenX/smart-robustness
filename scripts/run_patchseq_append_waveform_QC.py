"""Bounded, sealed sweep QC; no model fitting or baseline modifications."""

from __future__ import annotations

import math
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-append-waveform-QC-registration-1388.yaml')


def qc(v, i, rate):
    result = {'long_square_candidate': False, 'sweep_QC_pass': False, 'failures': []}
    if (v.ndim != 1 or i.ndim != 1 or v.shape != i.shape or len(v) < 2
        or not math.isfinite(rate) or rate <= 0):
        result['failures'].append('invalid-array-or-rate')
        return result
    if not np.isfinite(v).all() or not np.isfinite(i).all():
        result['failures'].append('nonfinite-samples')
        return result
    transitions = np.flatnonzero(np.diff(i)) + 1
    result['command_transition_count'] = len(transitions)
    if len(transitions) not in (2, 4):
        result['failures'].append('not-supported-single-step-waveform')
        return result
    if len(transitions) == 4:
        a, b = transitions[:2]
        if a/rate >= 0.4 or (b-a)/rate >= 0.4 or i[b] != i[0]:
            result['failures'].append('unsupported-test-pulse')
            return result
    start, stop = (int(x) for x in transitions[-2:])
    if (stop-start)/rate < 0.5 or i[stop] != i[0] or i[start] == i[0]:
        result['failures'].append('not-long-square-returning-to-baseline')
        return result
    result.update(long_square_candidate=True, step_start_index=start, step_stop_index=stop,
                  step_duration_s=(stop-start)/rate, command_baseline_pA=float(i[0]),
                  command_step_pA=float(i[start]), command_delta_pA=float(i[start]-i[0]))
    window, noise = int(0.5*rate), int(0.0015*rate)
    pre = (start-1-window, start-1)
    post = (len(v)-1-window, len(v)-1)
    if noise < 1 or window < 1 or pre[0] < 0 or post[0] < stop:
        result['failures'].append('incomplete-baseline-window')
        return result
    if not all(np.all(i[a:b] == i[0]) for a, b in (pre, post)):
        result['failures'].append('baseline-window-contains-command-transition')
        return result
    def centered_rms(values):
        return float(np.sqrt(np.mean((values-np.mean(values))**2)))
    pre_v, post_v = v[slice(*pre)], v[slice(*post)]
    metrics = {'pre_noise_RMS_mV': centered_rms(pre_v[:noise]),
               'post_noise_RMS_mV': centered_rms(post_v[-noise:]),
               'slow_noise_RMS_mV': centered_rms(pre_v),
               'absolute_mean_drift_mV': float(abs(np.mean(pre_v)-np.mean(post_v)))}
    result.update(metrics=metrics, pre_window=list(pre), post_window=list(post))
    limits = {'pre_noise_RMS_mV': 0.07, 'post_noise_RMS_mV': 0.07,
              'slow_noise_RMS_mV': 0.5, 'absolute_mean_drift_mV': 1.0}
    for name, value in metrics.items():
        if not math.isfinite(value) or value > limits[name]:
            result['failures'].append(name+'-exceeds-limit')
    result['sweep_QC_pass'] = not result['failures']
    return result


def hard_node(file, path, h5py):
    node = file
    for part in path.strip('/').split('/'):
        if not isinstance(node, h5py.Group) or type(node.get(part, getlink=True)) is not h5py.HardLink:
            raise ValueError('non-hard-or-missing-path')
        node = node[part]
    return node


def safe_dataset(node, h5py):
    if not isinstance(node, h5py.Dataset) or node.external or node.is_virtual:
        raise ValueError('external-virtual-or-nondataset')
    props = node.id.get_create_plist()
    if any(props.get_filter(index)[0] not in (1, 2, 3) for index in range(props.get_nfilters())):
        raise ValueError('unsupported-HDF5-filter')


def read_array(file, item, h5py, budget, limit):
    node = hard_node(file, item['path']+'/data', h5py)
    safe_dataset(node, h5py)
    if (node.ndim != 1 or list(node.shape) != item['data_shape'] or node.dtype.kind not in 'iuf'
        or node.size > 10000000):
        raise ValueError('unsupported-array-shape-type-or-size')
    cost = node.size*8
    if budget[0]+cost > limit:
        raise ValueError('total-decoded-array-budget-exceeded')
    budget[0] += cost
    return np.asarray(node[()], dtype=np.float64)


def instruments(file, path, h5py):
    group = hard_node(file, path, h5py)
    values = {}
    for name in ('gain', 'bias_current', 'bridge_balance', 'capacitance_compensation'):
        if name not in group:
            values[name] = {'present': False, 'undefined': 'absent'}
            continue
        try:
            node = hard_node(file, path+'/'+name, h5py)
            safe_dataset(node, h5py)
            if (node.shape not in ((), (1,)) or node.id.get_storage_size() > 4096
                or node.id.get_create_plist().get_nfilters() or node.dtype.kind not in 'iuf'):
                raise ValueError('unsupported-instrument-scalar')
            value = node[()].item()
            values[name] = {'present': True, 'dtype': str(node.dtype), 'value': value} if math.isfinite(value) else {
                'present': True, 'dtype': str(node.dtype), 'undefined': 'nonfinite'}
        except (ValueError, TypeError, OSError) as exc:
            values[name] = {'present': True, 'undefined': str(exc)}
    return values


def main():
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: digest(REG), Path(__file__): digest(__file__)}
    for key in ('pairing', 'independent_pairing', 'acquisition'):
        pins[Path(reg[key])] = reg[key+'_sha256']
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('registered evidence changed')
    pairing = yaml.load(Path(reg['pairing']).read_text(), Loader=yaml.CSafeLoader)
    independent = yaml.load(Path(reg['independent_pairing']).read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(p): h for p, h in independent['source_hashes'].items()})
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('source hashes differ')
    if version('h5py') != reg['read_policy']['parser'].split('==')[1]:
        raise ValueError('parser version differs')
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    import h5py
    acquisition = yaml.safe_load(Path(reg['acquisition']).read_text())
    receipts = {x['slot']: yaml.safe_load(Path(x['path']).read_text()) for x in acquisition['receipts']}
    output = Path(reg['output'])
    output.parent.mkdir()
    # Exclusive output creation occurs before reading samples: no silent replay.
    with (output.parent/'execution-started.yaml').open('x') as handle:
        yaml.safe_dump({'runner_sha256': digest(__file__), 'registration_sha256': digest(REG)}, handle)
    if len(pairing['records']) != 48:
        raise ValueError('fixed cohort changed')
    budget, cases = [0], []
    for case in pairing['records']:
        receipt = receipts[case['slot']]
        body = Path(receipt['complete_body_path'])
        before = digest(body)
        if before != receipt['complete_file_SHA256']:
            raise ValueError('recording integrity differs')
        outcomes = []
        series = {x['path']: x for x in case['series']}
        with h5py.File(body, 'r') as file:
            if file.attrs.get('nwb_version') != '2.2.5':
                raise ValueError('unexpected-NWB-version')
            for pair in case['sweeps']:
                result = {'sweep_number': pair['sweep_number'], 'sweep_QC_pass': False}
                try:
                    items = {series[p]['role']: series[p] for p in pair['series_paths']}
                    response, stimulus = items['response'], items['stimulus']
                    v = read_array(file, response, h5py, budget, reg['read_policy']['maximum_total_float64_decoded_bytes'])
                    i = read_array(file, stimulus, h5py, budget, reg['read_policy']['maximum_total_float64_decoded_bytes'])
                    v *= response['conversion']*1000
                    i *= stimulus['conversion']*1e12
                    result.update(qc(v, i, response['rate']), description=response['description'],
                                  instrument_metadata=instruments(file, response['path'], h5py))
                except (ValueError, TypeError, KeyError, OSError) as exc:
                    result.update(failures=['read-or-QC-error'], error=str(exc))
                outcomes.append(result)
        if digest(body) != before:
            raise ValueError('recording changed during QC')
        entry = {'slot': case['slot'], 'source_cell_specimen_id': case['source_cell_specimen_id'],
                 'recording_sha256': before, 'sweeps': outcomes, 'specimen_eligibility_established': False}
        with (output.parent/f'slot-{case["slot"]:03d}.yaml').open('x') as handle:
            yaml.safe_dump(entry, handle, sort_keys=False)
        cases.append(entry)
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('source changed during QC')
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'all-fixed-waveform-QC-outcomes-retained-independent-check-pending',
            'source_hashes': {str(p): h for p, h in pins.items()}, 'records': cases,
            'charged_float64_decoded_bytes': budget[0], 'specimen_eligibility_established': False,
            'model_fitting_performed': False, 'junction_correction_applied': False}, handle, sort_keys=False)
    print('Completed sweep-QC outcomes for all 48 fixed recordings; independent assessment pending')


if __name__ == '__main__':
    main()
