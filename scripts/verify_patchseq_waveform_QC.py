"""Independent fsum-based reconstruction of every sealed waveform QC outcome."""

from __future__ import annotations

import math
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-waveform-QC-verification-registration-1329.yaml')


def compare(expected, actual):
    if type(expected) is not type(actual):
        raise ValueError('type differs')
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise ValueError('keys differ')
        for name in expected:
            compare(expected[name], actual[name])
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise ValueError('length differs')
        for a, b in zip(expected, actual, strict=True):
            compare(a, b)
    elif type(expected) is float:
        if not math.isfinite(expected) or not math.isfinite(actual) or abs(expected-actual) > 1e-10+1e-12*abs(actual):
            raise ValueError('numeric value differs')
    elif expected != actual:
        raise ValueError('discrete value differs')


def reconstruct(v, i, rate):
    out = {'long_square_candidate': False, 'sweep_QC_pass': False, 'failures': []}
    if (v.ndim != 1 or i.ndim != 1 or len(v) != len(i) or len(v) < 2
        or not math.isfinite(rate) or rate <= 0):
        out['failures'] = ['invalid-array-or-rate']
        return out
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(i))):
        out['failures'] = ['nonfinite-samples']
        return out
    edges = np.nonzero(i[1:] != i[:-1])[0]+1
    out['command_transition_count'] = len(edges)
    if len(edges) not in (2, 4):
        out['failures'] = ['not-supported-single-step-waveform']
        return out
    if len(edges) == 4 and (edges[0]/rate >= 0.4 or (edges[1]-edges[0])/rate >= 0.4 or i[edges[1]] != i[0]):
        out['failures'] = ['unsupported-test-pulse']
        return out
    first, last = map(int, edges[-2:])
    if (last-first)/rate < 0.5 or i[last] != i[0] or i[first] == i[0]:
        out['failures'] = ['not-long-square-returning-to-baseline']
        return out
    out.update(long_square_candidate=True, step_start_index=first, step_stop_index=last,
        step_duration_s=(last-first)/rate, command_baseline_pA=float(i[0]),
        command_step_pA=float(i[first]), command_delta_pA=float(i[first]-i[0]))
    width, short = int(rate/2), int(rate*0.0015)
    pre, post = [first-1-width, first-1], [len(v)-1-width, len(v)-1]
    if short < 1 or width < 1 or pre[0] < 0 or post[0] < last:
        out['failures'] = ['incomplete-baseline-window']
        return out
    if any(not np.all(i[a:b] == i[0]) for a, b in (pre, post)):
        out['failures'] = ['baseline-window-contains-command-transition']
        return out
    def mean(values):
        return math.fsum(map(float, values))/len(values)
    def rms(values):
        center = mean(values)
        return math.sqrt(math.fsum((float(x)-center)**2 for x in values)/len(values))
    pv, qv = v[slice(*pre)], v[slice(*post)]
    metrics = {'pre_noise_RMS_mV': rms(pv[:short]), 'post_noise_RMS_mV': rms(qv[-short:]),
        'slow_noise_RMS_mV': rms(pv), 'absolute_mean_drift_mV': abs(mean(pv)-mean(qv))}
    out.update(metrics=metrics, pre_window=pre, post_window=post)
    for key, bound in [('pre_noise_RMS_mV', 0.07), ('post_noise_RMS_mV', 0.07),
                       ('slow_noise_RMS_mV', 0.5), ('absolute_mean_drift_mV', 1.)]:
        if not math.isfinite(metrics[key]) or metrics[key] > bound:
            out['failures'].append(key+'-exceeds-limit')
    out['sweep_QC_pass'] = len(out['failures']) == 0
    return out


def node_at(file, path, h5py):
    node = file
    for component in path.strip('/').split('/'):
        if not isinstance(node, h5py.Group) or type(node.get(component, getlink=True)) is not h5py.HardLink:
            raise ValueError('non-hard-or-missing-path')
        node = node[component]
    return node


def storage(node, h5py):
    if not isinstance(node, h5py.Dataset) or node.is_virtual or node.external:
        raise ValueError('external-virtual-or-nondataset')
    properties = node.id.get_create_plist()
    for n in range(properties.get_nfilters()):
        if properties.get_filter(n)[0] not in (1, 2, 3):
            raise ValueError('unsupported-HDF5-filter')


def scalar_metadata(file, path, h5py):
    group = node_at(file, path, h5py)
    result = {}
    for name in ('gain', 'bias_current', 'bridge_balance', 'capacitance_compensation'):
        if name not in group:
            result[name] = {'present': False, 'undefined': 'absent'}
            continue
        try:
            n = node_at(file, path+'/'+name, h5py)
            storage(n, h5py)
            if (n.shape not in ((), (1,)) or n.dtype.kind not in 'iuf' or n.id.get_storage_size() > 4096
                or n.id.get_create_plist().get_nfilters()):
                raise ValueError('unsupported-instrument-scalar')
            value = n[()].item()
            result[name] = {'present': True, 'dtype': str(n.dtype)}
            result[name].update({'value': value} if math.isfinite(value) else {'undefined': 'nonfinite'})
        except (ValueError, TypeError, OSError) as exc:
            result[name] = {'present': True, 'undefined': str(exc)}
    return result


def main():
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: digest(REG), Path(reg['result']): reg['result_sha256'], Path(reg['collector']): reg['collector_sha256']}
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('sealed result changed')
    result_path = Path(reg['result'])
    report = yaml.load(result_path.read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(p): h for p, h in report['source_hashes'].items()})
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('source changed')
    protocol = yaml.safe_load(Path(reg['original_protocol']).read_text())
    if version('h5py') != protocol['read_policy']['parser'].split('==')[1]:
        raise ValueError('parser differs')
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    import h5py
    acquired = yaml.safe_load(Path(protocol['acquisition']).read_text())
    receipts = {x['slot']: yaml.safe_load(Path(x['path']).read_text()) for x in acquired['receipts']}
    pairs = yaml.load(Path(protocol['pairing']).read_text(), Loader=yaml.CSafeLoader)['records']
    if len(pairs) != len(report['records']) or len(pairs) != 48:
        raise ValueError('case scope differs')
    summaries, charged = [], 0
    for source, saved in zip(pairs, report['records'], strict=True):
        if source['slot'] != saved['slot'] or source['source_cell_specimen_id'] != saved['source_cell_specimen_id']:
            raise ValueError('case identity differs')
        body = Path(receipts[source['slot']]['complete_body_path'])
        if digest(body) != saved['recording_sha256'] or saved['recording_sha256'] != receipts[source['slot']]['complete_file_SHA256']:
            raise ValueError('recording changed')
        local = yaml.safe_load((result_path.parent/f'slot-{source["slot"]:03d}.yaml').read_text())
        compare(local, saved)
        index = {s['path']: s for s in source['series']}
        if len(source['sweeps']) != len(saved['sweeps']):
            raise ValueError('sweep omitted')
        with h5py.File(body, 'r') as file:
            if file.attrs.get('nwb_version') != '2.2.5':
                raise ValueError('version differs')
            for membership, expected in zip(source['sweeps'], saved['sweeps'], strict=True):
                by_role = {index[p]['role']: index[p] for p in membership['series_paths']}
                arrays = {}
                for role in ('response', 'stimulus'):
                    item = by_role[role]
                    n = node_at(file, item['path']+'/data', h5py)
                    storage(n, h5py)
                    if n.ndim != 1 or list(n.shape) != item['data_shape'] or n.dtype.kind not in 'iuf' or n.size > 10000000:
                        raise ValueError('storage eligibility differs')
                    charged += n.size*8
                    if charged > protocol['read_policy']['maximum_total_float64_decoded_bytes']:
                        raise ValueError('array budget exceeded')
                    arrays[role] = np.array(n[()], dtype='float64')
                    arrays[role] *= item['conversion']*(1000 if role == 'response' else 1e12)
                actual = {'sweep_number': membership['sweep_number'],
                          **reconstruct(arrays['response'], arrays['stimulus'], by_role['response']['rate']),
                          'description': by_role['response']['description'],
                          'instrument_metadata': scalar_metadata(file, by_role['response']['path'], h5py)}
                compare(expected, actual)
        if digest(body) != saved['recording_sha256']:
            raise ValueError('recording changed during verification')
        summaries.append({'slot': source['slot'], 'verified_sweeps': len(saved['sweeps']),
            'verified_candidates': sum(s['long_square_candidate'] for s in saved['sweeps']),
            'verified_sweep_passes': sum(s['sweep_QC_pass'] for s in saved['sweeps'])})
    if len({s['slot'] for s in summaries}) != 48 or charged != report['charged_float64_decoded_bytes']:
        raise ValueError('scope or charged bytes differs')
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('evidence changed during verification')
    for flag in ('specimen_eligibility_established', 'model_fitting_performed', 'junction_correction_applied'):
        if report[flag] is not False:
            raise ValueError('claim scope differs')
    output = Path(reg['output'])
    output.parent.mkdir()
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'all-array-QC-outcomes-independently-reconstructed',
            'verifier_sha256': digest(__file__), 'source_hashes': {str(p): h for p, h in pins.items()},
            'records': summaries, 'charged_float64_decoded_bytes': charged,
            'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('Independently reconstructed all 2330 sweep QC outcomes and instrument scalar metadata')


if __name__ == '__main__':
    main()
