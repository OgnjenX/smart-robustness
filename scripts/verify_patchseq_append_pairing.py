"""Independently validate this all-pass metadata result, not waveform quality.

Fails closed on any nonpass; a future mixed result needs separate assessment.
No pairing collector import, NWB access, or waveform samples.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml
from verify_patchseq_whole_file_integrity import digest

SOURCE = Path('results/patchseq-append-trace-schema-1381/manifest.yaml')
RESULT = Path('results/patchseq-append-pairing-1385/manifest.yaml')
OUTPUT = Path('results/patchseq-append-pairing-independent-1386/manifest.yaml')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def raw(item):
    require(isinstance(item, dict) and 'undefined' not in item, 'undefined scalar')
    return item['value'] if 'value' in item else item.get('decoded_text')


def av(node, name):
    return raw(node['attributes'][name]['scalar'])


def verify_case(descriptors, outcome):
    nodes = {x['path']: x for x in descriptors}
    require(len(nodes) == len(descriptors), 'duplicate path')
    actual = {}
    for path, node in nodes.items():
        if node.get('object_type') != 'group':
            continue
        attr = node.get('attributes', {}).get('neurodata_type', {})
        if not attr.get('present'):
            continue
        kind = av(node, 'neurodata_type')
        if kind not in ('CurrentClampSeries', 'CurrentClampStimulusSeries'):
            continue
        response = kind == 'CurrentClampSeries'
        root = '/acquisition/' if response else '/stimulus/presentation/'
        require(path.startswith(root) and '/' not in path[len(root):], 'wrong root')
        require(av(node, 'namespace') == 'core', 'wrong namespace')
        fields = {}
        for name in ('sweep_number', 'stimulus_description'):
            values = []
            if node['attributes'].get(name, {}).get('present'):
                values.append(av(node, name))
            if path+'/'+name in nodes:
                values.append(raw(nodes[path+'/'+name].get('scalar_metadata')))
            require(bool(values) and all(v == values[0] for v in values), 'conflicting identity')
            fields[name] = values[0]
        sweep, label = fields['sweep_number'], fields['stimulus_description']
        require(type(sweep) is int and 0 <= sweep < 2**32, 'invalid sweep')
        require(type(label) is str and bool(label), 'invalid description')
        data, clock = nodes[path+'/data'], nodes[path+'/starting_time']
        require(all(n.get('object_type') == 'dataset' and n.get('link_type') == 'HardLink'
                    for n in (data, clock)), 'unsafe metadata link')
        shape = data.get('shape')
        require(type(shape) is list and len(shape) == 1 and type(shape[0]) is int and shape[0] > 0,
                'invalid shape')
        require(np.dtype(data['dtype']).kind in 'iuf', 'invalid dtype')
        require(av(data, 'unit') == ('volts' if response else 'amperes'), 'invalid unit')
        scale, start, rate = av(data, 'conversion'), raw(clock.get('scalar_metadata')), av(clock, 'rate')
        require(all(type(v) in (float, int) and math.isfinite(v) for v in (scale, start, rate))
                and scale != 0 and rate > 0, 'invalid scaling or clock')
        require(not data['attributes'].get('offset', {}).get('present'), 'offset unsupported')
        require(path+'/timestamps' not in nodes, 'timestamps unsupported')
        unit = av(clock, 'unit')
        require(unit in ('seconds', 'Seconds'), 'invalid time unit')
        actual[path] = {'path': path, 'role': 'response' if response else 'stimulus',
            'sweep_number': sweep, 'description': label, 'data_shape': shape, 'data_dtype': data['dtype'],
            'unit': av(data, 'unit'), 'conversion': scale, 'starting_time': start, 'rate': rate,
            'time_unit': unit, 'exporter_time_spelling_deviation': unit == 'Seconds', 'failures': []}
    require(len(outcome['series']) == len(actual), 'omitted or extra target series')
    require({x['path']: x for x in outcome['series']} == actual, 'series values disagree')
    require(not outcome['unkeyed_series_paths'], 'unkeyed result not supported')
    grouped = {}
    for path, item in actual.items():
        roles = grouped.setdefault(item['sweep_number'], {})
        require(item['role'] not in roles, 'duplicate partner')
        roles[item['role']] = path
    require(len(outcome['sweeps']) == len(grouped), 'omitted or extra sweep')
    require(len({x['sweep_number'] for x in outcome['sweeps']}) == len(grouped), 'duplicate output sweep')
    for pair in outcome['sweeps']:
        roles = grouped[pair['sweep_number']]
        require(set(roles) == {'stimulus', 'response'}, 'missing partner')
        require(len(pair['series_paths']) == 2 and set(pair['series_paths']) == set(roles.values()), 'wrong membership')
        r, s = (actual[roles[k]] for k in ('response', 'stimulus'))
        require(all(r[k] == s[k] for k in ('description', 'data_shape', 'starting_time', 'rate')), 'alignment differs')
        require(pair['metadata_pair_pass'] is True and pair['failures'] == [], 'nonpass needs distinct assessment')
    require(outcome['metadata_pair_passes'] == len(grouped) and outcome['metadata_pair_nonpasses'] == 0,
            'counts disagree')
    require(outcome['physiology_eligibility_established'] is False, 'overstated physiology')
    expected_other = [{'path': x['path'], 'declared_type': av(x, 'neurodata_type')}
                      for x in descriptors if x.get('object_type') == 'group'
                      and x.get('attributes', {}).get('neurodata_type', {}).get('present')
                      and av(x, 'neurodata_type') not in ('CurrentClampSeries', 'CurrentClampStimulusSeries')]
    require(outcome['nontarget_series'] == expected_other, 'nontarget retention differs')
    return len(grouped)


def main():
    pins = {SOURCE: '7de14ca2a3ef2f42b20b82eaf7d5f3d3dbc6b6e8a2b813b41231fdde76f90aac',
            RESULT: digest(RESULT)}
    require(all(digest(p) == h for p, h in pins.items()), 'sealed result changed')
    source = yaml.load(SOURCE.read_text(), Loader=yaml.CSafeLoader)
    result = yaml.load(RESULT.read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(p): h for p, h in result['source_hashes'].items()})
    require(all(digest(p) == h for p, h in pins.items()), 'sealed source changed')
    require(len(source['records']) == len(result['records']) == 48, 'scope changed')
    counts = []
    for src, out in zip(source['records'], result['records'], strict=True):
        require(src['slot'] == out['slot'] and src['source_cell_specimen_id'] == out['source_cell_specimen_id'],
                'case identity differs')
        counts.append({'slot': src['slot'], 'independently_verified_pairs': verify_case(src['records'], out)})
    require(len({x['slot'] for x in counts}) == 48, 'duplicate case')
    require(result['summary'] == {'metadata_pair_passes': sum(x['independently_verified_pairs'] for x in counts),
                                 'metadata_pair_nonpasses': 0, 'unkeyed_series': 0}, 'summary differs')
    require(result['new_HDF5_reads'] == 0 and result['waveform_reads'] is False
            and result['specimen_eligibility_established'] is False, 'scope differs')
    require(all(digest(p) == h for p, h in pins.items()), 'evidence changed during verification')
    OUTPUT.parent.mkdir()
    with OUTPUT.open('x') as handle:
        yaml.safe_dump({'status': 'all-metadata-pair-memberships-values-and-nontarget-retention-independently-verified',
                        'verifier_sha256': digest(__file__), 'source_hashes': {str(p): h for p, h in pins.items()},
                        'records': counts, 'new_HDF5_reads': 0, 'waveform_reads': False,
                        'specimen_eligibility_established': False}, handle, sort_keys=False)
    print(f'Independently verified {sum(x["independently_verified_pairs"] for x in counts)} metadata pairs across 48 recordings')


if __name__ == '__main__':
    main()
