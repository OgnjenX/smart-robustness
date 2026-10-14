"""Independent exact health-code inventory for the observed valid VC pairs."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from verify_patchseq_whole_file_integrity import digest

RESULT = Path('results/patchseq-health-protocols-1361/manifest.yaml')
OUTPUT = Path('results/patchseq-health-protocols-independent-1362/manifest.yaml')


def require(condition):
    if not condition:
        raise ValueError('independent health code, pairing or metadata differs')


def value(raw):
    if not raw or 'undefined' in raw:
        return None
    return raw['value'] if 'value' in raw else raw.get('decoded_text')


def attr(node, name):
    return value(node.get('attributes', {}).get(name, {}).get('scalar'))


def all_locations(group, children, name):
    found = []
    entry = group.get('attributes', {}).get(name, {})
    if entry.get('present'):
        found.append(value(entry.get('scalar')))
    if name in children:
        found.append(value(children[name].get('scalar_metadata')))
    if not found or any(v is None or v != found[0] for v in found):
        return None
    return found[0]


def reconstruct(records, codes):
    children = defaultdict(dict)
    for node in records:
        parent, _, name = node['path'].rpartition('/')
        children[parent][name] = node
    targets, other = [], []
    for group in records:
        if group.get('object_type') != 'group' or attr(group, 'neurodata_type') is None:
            continue
        declared = attr(group, 'neurodata_type')
        child = children[group['path']]
        description = all_locations(group, child, 'stimulus_description')
        if type(description) is not str or not description:
            description = None
        parsed = None
        if description is not None:
            normalized = description[:-5] if description.endswith('_DA_0') else description
            parsed = normalized.partition('[')[0]
        roles = [name for name, members in codes.items() if parsed in members]
        if not roles:
            other.append({'path': group['path'], 'declared_type': declared, 'raw_description': description, 'parsed_code': parsed})
            continue
        require(len(roles) == 1)
        protocol = roles[0]
        # This checker intentionally covers the observed all-valid VC result.
        require(protocol != 'blowout' and declared in ('VoltageClampSeries', 'VoltageClampStimulusSeries'))
        role = 'response' if declared == 'VoltageClampSeries' else 'stimulus'
        root = '/acquisition/' if role == 'response' else '/stimulus/presentation/'
        require(group['path'].startswith(root) and '/' not in group['path'][len(root):] and attr(group, 'namespace') == 'core')
        sweep = all_locations(group, child, 'sweep_number')
        require(type(sweep) is int and 0 <= sweep < 2**32)
        data, clock = child['data'], child['starting_time']
        require(data['object_type'] == clock['object_type'] == 'dataset' and data['link_type'] == clock['link_type'] == 'HardLink')
        require(len(data['shape']) == 1 and type(data['shape'][0]) is int and data['shape'][0] >= 2
                and np.dtype(data['dtype']).kind in 'iuf')
        unit, scale, start, rate = attr(data, 'unit'), attr(data, 'conversion'), value(clock.get('scalar_metadata')), attr(clock, 'rate')
        require(unit == ('amperes' if role == 'response' else 'volts'))
        require(all(type(v) in (float, int) and math.isfinite(v) for v in (scale, start, rate)) and scale != 0 and rate > 0)
        require(not data['attributes'].get('offset', {}).get('present') and 'timestamps' not in child)
        require(attr(clock, 'unit') in ('seconds', 'Seconds'))
        targets.append({'path': group['path'], 'protocol': protocol, 'role': role, 'clamp': 'VC',
            'declared_type': declared, 'sweep_number': sweep, 'raw_description': description,
            'parsed_code': parsed, 'shape': data['shape'], 'dtype': data['dtype'], 'unit': unit,
            'conversion': scale, 'start': start, 'rate': rate, 'time_unit': attr(clock, 'unit'), 'failures': []})
    grouped = defaultdict(list)
    for item in targets:
        grouped[(item['protocol'], item['sweep_number'])].append(item)
    pairs = []
    for (protocol, sweep), members in sorted(grouped.items()):
        require(len(members) == 2 and {s['role'] for s in members} == {'response', 'stimulus'})
        require(all(members[0][k] == members[1][k] for k in ('raw_description', 'parsed_code', 'shape', 'start', 'rate', 'clamp')))
        pairs.append({'protocol': protocol, 'sweep_number': sweep, 'series_paths': [s['path'] for s in members],
                      'metadata_pair_pass': True, 'failures': []})
    return {'health_series': targets, 'health_pairs': pairs, 'nonhealth_series': other,
            'missing_protocols': [name for name in codes if not any(p['protocol'] == name for p in pairs)],
            'specimen_eligibility_established': False}


def main():
    pins = {RESULT: 'f7989b5875b50e3d616a32ff24116f1823c6a14f1df574c5265ab04b42986e21'}
    require(digest(RESULT) == pins[RESULT])
    report = yaml.load(RESULT.read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(p): h for p, h in report['source_hashes'].items()})
    require(all(digest(p) == h for p, h in pins.items()))
    reg = yaml.safe_load(Path('docs/validation-results/post2008-patchseq-health-protocol-registration-1360.yaml').read_text())
    source = yaml.load(Path(reg['metadata']).read_text(), Loader=yaml.CSafeLoader)
    require(len(source['records']) == len(report['records']) == 48)
    summaries = []
    for raw, saved in zip(source['records'], report['records'], strict=True):
        expected = {'slot': raw['slot'], 'source_cell_specimen_id': raw['source_cell_specimen_id'], **reconstruct(raw['records'], reg['codes'])}
        require(saved == expected)
        summaries.append({'slot': raw['slot'], 'verified_health_pairs': len(saved['health_pairs']),
                          'missing_protocols': saved['missing_protocols']})
    require(len({r['slot'] for r in summaries}) == 48 and all(digest(p) == h for p, h in pins.items()))
    require(report['new_HDF5_reads'] == 0 and report['waveform_reads'] is False and report['specimen_eligibility_established'] is False)
    OUTPUT.parent.mkdir()
    with OUTPUT.open('x') as handle:
        yaml.safe_dump({'status': 'all-health-code-pairs-nontarget-retention-and-missingness-independently-reconstructed',
            'verifier_sha256': digest(__file__), 'source_hashes': {str(p): h for p, h in pins.items()},
            'records': summaries, 'waveform_reads': False, 'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('Independently reconstructed all health protocol metadata and bounded-code missingness')


if __name__ == '__main__':
    main()
