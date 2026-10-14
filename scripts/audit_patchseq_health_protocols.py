"""Exact official health code tags and metadata pairing; never sample reads."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from check_patchseq_pairing import attribute, concordant, scalar
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-health-protocol-registration-1360.yaml')
TYPES = {'VoltageClampSeries': ('response', 'VC'), 'VoltageClampStimulusSeries': ('stimulus', 'VC'),
         'CurrentClampSeries': ('response', 'CC'), 'CurrentClampStimulusSeries': ('stimulus', 'CC')}


def code(description):
    if type(description) is not str:
        return None
    description = description.removesuffix('_DA_0')
    return description.split('[', 1)[0]


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def evaluate(records, codes):
    children = defaultdict(dict)
    for item in records:
        parent, _, name = item['path'].rpartition('/')
        children[parent][name] = item
    health, other = [], []
    for node in records:
        if node.get('object_type') != 'group':
            continue
        declared = attribute(node, 'neurodata_type')
        if declared is None:
            continue
        child = children[node['path']]
        description = concordant(node, child, 'stimulus_description', lambda v: type(v) is str and bool(v))
        parsed = code(description)
        roles = [name for name, values in codes.items() if parsed in values]
        if not roles:
            other.append({'path': node['path'], 'declared_type': declared, 'raw_description': description, 'parsed_code': parsed})
            continue
        if len(roles) != 1:
            raise ValueError('ambiguous registered health code')
        protocol = roles[0]
        role, clamp = TYPES.get(declared, (None, None))
        sweep = concordant(node, child, 'sweep_number', lambda v: type(v) is int and 0 <= v < 2**32)
        failures = []
        if role is None or clamp != ('CC' if protocol == 'blowout' else 'VC'):
            failures.append('unsupported-health-series-type')
        root = '/acquisition/' if role == 'response' else '/stimulus/presentation/'
        if not node['path'].startswith(root) or '/' in node['path'][len(root):]:
            failures.append('unsupported-series-root')
        if attribute(node, 'namespace') != 'core':
            failures.append('unsupported-namespace')
        if sweep is None:
            failures.append('missing-invalid-or-conflicting-sweep-number')
        data, clock = child.get('data', {}), child.get('starting_time', {})
        shape = data.get('shape')
        if (data.get('object_type') != 'dataset' or data.get('link_type') != 'HardLink'
            or type(shape) is not list or len(shape) != 1 or type(shape[0]) is not int or shape[0] < 2):
            failures.append('unsupported-data-shape-or-link')
        try:
            kind = np.dtype(data.get('dtype', 'object')).kind
        except (ValueError, TypeError):
            kind = None
        if kind not in ('i', 'u', 'f'):
            failures.append('unsupported-data-dtype')
        expected_unit = 'amperes' if (clamp, role) in (('VC', 'response'), ('CC', 'stimulus')) else 'volts'
        unit, conversion = attribute(data, 'unit'), attribute(data, 'conversion')
        if unit != expected_unit:
            failures.append('unsupported-unit')
        if not finite(conversion) or conversion == 0:
            failures.append('missing-invalid-conversion')
        if data.get('attributes', {}).get('offset', {}).get('present'):
            failures.append('unsupported-offset')
        start, rate = scalar(clock.get('scalar_metadata')), attribute(clock, 'rate')
        if (clock.get('object_type') != 'dataset' or clock.get('link_type') != 'HardLink'
            or not finite(start) or not finite(rate) or rate <= 0):
            failures.append('missing-invalid-clock')
        time_unit = attribute(clock, 'unit')
        if time_unit not in ('seconds', 'Seconds') or 'timestamps' in child:
            failures.append('unsupported-clock-convention')
        health.append({'path': node['path'], 'protocol': protocol, 'role': role, 'clamp': clamp,
            'declared_type': declared, 'sweep_number': sweep, 'raw_description': description,
            'parsed_code': parsed, 'shape': shape, 'dtype': data.get('dtype'), 'unit': unit,
            'conversion': conversion, 'start': start, 'rate': rate, 'time_unit': time_unit, 'failures': failures})
    grouped = defaultdict(list)
    for series in health:
        if series['sweep_number'] is not None:
            grouped[(series['protocol'], series['sweep_number'])].append(series)
    pairs = []
    for (protocol, sweep), items in sorted(grouped.items()):
        responses = [s for s in items if s['role'] == 'response']
        stimuli = [s for s in items if s['role'] == 'stimulus']
        failures = []
        if len(responses) != 1 or len(stimuli) != 1 or len(items) != 2:
            failures.append('nonunique-or-missing-health-partner')
        else:
            response, stimulus = responses[0], stimuli[0]
            if any(s['failures'] for s in items):
                failures.append('series-metadata-failure')
            for name in ('raw_description', 'parsed_code', 'shape', 'start', 'rate', 'clamp'):
                if response[name] != stimulus[name]:
                    failures.append('pair-'+name+'-disagreement')
        pairs.append({'protocol': protocol, 'sweep_number': sweep, 'series_paths': [s['path'] for s in items],
                      'metadata_pair_pass': not failures, 'failures': failures})
    return {'health_series': health, 'health_pairs': pairs, 'nonhealth_series': other,
            'missing_protocols': [name for name in codes if not any(p['protocol'] == name and p['metadata_pair_pass'] for p in pairs)],
            'specimen_eligibility_established': False}


def main():
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: digest(REG), Path(__file__): digest(__file__)}
    for name in ('metadata', 'independent_metadata'):
        pins[Path(reg[name])] = reg[name+'_sha256']
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('registered evidence changed')
    verified = yaml.load(Path(reg['independent_metadata']).read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(p): h for p, h in verified['source_hashes'].items()})
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('source changed')
    metadata = yaml.load(Path(reg['metadata']).read_text(), Loader=yaml.CSafeLoader)
    if len(metadata['records']) != 48:
        raise ValueError('fixed scope changed')
    results = [{'slot': entry['slot'], 'source_cell_specimen_id': entry['source_cell_specimen_id'],
                **evaluate(entry['records'], reg['codes'])} for entry in metadata['records']]
    if len({r['slot'] for r in results}) != 48 or any(digest(p) != h for p, h in pins.items()):
        raise ValueError('source or scope changed')
    output = Path(reg['output'])
    output.parent.mkdir()
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'health-protocol-metadata-outcomes-retained-independent-check-pending',
            'source_hashes': {str(p): h for p, h in pins.items()}, 'records': results,
            'new_HDF5_reads': 0, 'waveform_reads': False, 'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('Retained exact health-protocol metadata coverage for all 48 files')


if __name__ == '__main__':
    main()
