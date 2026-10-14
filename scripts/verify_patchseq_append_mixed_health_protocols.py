"""Independent mixed health metadata reconstruction without collector imports."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from verify_patchseq_health_protocols import all_locations, attr, require, value
from verify_patchseq_whole_file_integrity import digest, same

REG = Path('docs/validation-results/post2008-patchseq-append-mixed-health-verification-registration-1396.yaml')


def reconstruct(records, codes):
    descendants = defaultdict(dict)
    for node in records:
        parent, _, name = node['path'].rpartition('/')
        descendants[parent][name] = node
    targets, others = [], []
    kinds = {'CurrentClampSeries': ('response', 'CC'), 'CurrentClampStimulusSeries': ('stimulus', 'CC'),
             'VoltageClampSeries': ('response', 'VC'), 'VoltageClampStimulusSeries': ('stimulus', 'VC')}
    for node in records:
        if node.get('object_type') != 'group' or attr(node, 'neurodata_type') is None:
            continue
        declared = attr(node, 'neurodata_type')
        children = descendants[node['path']]
        text = all_locations(node, children, 'stimulus_description')
        if type(text) is not str or not text:
            text = None
        parsed = text.removesuffix('_DA_0').partition('[')[0] if text is not None else None
        names = [name for name, aliases in codes.items() if parsed in aliases]
        if not names:
            others.append({'path': node['path'], 'declared_type': declared,
                           'raw_description': text, 'parsed_code': parsed})
            continue
        require(len(names) == 1)
        protocol = names[0]
        role, clamp = kinds.get(declared, (None, None))
        failures = []
        if role is None or clamp != ('CC' if protocol == 'blowout' else 'VC'):
            failures.append('unsupported-health-series-type')
        root = '/acquisition/' if role == 'response' else '/stimulus/presentation/'
        if not node['path'].startswith(root) or '/' in node['path'][len(root):]:
            failures.append('unsupported-series-root')
        if attr(node, 'namespace') != 'core':
            failures.append('unsupported-namespace')
        sweep = all_locations(node, children, 'sweep_number')
        if type(sweep) is not int or not 0 <= sweep < 2**32:
            sweep = None
            failures.append('missing-invalid-or-conflicting-sweep-number')
        data, clock = children.get('data', {}), children.get('starting_time', {})
        shape = data.get('shape')
        if (data.get('object_type') != 'dataset' or data.get('link_type') != 'HardLink'
            or type(shape) is not list or len(shape) != 1 or type(shape[0]) is not int or shape[0] < 2):
            failures.append('unsupported-data-shape-or-link')
        try:
            dtype_kind = np.dtype(data.get('dtype', 'object')).kind
        except (ValueError, TypeError):
            dtype_kind = None
        if dtype_kind not in ('i', 'u', 'f'):
            failures.append('unsupported-data-dtype')
        unit, scale = attr(data, 'unit'), attr(data, 'conversion')
        current = (clamp == 'VC' and role == 'response') or (clamp == 'CC' and role == 'stimulus')
        if unit != ('amperes' if current else 'volts'):
            failures.append('unsupported-unit')
        valid = lambda x: type(x) in (int, float) and math.isfinite(x)
        if not valid(scale) or scale == 0:
            failures.append('missing-invalid-conversion')
        if data.get('attributes', {}).get('offset', {}).get('present'):
            failures.append('unsupported-offset')
        start, rate = value(clock.get('scalar_metadata')), attr(clock, 'rate')
        if (clock.get('object_type') != 'dataset' or clock.get('link_type') != 'HardLink'
            or not valid(start) or not valid(rate) or rate <= 0):
            failures.append('missing-invalid-clock')
        time_unit = attr(clock, 'unit')
        if time_unit not in ('seconds', 'Seconds') or 'timestamps' in children:
            failures.append('unsupported-clock-convention')
        targets.append({'path': node['path'], 'protocol': protocol, 'role': role, 'clamp': clamp,
                        'declared_type': declared, 'sweep_number': sweep, 'raw_description': text,
                        'parsed_code': parsed, 'shape': shape, 'dtype': data.get('dtype'), 'unit': unit,
                        'conversion': scale, 'start': start, 'rate': rate, 'time_unit': time_unit,
                        'failures': failures})
    buckets = defaultdict(list)
    for entry in targets:
        if entry['sweep_number'] is not None:
            buckets[(entry['protocol'], entry['sweep_number'])].append(entry)
    pairs = []
    for (protocol, sweep), members in sorted(buckets.items()):
        failures = []
        responses = [s for s in members if s['role'] == 'response']
        stimuli = [s for s in members if s['role'] == 'stimulus']
        if len(members) != 2 or len(responses) != 1 or len(stimuli) != 1:
            failures.append('nonunique-or-missing-health-partner')
        else:
            if any(s['failures'] for s in members):
                failures.append('series-metadata-failure')
            for key in ('raw_description', 'parsed_code', 'shape', 'start', 'rate', 'clamp'):
                if responses[0][key] != stimuli[0][key]:
                    failures.append('pair-'+key+'-disagreement')
        pairs.append({'protocol': protocol, 'sweep_number': sweep, 'series_paths': [s['path'] for s in members],
                      'metadata_pair_pass': not failures, 'failures': failures})
    return {'health_series': targets, 'health_pairs': pairs, 'nonhealth_series': others,
            'missing_protocols': [name for name in codes if not any(
                p['protocol'] == name and p['metadata_pair_pass'] for p in pairs)],
            'specimen_eligibility_established': False}


def main():
    reg = yaml.safe_load(REG.read_text())
    report = yaml.load(Path(reg['result']).read_text(), Loader=yaml.CSafeLoader)
    pins = {REG: digest(REG), Path(reg['result']): reg['result_sha256'], Path(__file__): digest(__file__),
            Path('scripts/verify_patchseq_health_protocols.py'): digest('scripts/verify_patchseq_health_protocols.py')}
    pins.update({Path(p): h for p, h in report['source_hashes'].items()})
    require(all(digest(p) == h for p, h in pins.items()))
    protocol = yaml.safe_load(Path(reg['protocol']).read_text())
    source = yaml.load(Path(protocol['metadata']).read_text(), Loader=yaml.CSafeLoader)
    require(len(source['records']) == len(report['records']) == 48)
    summaries = []
    for original, saved in zip(source['records'], report['records'], strict=True):
        expected = {'slot': original['slot'], 'source_cell_specimen_id': original['source_cell_specimen_id'],
                    **reconstruct(original['records'], protocol['codes'])}
        same(expected, saved)
        summaries.append({'slot': saved['slot'], 'verified_health_pairs': len(saved['health_pairs']),
                          'metadata_pair_nonpasses': sum(not p['metadata_pair_pass'] for p in saved['health_pairs']),
                          'missing_protocols': saved['missing_protocols']})
    require(len({r['slot'] for r in summaries}) == 48)
    require(report['new_HDF5_reads'] == 0 and report['waveform_reads'] is False
            and report['specimen_eligibility_established'] is False)
    require(all(digest(p) == h for p, h in pins.items()))
    output = Path(reg['output'])
    output.parent.mkdir()
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'all-health-code-pairs-nontarget-retention-and-missingness-independently-reconstructed',
                       'source_hashes': {str(p): h for p, h in pins.items()}, 'records': summaries,
                       'collector_imported': False, 'new_HDF5_reads': 0, 'waveform_reads': False,
                       'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('All mixed health metadata passes and nonpasses independently reconstructed')


if __name__ == '__main__':
    main()
