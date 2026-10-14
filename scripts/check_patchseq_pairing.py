"""Metadata-only sweep pairing; this module never opens an NWB file."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-pairing-registration-1323.yaml')


def scalar(item):
    if not item or 'undefined' in item:
        return None
    if 'value' in item:
        return item['value']
    return item.get('decoded_text')


def attribute(node, name):
    return scalar(node.get('attributes', {}).get(name, {}).get('scalar'))


def concordant(node, children, name, valid):
    """All present locations count, including an undecodable present value."""
    values = []
    entry = node.get('attributes', {}).get(name, {})
    if entry.get('present'):
        values.append(scalar(entry.get('scalar')))
    child = children.get(name)
    if child is not None:
        values.append(scalar(child.get('scalar_metadata')))
    if not values or not all(valid(v) for v in values) or any(v != values[0] for v in values):
        return None
    return values[0]


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def series(node, children, role):
    failures = []
    sweep = concordant(node, children, 'sweep_number',
                       lambda v: type(v) is int and 0 <= v <= 4294967295)
    description = concordant(node, children, 'stimulus_description',
                             lambda v: type(v) is str and bool(v))
    if sweep is None:
        failures.append('missing-invalid-or-conflicting-sweep-number')
    if description is None:
        failures.append('missing-invalid-or-conflicting-description')
    root = '/acquisition/' if role == 'response' else '/stimulus/presentation/'
    if not node['path'].startswith(root) or node['path'][len(root):].count('/'):
        failures.append('unsupported-series-location')
    if attribute(node, 'namespace') != 'core':
        failures.append('unsupported-namespace')
    data = children.get('data', {})
    shape = data.get('shape')
    if (data.get('object_type') != 'dataset' or data.get('link_type') != 'HardLink'
        or type(shape) is not list or len(shape) != 1 or type(shape[0]) is not int or shape[0] <= 0):
        failures.append('unsupported-data-shape-or-link')
    try:
        kind = np.dtype(data.get('dtype', 'object')).kind
    except (TypeError, ValueError):
        kind = None
    if kind not in ('i', 'u', 'f'):
        failures.append('unsupported-data-dtype')
    unit = attribute(data, 'unit')
    if unit != ('volts' if role == 'response' else 'amperes'):
        failures.append('unsupported-data-unit')
    conversion = attribute(data, 'conversion')
    if not finite(conversion) or conversion == 0:
        failures.append('missing-invalid-conversion')
    if data.get('attributes', {}).get('offset', {}).get('present'):
        failures.append('explicit-offset-not-supported-by-contract')
    clock = children.get('starting_time', {})
    start, rate = scalar(clock.get('scalar_metadata')), attribute(clock, 'rate')
    clock_unit = attribute(clock, 'unit')
    if not finite(start) or not finite(rate) or rate <= 0:
        failures.append('missing-invalid-clock')
    if clock.get('object_type') != 'dataset' or clock.get('link_type') != 'HardLink':
        failures.append('unsupported-clock-link')
    if clock_unit not in ('seconds', 'Seconds'):
        failures.append('unsupported-time-unit')
    if 'timestamps' in children:
        failures.append('timestamps-require-separate-validation')
    return {'path': node['path'], 'role': role, 'sweep_number': sweep,
            'description': description, 'data_shape': shape, 'data_dtype': data.get('dtype'),
            'unit': unit, 'conversion': conversion, 'starting_time': start, 'rate': rate,
            'time_unit': clock_unit, 'exporter_time_spelling_deviation': clock_unit == 'Seconds',
            'failures': failures}


def evaluate(records):
    by_path = {r['path']: r for r in records}
    if len(by_path) != len(records):
        raise ValueError('duplicate descriptor path')
    targets, nontargets = [], []
    children_by_parent = defaultdict(dict)
    for r in records:
        parent, _, name = r['path'].rpartition('/')
        children_by_parent[parent][name] = r
    for r in records:
        if r.get('object_type') != 'group':
            continue
        declared = attribute(r, 'neurodata_type')
        role = {'CurrentClampSeries': 'response', 'CurrentClampStimulusSeries': 'stimulus'}.get(declared)
        if role is None:
            if declared is not None:
                nontargets.append({'path': r['path'], 'declared_type': declared})
            continue
        targets.append(series(r, children_by_parent[r['path']], role))
    buckets = defaultdict(list)
    unkeyed = []
    for item in targets:
        if item['sweep_number'] is None:
            unkeyed.append(item['path'])
        else:
            buckets[item['sweep_number']].append(item)
    pairs = []
    for sweep, items in sorted(buckets.items()):
        responses = [x for x in items if x['role'] == 'response']
        stimuli = [x for x in items if x['role'] == 'stimulus']
        failures = []
        if len(responses) != 1 or len(stimuli) != 1:
            failures.append('nonunique-or-missing-sweep-partner')
        else:
            response, stimulus = responses[0], stimuli[0]
            if any(x['failures'] for x in items):
                failures.append('series-metadata-failure')
            for field in ('description', 'starting_time', 'rate', 'data_shape'):
                if response[field] != stimulus[field]:
                    failures.append('pair-'+field+'-disagreement')
        pairs.append({'sweep_number': sweep, 'series_paths': [x['path'] for x in items],
                      'metadata_pair_pass': not failures, 'failures': failures})
    return {'series': targets, 'sweeps': pairs, 'unkeyed_series_paths': unkeyed,
            'nontarget_series': nontargets,
            'metadata_pair_passes': sum(x['metadata_pair_pass'] for x in pairs),
            'metadata_pair_nonpasses': sum(not x['metadata_pair_pass'] for x in pairs),
            'physiology_eligibility_established': False}


def main():
    registration = yaml.safe_load(REG.read_text())
    pins = {REG: digest(REG), Path(__file__): digest(__file__)}
    for key in ('input', 'independent_input'):
        pins[Path(registration[key])] = registration[key+'_sha256']
    for path, expected in pins.items():
        if digest(path) != expected:
            raise ValueError('registered input changed')
    report = yaml.load(Path(registration['input']).read_text(), Loader=yaml.CSafeLoader)
    independent = yaml.load(Path(registration['independent_input']).read_text(), Loader=yaml.CSafeLoader)
    if independent['status'] != 'all-selected-trace-metadata-and-access-whitelists-independently-reconstructed':
        raise ValueError('independent metadata gate not established')
    pins.update({Path(p): h for p, h in independent['source_hashes'].items()})
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('original source changed')
    if len(report['records']) != 48 or len(independent['records']) != 48:
        raise ValueError('fixed 48-case scope changed')
    results = []
    for item, verified in zip(report['records'], independent['records'], strict=True):
        if item['slot'] != verified['slot'] or item['status'] != 'schema-metadata-inventoried':
            raise ValueError('scope or completion mismatch')
        results.append({'slot': item['slot'], 'source_cell_specimen_id': item['source_cell_specimen_id'],
                        **evaluate(item['records'])})
    if len({x['slot'] for x in results}) != 48 or any(digest(p) != h for p, h in pins.items()):
        raise ValueError('scope or evidence changed during evaluation')
    totals = Counter()
    for item in results:
        totals.update({k: item[k] for k in ('metadata_pair_passes', 'metadata_pair_nonpasses')})
        totals['unkeyed_series'] += len(item['unkeyed_series_paths'])
    output = Path(registration['output'])
    output.parent.mkdir()
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'metadata-pairing-outcomes-retained-independent-check-pending',
                        'source_hashes': {str(p): h for p, h in pins.items()}, 'summary': dict(totals),
                        'records': results, 'new_HDF5_reads': 0, 'waveform_reads': False,
                        'specimen_eligibility_established': False}, handle, sort_keys=False)
    print(dict(totals))


if __name__ == '__main__':
    main()
