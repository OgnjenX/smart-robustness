"""Metadata-only full-support ledger. No neuron replay or recording-array reader."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path

import yaml
from patchseq_fit_search import candidate_dict, candidates

PLAN = Path('docs/validation-results/post2008-patchseq-numerical-admissibility-plan-1497.yaml')
PROTOCOL = Path('docs/validation-results/post2008-patchseq-isolated-fit-registration-1436.yaml')
SOURCE = Path('docs/validation-results/post2008-patchseq-fitting-target-registration-1429.yaml')
ORIGINAL = Path('results/patchseq-isolated-fit-1438/checkpoint.json')
BENCHMARK = Path('results/patchseq-precision-synthetic-benchmark-1494/checkpoint.json')
OUTPUT = Path('results/patchseq-numerical-workload-1499/manifest.json')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_metadata(path, pins):
    path = Path(path)
    if path.suffix not in ('.yaml', '.json'):
        raise ValueError('metadata-only reader forbids recording bodies')
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    name = str(path)
    if name in pins and pins[name] != sha:
        raise ValueError('metadata hash differs: '+name)
    pins[name] = sha
    return yaml.safe_load(raw) if path.suffix == '.yaml' else json.loads(raw)


def assemble(protocol, catalog, pairings, table, original, benchmark):
    strata = protocol['strata']
    n = protocol['search']['candidates_per_stratum']
    if len(set(strata)) != len(strata) or len(table) != n:
        raise ValueError('stratum or parameter table scope differs')
    summaries = {s: {'groups': set(), 'recordings': set(), 'sweeps': 0, 'intervals': 0} for s in strata}
    sweeps, seen, recordings = [], set(), set()
    for record in catalog['records']:
        if record['partition'] != 'fitting':
            continue  # Partition gate before resolving any pairing metadata.
        st = record['source_group']+'-'+record['layer_label']
        if st not in summaries:
            raise ValueError('unexpected fitting stratum')
        wanted = [s for s in record['sweeps'] if s['state'] == 'fitting-response-characterized']
        if not wanted:
            continue
        recording = (record['source_batch'], record['slot'])
        if recording in recordings:
            raise ValueError('duplicate characterized recording')
        recordings.add(recording)
        found = [p for p in pairings[record['source_batch']]['records'] if p['slot'] == record['slot']]
        if len(found) != 1:
            raise ValueError('pairing identity missing or duplicated')
        paired = found[0]
        series = {p['path']: p for p in paired['series']}
        if len(series) != len(paired['series']):
            raise ValueError('duplicate series descriptor')
        pairs = {p['sweep_number']: p for p in paired['sweeps']}
        if len(pairs) != len(paired['sweeps']):
            raise ValueError('duplicate paired sweep')
        for sweep in wanted:
            key = [record['source_batch'], record['slot'], sweep['sweep_number']]
            if tuple(key) in seen:
                raise ValueError('duplicate characterized sweep')
            seen.add(tuple(key))
            pair = pairs[sweep['sweep_number']]
            descriptors = [series[p] for p in pair['series_paths']]
            if len(descriptors) != 2 or {p['role'] for p in descriptors} != {'stimulus', 'response'}:
                raise ValueError('paired metadata roles differ')
            roles = {p['role']: p for p in descriptors}
            shape, rate = roles['stimulus']['data_shape'], roles['response']['rate']
            if (len(shape) != 1 or type(shape[0]) is not int or shape[0] < 2
                or isinstance(rate, bool) or not math.isfinite(rate) or not 20000 < rate <= 1000000
                or round(rate) != rate):
                raise ValueError('invalid source grid metadata')
            warm = round(rate)
            row = {'stratum': st, 'source_key': key, 'frozen_group_id': record['frozen_group_id'],
                   'source_cell_specimen_id': record['source_cell_specimen_id'],
                   'rate_hz': rate, 'command_samples': shape[0], 'warmup_samples': warm,
                   'native_intervals': warm+shape[0]-1, 'target_metadata_sha256': identity(sweep['features']),
                   'stimulus_metadata_sha256': identity(roles['stimulus']),
                   'response_grid_metadata_sha256': identity(roles['response'])}
            sweeps.append(row)
            summary = summaries[st]
            summary['groups'].add(record['frozen_group_id'])
            summary['recordings'].add(recording)
            summary['sweeps'] += 1
            summary['intervals'] += row['native_intervals']
    if any(s['sweeps'] == 0 for s in summaries.values()):
        raise ValueError('empty characterized stratum')
    sweeps.sort(key=lambda r: (strata.index(r['stratum']), *r['source_key']))
    entries = [{'stratum': st, 'candidate_index': i, 'parameter_index': i} for st in strata for i in range(n)]
    cases = [{'stratum': s['stratum'], 'candidate_index': i, 'source_key': s['source_key']}
             for s in sweeps for i in range(n)]
    retained = {}
    for st, state in original['strata'].items():
        if st not in strata:
            raise ValueError('historical stratum differs')
        retained[st] = len(state['candidates'])
        for key, row in state['candidates'].items():
            i = int(key)
            if str(i) != key or not 0 <= i < n or canonical(row['parameters']) != canonical(table[i]):
                raise ValueError('historical candidate parameters changed')
    by_stratum = {st: {'supported_recordings': len(s['recordings']), 'supported_groups': len(s['groups']),
        'sweeps': s['sweeps'], 'candidate_sweep_cases': s['sweeps']*n,
        'native_intervals_per_parameter_vector': s['intervals']} for st, s in summaries.items()}
    intervals = sum(s['intervals'] for s in summaries.values())*n
    required = {f'{bits}/{factor}' for bits in (53, 160, 320) for factor in (4, 8, 16)}
    if set(benchmark['cases']) != required:
        raise ValueError('synthetic timing arm scope differs')
    seconds = {}
    for name, timing in benchmark['cases'].items():
        if (type(timing['native_steps']) is not int or timing['native_steps'] <= 0
            or not math.isfinite(timing['elapsed_s']) or timing['elapsed_s'] <= 0):
            raise ValueError('invalid synthetic timing')
        seconds[name] = timing['elapsed_s']/timing['native_steps']*intervals
    return {'status': 'metadata-only-full-support-ledger-not-execution-authority',
        'parameters': table, 'candidate_table_sha256': identity(table),
        'candidate_stratum_entries': entries, 'sweeps': sweeps, 'candidate_sweep_cases': cases,
        'by_stratum': by_stratum, 'original_retained_rows': retained,
        'totals': {'parameter_vectors': n, 'candidate_stratum_entries': len(entries),
                   'supported_groups': sum(len(s['groups']) for s in summaries.values()),
                   'training_sweeps': len(sweeps), 'candidate_sweep_cases': len(cases),
                   'warmup_inclusive_native_intervals': intervals},
        'rough_serial_seconds_by_arm': seconds,
        'rough_nine_arm_serial_seconds': math.fsum(seconds.values()),
        'nine_arm_five_state_uncompressed_bytes': (intervals+len(cases))*5*8*9,
        'cost_is_synthetic_extrapolation_not_runtime_guarantee': True,
        'recording_arrays_opened': False, 'simulations_run': False,
        'fitting_restart_authorized': False, 'held_out_access_authorized': False,
        'network_promotion': False, 'scientific_goal_complete': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    output = parser.parse_args().output
    pins = {}
    plan, protocol = read_metadata(PLAN, pins), read_metadata(PROTOCOL, pins)
    runtime = {name: version(name) for name in protocol['runtime']}
    if runtime != protocol['runtime']:
        raise ValueError('frozen runtime differs')
    pins[protocol['training_catalog']] = protocol['training_catalog_sha256']
    catalog = read_metadata(protocol['training_catalog'], pins)
    pins[str(SOURCE)] = catalog['source_hashes'][str(SOURCE)]
    source = read_metadata(SOURCE, pins)
    pairings = {}
    for batch in source['batches']:
        path = batch['pairing']
        pins[path] = catalog['source_hashes'][path]
        pairings[batch['name']] = read_metadata(path, pins)
    pins[str(ORIGINAL)] = '29539a6dc1feb29b3ba2d4d87a9324b91ead0681eddf6b6db80abad80aa7f105'
    pins[str(BENCHMARK)] = '145b0a5ec6caad807ed4f9e4f1b32bcb977396ad125f6df5c909216a9c956bc3'
    original, benchmark = read_metadata(ORIGINAL, pins), read_metadata(BENCHMARK, pins)
    table = [candidate_dict(p) for p in candidates(protocol['search'])]
    report = assemble(protocol, catalog, pairings, table, original, benchmark)
    expected = plan['frozen_workload']
    if (report['by_stratum'] != expected['by_stratum']
        or report['candidate_table_sha256'] != expected['candidate_table_sha256']):
        raise ValueError('ledger does not reproduce disclosed plan scope')
    for name in ('build_patchseq_numerical_workload.py', 'patchseq_fit_search.py', 'patchseq_isolated_hh.py'):
        path = Path(__file__).with_name(name)
        pins[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in pins.items()):
        raise ValueError('metadata or source changed while accounting')
    report['source_hashes'] = pins
    report['runtime'] = runtime
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as file:
        file.write(canonical(report))
    print('Metadata ledger retained:', report['totals'], flush=True)


if __name__ == '__main__':
    main()
