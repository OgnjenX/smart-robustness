"""Independent metadata-only support reconstruction; never imports the builder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import qmc

PROTOCOL = Path('docs/validation-results/post2008-patchseq-isolated-fit-registration-1436.yaml')
PLAN = Path('docs/validation-results/post2008-patchseq-numerical-admissibility-plan-1497.yaml')
SOURCE = Path('docs/validation-results/post2008-patchseq-fitting-target-registration-1429.yaml')
ORIGINAL = Path('results/patchseq-isolated-fit-1438/checkpoint.json')
BENCHMARK = Path('results/patchseq-precision-synthetic-benchmark-1494/checkpoint.json')
INPUT = Path('results/patchseq-numerical-workload-1499/manifest.json')
OUTPUT = Path('results/patchseq-numerical-workload-independent-1500/manifest.yaml')
FIELDS = ('capacitance_pF', 'leak_nS', 'leak_reversal_mV', 'sodium_nS', 'potassium_nS',
          'm_current_nS', 'threshold_shift_mV', 'm_tau_max_ms')


def serialized(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal(a, b, message):
    if serialized(a) != serialized(b):
        raise ValueError(message)


def verify(report, protocol, catalog, pairings, table, original, benchmark):
    equal(report['parameters'], table, 'parameter vector table differs')
    equal(report['candidate_table_sha256'], hashlib.sha256(serialized(table)).hexdigest(), 'table digest differs')
    n = protocol['search']['candidates_per_stratum']
    strata = protocol['strata']
    actual_strata = {r['source_group']+'-'+r['layer_label'] for r in catalog['records'] if r['partition'] == 'fitting'}
    if not actual_strata <= set(strata):
        raise ValueError('unexpected fitting stratum')
    if len(table) != n or len(set(strata)) != len(strata):
        raise ValueError('frozen identity scope differs')
    entries = [{'stratum': st, 'candidate_index': i, 'parameter_index': i} for st in strata for i in range(n)]
    equal(report['candidate_stratum_entries'], entries, 'candidate stratum scope differs')
    expected_sweeps, summary = [], {}
    for st in strata:
        group, layer = st.split('-')
        records = [r for r in catalog['records'] if r['partition'] == 'fitting'
                   and (r['source_group'], r['layer_label']) == (group, layer)]
        groups, recordings, sweep_rows = set(), set(), []
        for record in records:
            selected = [s for s in record['sweeps'] if s['state'] == 'fitting-response-characterized']
            if not selected:
                continue
            rk = (record['source_batch'], record['slot'])
            if rk in recordings:
                raise ValueError('duplicate source recording')
            recordings.add(rk)
            groups.add(record['frozen_group_id'])
            paired = [p for p in pairings[rk[0]]['records'] if p['slot'] == rk[1]]
            if len(paired) != 1:
                raise ValueError('ambiguous pairing metadata')
            paired = paired[0]
            for sweep in selected:
                matches = [p for p in paired['sweeps'] if p['sweep_number'] == sweep['sweep_number']]
                if len(matches) != 1:
                    raise ValueError('ambiguous paired sweep')
                descriptors = [p for path in matches[0]['series_paths'] for p in paired['series'] if p['path'] == path]
                if len(descriptors) != 2 or sorted(p['role'] for p in descriptors) != ['response', 'stimulus']:
                    raise ValueError('stimulus/grid metadata identity differs')
                stimulus = next(p for p in descriptors if p['role'] == 'stimulus')
                response = next(p for p in descriptors if p['role'] == 'response')
                rate, samples = response['rate'], stimulus['data_shape'][0]
                row = {'stratum': st, 'source_key': [*rk, sweep['sweep_number']],
                       'frozen_group_id': record['frozen_group_id'],
                       'source_cell_specimen_id': record['source_cell_specimen_id'],
                       'rate_hz': rate, 'command_samples': samples, 'warmup_samples': round(rate),
                       'native_intervals': samples+round(rate)-1,
                       'target_metadata_sha256': hashlib.sha256(serialized(sweep['features'])).hexdigest(),
                       'stimulus_metadata_sha256': hashlib.sha256(serialized(stimulus)).hexdigest(),
                       'response_grid_metadata_sha256': hashlib.sha256(serialized(response)).hexdigest()}
                sweep_rows.append(row)
        if not sweep_rows or len({tuple(r['source_key']) for r in sweep_rows}) != len(sweep_rows):
            raise ValueError('missing or duplicate sweep identity')
        expected_sweeps.extend(sorted(sweep_rows, key=lambda r: r['source_key']))
        summary[st] = {'supported_recordings': len(recordings), 'supported_groups': len(groups),
                       'sweeps': len(sweep_rows), 'candidate_sweep_cases': n*len(sweep_rows),
                       'native_intervals_per_parameter_vector': sum(r['native_intervals'] for r in sweep_rows)}
    equal(report['sweeps'], expected_sweeps, 'source sweep/group/grid/target scope differs')
    if len({tuple(r['source_key']) for r in expected_sweeps}) != len(expected_sweeps):
        raise ValueError('source identity reused across strata')
    equal(report['by_stratum'], summary, 'stratum support summary differs')
    expected_cases = [{'stratum': row['stratum'], 'candidate_index': index, 'source_key': row['source_key']}
                      for row in expected_sweeps for index in range(n)]
    equal(report['candidate_sweep_cases'], expected_cases, 'candidate sweep support differs')
    intervals = n*sum(row['native_intervals'] for row in expected_sweeps)
    totals = {'parameter_vectors': n, 'candidate_stratum_entries': n*len(strata),
              'supported_groups': sum(r['supported_groups'] for r in summary.values()),
              'training_sweeps': len(expected_sweeps), 'candidate_sweep_cases': len(expected_cases),
              'warmup_inclusive_native_intervals': intervals}
    equal(report['totals'], totals, 'full table totals differ')
    retained = {}
    for st, old in original['strata'].items():
        if st not in strata:
            raise ValueError('historical stratum outside scope')
        retained[st] = len(old['candidates'])
        for key, row in old['candidates'].items():
            index = int(key)
            if str(index) != key or not 0 <= index < n:
                raise ValueError('historical candidate identity differs')
            equal(row['parameters'], table[index], 'historical parameter values differ')
    equal(report['original_retained_rows'], retained, 'retained prefix scope differs')
    expected_arms = {f'{b}/{f}' for b in (53, 160, 320) for f in (4, 8, 16)}
    if set(benchmark['cases']) != expected_arms or set(report['rough_serial_seconds_by_arm']) != expected_arms:
        raise ValueError('benchmark arm scope differs')
    costs = [intervals*r['elapsed_s']/r['native_steps'] for r in benchmark['cases'].values()]
    if not math.isclose(report['rough_nine_arm_serial_seconds'], sum(costs), rel_tol=1e-12, abs_tol=0):
        raise ValueError('rough timing total differs')
    for key, value in benchmark['cases'].items():
        if not math.isclose(report['rough_serial_seconds_by_arm'][key],
                            intervals*value['elapsed_s']/value['native_steps'], rel_tol=1e-12, abs_tol=0):
            raise ValueError('rough timing arm differs')
    equal(report['nine_arm_five_state_uncompressed_bytes'], (intervals+len(expected_cases))*40*9, 'storage total differs')
    for name in ('recording_arrays_opened', 'simulations_run', 'fitting_restart_authorized',
                 'held_out_access_authorized', 'network_promotion', 'scientific_goal_complete'):
        if report[name] is not False:
            raise ValueError('metadata ledger exceeds execution/access boundary')
    if report['cost_is_synthetic_extrapolation_not_runtime_guarantee'] is not True:
        raise ValueError('cost provenance caveat missing')
    equal(report['status'], 'metadata-only-full-support-ledger-not-execution-authority', 'ledger status differs')
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=INPUT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    input_sha = hashlib.sha256(raw).hexdigest()
    report = json.loads(raw)
    pins = dict(report['source_hashes'])
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError('ledger source snapshot differs')

    def load(path, expected=None):
        path = Path(path)
        if path.suffix not in ('.json', '.yaml'):
            raise ValueError('auditor forbids recording body reads')
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError('independent metadata anchor differs')
        pins[str(path)] = actual
        return json.loads(path.read_text()) if path.suffix == '.json' else yaml.safe_load(path.read_text())

    protocol, plan = load(PROTOCOL), load(PLAN)
    runtime = {name: version(name) for name in protocol['runtime']}
    equal(runtime, protocol['runtime'], 'frozen verifier runtime differs')
    equal(report['runtime'], runtime, 'ledger runtime differs')
    catalog = load(protocol['training_catalog'], protocol['training_catalog_sha256'])
    source = load(SOURCE, catalog['source_hashes'][str(SOURCE)])
    pairings = {b['name']: load(b['pairing'], catalog['source_hashes'][b['pairing']]) for b in source['batches']}
    original = load(ORIGINAL, '29539a6dc1feb29b3ba2d4d87a9324b91ead0681eddf6b6db80abad80aa7f105')
    benchmark = load(BENCHMARK, '145b0a5ec6caad807ed4f9e4f1b32bcb977396ad125f6df5c909216a9c956bc3')
    search = protocol['search']
    points = qmc.Sobol(d=8, scramble=False).random_base2(int(math.log2(search['candidates_per_stratum'])))
    bounds = np.array([search['bounds'][name] for name in FIELDS])
    values = points*(bounds[:, 1]-bounds[:, 0])+bounds[:, 0]
    table = [dict(zip(FIELDS, map(float, row), strict=True)) for row in values]
    totals = verify(report, protocol, catalog, pairings, table, original, benchmark)
    equal(report['candidate_table_sha256'], plan['frozen_workload']['candidate_table_sha256'], 'plan table anchor differs')
    equal(report['by_stratum'], plan['frozen_workload']['by_stratum'], 'plan support anchor differs')
    pins[str(Path(__file__))] = sha(__file__)
    if sha(args.input) != input_sha or any(sha(p) != h for p, h in pins.items()):
        raise ValueError('metadata changed during independent verification')
    result = {'status': 'independent-metadata-workload-accounting-verified-not-scientific-validation',
              'input': str(args.input), 'input_sha256': input_sha, 'source_hashes': pins,
              'all_metadata_checks_agree': True, 'totals': totals, 'recording_arrays_opened': False,
              'simulations_run': False, 'fitting_restart_authorized': False, 'held_out_access_authorized': False,
              'scientific_goal_complete': False, 'shared_dependency': 'SciPy Sobol direction numbers; separate table/support construction'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        yaml.safe_dump(result, file, sort_keys=False)
    print('Independent workload accounting verified:', totals, flush=True)


if __name__ == '__main__':
    main()
