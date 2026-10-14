"""Explicit amended fitter; retains original checkpoint and excludes self-verification."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
from importlib.metadata import version
from pathlib import Path

from patchseq_compiled_fit_engine import adapter, numerical_controls, score_batch
from patchseq_fit_search import candidate_dict, candidates, selected_candidate
from patchseq_hh_compiled_driver import compiled_driver
from run_patchseq_isolated_fit import (
    checkpoint,
    fitting_records,
    prepare_inputs,
    read_yaml,
    validate_rows,
)
from verify_patchseq_compiled_fit import verify_candidate
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-compiled-fitting-amendment-1466.yaml')


def retained_scope(original, table, strata):
    if original.get('status') is not None or not set(original['strata']) <= set(strata):
        raise ValueError('transition requires a partial original checkpoint')
    if original['candidate_table'] != [candidate_dict(p) for p in table]:
        raise ValueError('original candidate table differs')
    for rows in original['strata'].values():
        validate_rows(rows['candidates'], table)
        if rows.get('numerical_controls') or rows.get('selected_candidate') is not None:
            raise ValueError('selected-stage original checkpoint requires a separate transition design')
        ids = sorted(map(int, rows['candidates']))
        if ids != list(range(len(ids))) or len(ids) % 4:
            raise ValueError('original checkpoint is not a complete contiguous batch prefix')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    reg = read_yaml(REG)
    if Path(args.output) != Path(reg['output']):
        raise ValueError('amended output scope differs')
    seal = read_yaml(reg['amended_execution_seal'])
    pins = dict(seal['source_sha256'])
    pins[reg['amended_execution_seal']] = digest(reg['amended_execution_seal'])
    verified = read_yaml(reg['verified_catalog'])
    if verified['status'] != 'all-training-response-features-and-held-out-no-access-boundaries-independently-reconstructed':
        raise ValueError('independent training catalog not verified')
    pins.update(verified['source_hashes'])
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('amended seal source differs')
    if any(version(p) != v for p, v in reg['runtime'].items()):
        raise ValueError('amended runtime differs')
    equivalent = json.loads(Path(reg['equivalence_checkpoint']).read_text())
    if (digest(reg['equivalence_checkpoint']) != reg['equivalence_sha256']
        or equivalent.get('all_cases_pass') is not True or len(equivalent['cases']) != 386
        or any(c['passed'] is not True for c in equivalent['cases'].values())
        or equivalent['biological_data_used'] is not False):
        raise ValueError('direct equivalence evidence missing or changed')
    for p, h in equivalent['source_hashes'].items():
        if digest(p) != h:
            raise ValueError('equivalence source changed')
    for key in ('training_catalog', 'verified_catalog'):
        if digest(reg[key]) != reg[key+'_sha256']:
            raise ValueError('training prerequisites differ')
    catalog = read_yaml(reg['training_catalog'])
    source = read_yaml('docs/validation-results/post2008-patchseq-fitting-target-registration-1429.yaml')
    nominal = read_yaml(source['nominal'])
    table = candidates(reg['search'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive predecessor lock prevents original resume throughout the new run.
    with (Path(reg['predecessor_directory'])/'runner.lock').open('a') as old_lock, (output/'runner.lock').open('a') as lock:
        fcntl.flock(old_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        predecessor = Path(reg['predecessor_directory'])/'checkpoint.json'
        sha = digest(predecessor)
        original = json.loads(predecessor.read_text())
        retained_scope(original, table, reg['strata'])
        for p, h in original['source_hashes'].items():
            if digest(p) != h:
                raise ValueError('original checkpoint source differs')
        archive = output/'predecessor-checkpoint.json'
        if not archive.exists():
            with archive.open('xb') as file:
                file.write(predecessor.read_bytes())
        if digest(archive) != sha:
            raise ValueError('original checkpoint changed after transition capture')
        with compiled_driver(output/'build') as (compiled, build):
            replay = adapter(compiled)
            pins.update(original['source_hashes'])
            pins.update({str(REG): digest(REG), reg['equivalence_checkpoint']: reg['equivalence_sha256'],
                         str(archive): sha, str(predecessor): sha,
                         build['binary_path']: build['binary_sha256'],
                         str(output/'build'/'build.yaml'): digest(output/'build'/'build.yaml')})
            path = output/'checkpoint.json'
            state = json.loads(path.read_text()) if path.exists() else {
                'source_hashes': pins, 'candidate_table': original['candidate_table'],
                'strata': copy.deepcopy(original['strata']), 'predecessor_checkpoint_sha256': sha,
                'compiled_build': build, 'fitting_backend': 'compiled-RK4-with-retained-Brian2-prefix',
                'transition_checks': {}, 'held_out_response_features_read': False}
            if state['source_hashes'] != pins or state['compiled_build'] != build:
                raise ValueError('amended checkpoint source or binary differs')
            for stratum in reg['strata']:
                rows = state['strata'].setdefault(stratum, {'candidates': {}})
                validate_rows(rows['candidates'], table)
                records = fitting_records(catalog, stratum)
                inputs, opened, charged = prepare_inputs(records, source, nominal)
                old = original['strata'].get(stratum, {}).get('candidates', {})
                checks = state['transition_checks'].setdefault(stratum, {})
                if old and (opened != original['strata'][stratum]['opened_training_files']
                    or charged != original['strata'][stratum]['decoded_command_float64_bytes']):
                    raise ValueError('original source accounting differs')
                for key in sorted(old, key=int):
                    if key not in checks:
                        checks[key] = verify_candidate(compiled, old[key]['parameters'], records,
                            inputs, old[key], reg['replay']['warmup_s'])
                        checkpoint(path, state)
                    if checks[key]['passed'] is not True or rows['candidates'][key] != old[key]:
                        raise ValueError('retained original score disagreement; transition stopped')
                rows.update(opened_training_files=opened, decoded_command_float64_bytes=charged)
                origins = rows.setdefault('candidate_backends', {k: 'retained-Brian2' for k in old})
                width = reg['search']['candidate_batch_maximum']
                for start in range(0, len(table), width):
                    ids = list(range(start, min(start+width, len(table))))
                    if all(str(i) in rows['candidates'] for i in ids):
                        continue
                    if any(str(i) in rows['candidates'] for i in ids):
                        raise ValueError('partial amended batch')
                    results = score_batch(replay, [table[i] for i in ids], records, inputs,
                                          reg['replay']['warmup_s'])
                    rows['candidates'].update({str(i): r for i, r in zip(ids, results, strict=True)})
                    origins.update({str(i): 'compiled-RK4' for i in ids})
                    if any(digest(p) != h for p, h in pins.items()):
                        raise ValueError('source changed during amended fit')
                    checkpoint(path, state)
                    print(stratum, 'compiled candidates retained', len(rows['candidates']), '/ 128', flush=True)
                chosen = selected_candidate({int(i): r for i, r in rows['candidates'].items()}, len(table))
                rows['selected_candidate'] = chosen
                controls = rows.setdefault('numerical_controls', {})
                if chosen is not None:
                    for key, (command, rate, holding, epoch) in inputs.items():
                        case = json.dumps(key)
                        if case in controls:
                            continue
                        try:
                            controls[case] = numerical_controls(replay, table[chosen], command, rate,
                                                               holding, epoch, reg['replay']['warmup_s'])
                        except (ValueError, FloatingPointError, OverflowError) as exc:
                            controls[case] = {'passed': False, 'error': str(exc)}
                        checkpoint(path, state)
                rows['numerical_controls_complete'] = chosen is not None and len(controls) == len(inputs)
                rows['all_numerical_controls_pass'] = rows['numerical_controls_complete'] and all(
                    c['passed'] for c in controls.values())
                checkpoint(path, state)
            if any(digest(p) != h for p, h in pins.items()):
                raise ValueError('source changed before completion')
            state.update(status='training-search-and-selected-numerical-controls-retained-independent-verification-pending',
                         network_promotion=False, frozen_baseline_modified=False)
            checkpoint(path, state)


if __name__ == '__main__':
    main()
