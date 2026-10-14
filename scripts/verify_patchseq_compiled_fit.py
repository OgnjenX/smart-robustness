"""Prospective compiled verifier route; no alteration of fitting or NumPy route."""

from __future__ import annotations

import fcntl
import json
from importlib.metadata import version
from pathlib import Path

from audit_patchseq_fit_checkpoint import audit
from patchseq_hh_compiled_driver import compiled_driver
from patchseq_independent_fit_inputs import inputs_for_training
from patchseq_independent_fit_loss import reconstruct_training_loss
from verify_patchseq_fit_checkpoint_structure import (
    digest,
    read_yaml,
    terminal_checkpoint,
    verify_source_pins,
    verify_training_bodies,
)
from verify_patchseq_fitting_targets import reconstruct
from verify_patchseq_isolated_fit_replay import close, save

REG = Path('docs/validation-results/post2008-patchseq-compiled-verifier-amendment-1457.yaml')


def verify_candidate(replay, parameter, records, inputs, expected, warmup):
    predictions, errors = {}, []
    for key, (command, rate, holding, epoch) in inputs.items():
        try:
            v = replay([parameter], command, rate, holding, warmup)[0]
            predictions[key] = reconstruct(v, command, rate, epoch, holding)
        except (ValueError, FloatingPointError, OverflowError) as exc:
            predictions[key] = None
            errors.append({'source_key': list(key), 'independent_error': str(exc)})
    result = reconstruct_training_loss(records, predictions)
    same_groups = set(result['group_losses']) == set(expected['group_losses'])
    checks = {'group_support': same_groups, 'total_loss': close(result['loss'], expected['loss']),
              'group_losses': same_groups and all(close(value, expected['group_losses'][g])
                                                  for g, value in result['group_losses'].items()),
              'observable_gates': result['all_applicable_gates_pass'] == expected['all_training_observable_gates_pass'],
              'failure_support': {tuple(e['source_key']) for e in errors} == {
                  tuple(e['source_key']) for e in expected['simulation_failures']}}
    return {'passed': all(checks.values()), 'checks': checks, 'independent_loss': result['loss'],
            'group_losses': result['group_losses'], 'simulation_failures': errors,
            'all_training_observable_gates_pass': result['all_applicable_gates_pass']}


def require_equivalence(result):
    if (result.get('status') != 'synthetic-compiled-equivalence-retained'
        or result.get('all_cases_pass') is not True or result.get('biological_data_used') is not False
        or len(result['cases']) != 386 or any(r['passed'] is not True for r in result['cases'])):
        raise ValueError('fixed compiled equivalence prerequisite missing')


def main():
    reg = read_yaml(REG)
    if reg['verification_only_amendment'] is not True:
        raise ValueError('compiled route not explicitly sealed')
    if digest(reg['equivalence_result']) != reg['equivalence_sha256']:
        raise ValueError('equivalence evidence differs')
    equivalent = read_yaml(reg['equivalence_result'])
    require_equivalence(equivalent)
    verify_source_pins(equivalent['source_hashes'])
    protocol, seal = read_yaml(reg['protocol']), read_yaml(reg['seal'])
    if any(version(p) != v for p, v in protocol['runtime'].items()):
        raise ValueError('compiled verification runtime differs')
    structure = read_yaml(reg['structure_result'])
    if structure['status'] != 'structural-accounting-only-forward-replay-not-verified' or structure['held_out_access_authorized'] is not False:
        raise ValueError('terminal source audit prerequisite missing')
    catalog = read_yaml(protocol['training_catalog'])
    source = read_yaml('docs/validation-results/post2008-patchseq-fitting-target-registration-1429.yaml')
    nominal = read_yaml(source['nominal'])
    with terminal_checkpoint(protocol['output']) as (original, sha):
        if sha != structure['checkpoint_sha256']:
            raise ValueError('main result changed since source audit')
        audit(original, protocol, catalog, seal)
        verify_source_pins(original['source_hashes'])
        verify_source_pins(structure['source_hashes'])
        output = Path(reg['output_directory'])
        output.mkdir(parents=True, exist_ok=True)
        with (output/'verifier.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with compiled_driver(output/'build') as (replay, build):
                code = [REG, Path(__file__), Path('scripts/patchseq_hh_compiled_driver.py'),
                        Path('scripts/patchseq_hh_replay_kernel.cpp'),
                        Path('scripts/patchseq_independent_fit_inputs.py'),
                        Path('scripts/patchseq_independent_fit_loss.py'),
                        Path('scripts/verify_patchseq_fitting_targets.py'),
                        Path('scripts/verify_patchseq_isolated_fit_replay.py')]
                pins = {str(p): digest(p) for p in code}
                pins.update(original['source_hashes']); pins.update(structure['source_hashes'])
                pins.update(equivalent['source_hashes'])
                pins.update({reg['structure_result']: digest(reg['structure_result']),
                             reg['equivalence_result']: reg['equivalence_sha256'],
                             build['binary_path']: build['binary_sha256'],
                             str(output/'build'/'build.yaml'): digest(output/'build'/'build.yaml')})
                path = output/'checkpoint.json'
                state = json.loads(path.read_text()) if path.exists() else {
                    'input_checkpoint_sha256': sha, 'source_hashes': pins,
                    'compiled_build': build, 'verification_backend': 'independent-compiled-RK4', 'strata': {}}
                if state['input_checkpoint_sha256'] != sha or state['source_hashes'] != pins or state['compiled_build'] != build:
                    raise ValueError('compiled checkpoint source or binary differs')
                if not set(state['strata']) <= set(protocol['strata']):
                    raise ValueError('compiled checkpoint stratum scope differs')
                for stratum in protocol['strata']:
                    gene, layer = stratum.split('-')
                    records = [r for r in catalog['records'] if r['partition'] == 'fitting'
                               and (r['source_group'], r['layer_label']) == (gene, layer)]
                    reference = original['strata'][stratum]
                    rows = state['strata'].setdefault(stratum, {'candidates': {}})
                    if not set(rows['candidates']) <= set(reference['candidates']):
                        raise ValueError('compiled checkpoint candidate scope differs')
                    if set(rows['candidates']) == set(reference['candidates']):
                        continue
                    inputs, ledger, charged = inputs_for_training(records, source, nominal)
                    if ledger != reference['opened_training_files'] or charged != reference['decoded_command_float64_bytes']:
                        raise ValueError('compiled command reader accounting differs')
                    rows.update(opened_training_files=ledger, decoded_command_float64_bytes=charged)
                    for key in sorted(reference['candidates'], key=int):
                        if key in rows['candidates']:
                            continue
                        expected = reference['candidates'][key]
                        rows['candidates'][key] = verify_candidate(replay, expected['parameters'], records,
                            inputs, expected, protocol['replay']['warmup_s'])
                        verify_source_pins(pins); save(path, state)
                        print(stratum, 'compiled candidates retained', len(rows['candidates']), '/', len(reference['candidates']), flush=True)
                if verify_training_bodies(original, protocol, catalog) != structure['training_body_hashes']:
                    raise ValueError('recording bodies changed during compiled verification')
                verify_source_pins(pins)
                state.update(status='all-candidate-forward-feature-loss-reconstruction-retained',
                             all_reconstructions_agree=all(r['passed'] for s in state['strata'].values() for r in s['candidates'].values()),
                             held_out_access_authorized=False, selected_numerical_controls_independently_verified=False,
                             biological_validation_complete=False, network_promotion=False,
                             frozen_baseline_modified=False)
                save(path, state)


if __name__ == '__main__':
    main()
