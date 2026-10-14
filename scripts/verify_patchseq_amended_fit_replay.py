"""Independent complete-table replay; gated on terminal source/structure audit."""

from __future__ import annotations

import fcntl
import json
import math
import os
from importlib.metadata import version
from pathlib import Path

from audit_patchseq_fit_checkpoint import audit
from patchseq_hh_independent_replay import replay
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

REG = Path('docs/validation-results/post2008-patchseq-amended-independent-replay-registration-1471.yaml')


def close(left, right):
    return (math.isfinite(left) and math.isfinite(right)
            and abs(left-right) <= 1e-6+1e-8*abs(right))


def verify_candidate(parameter, records, inputs, expected, warmup):
    predictions, failures = {}, []
    for key, (command, rate, holding, epoch) in inputs.items():
        try:
            voltage = replay([parameter], command, rate, holding, warmup)[0]
            predictions[key] = reconstruct(voltage, command, rate, epoch, holding)
        except (ValueError, FloatingPointError, OverflowError) as exc:
            predictions[key] = None
            failures.append({'source_key': list(key), 'independent_error': str(exc)})
    actual = reconstruct_training_loss(records, predictions)
    flags = {
        'group_support': set(actual['group_losses']) == set(expected['group_losses']),
        'total_loss': close(actual['loss'], expected['loss']),
        'group_losses': set(actual['group_losses']) == set(expected['group_losses']) and all(
            close(actual['group_losses'][g], expected['group_losses'][g]) for g in actual['group_losses']),
        'observable_gates': actual['all_applicable_gates_pass'] == expected['all_training_observable_gates_pass'],
        'failure_support': {tuple(f['source_key']) for f in failures} == {
            tuple(f['source_key']) for f in expected['simulation_failures']},
    }
    return {'passed': all(flags.values()), 'checks': flags, 'independent_loss': actual['loss'],
            'group_losses': actual['group_losses'], 'simulation_failures': failures,
            'all_training_observable_gates_pass': actual['all_applicable_gates_pass']}


def save(path, state):
    temporary = path.with_suffix('.pending')
    with temporary.open('w') as file:
        json.dump(state, file, sort_keys=True, allow_nan=False)
        file.flush(); os.fsync(file.fileno())
    os.replace(temporary, path)


def main():
    reg = read_yaml(REG)
    structure = read_yaml(reg['structure_result'])
    if (structure['status'] != 'structural-accounting-only-forward-replay-not-verified'
        or structure.get('forward_simulations_verified') is not False
        or structure['held_out_access_authorized'] is not False):
        raise ValueError('terminal source audit prerequisite missing')
    protocol, seal = read_yaml(reg['protocol']), read_yaml(reg['seal'])
    if any(version(package) != expected for package, expected in protocol['runtime'].items()):
        raise ValueError('independent runtime differs from sealed source contract')
    catalog = read_yaml(protocol['training_catalog'])
    source = read_yaml('docs/validation-results/post2008-patchseq-fitting-target-registration-1429.yaml')
    nominal = read_yaml(source['nominal'])
    output = Path(reg['output_directory'])
    output.mkdir(parents=True, exist_ok=True)
    with terminal_checkpoint(protocol['output']) as (original, sha), (output/'verifier.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if sha != structure['checkpoint_sha256']:
            raise ValueError('terminal checkpoint changed since source audit')
        audit(original, protocol, catalog, seal)
        verify_source_pins(original['source_hashes'])
        verify_source_pins(structure['source_hashes'])
        code = (Path(__file__), Path('scripts/patchseq_independent_fit_inputs.py'),
                Path('scripts/patchseq_hh_independent_replay.py'),
                Path('scripts/patchseq_independent_fit_loss.py'),
                Path('scripts/verify_patchseq_fitting_targets.py'), REG)
        pins = {str(p): digest(p) for p in code}
        pins[reg['structure_result']] = digest(reg['structure_result'])
        pins.update(structure['source_hashes'])
        pins.update(original['source_hashes'])
        path = output/'checkpoint.json'
        state = json.loads(path.read_text()) if path.exists() else {
            'input_checkpoint_sha256': sha, 'source_hashes': pins,
            'verification_backend': 'independent-NumPy-RK4', 'strata': {}}
        if state['input_checkpoint_sha256'] != sha or state['source_hashes'] != pins:
            raise ValueError('independent checkpoint belongs to different inputs or code')
        for stratum in protocol['strata']:
            gene, layer = stratum.split('-')
            records = [r for r in catalog['records'] if r['partition'] == 'fitting'
                       and (r['source_group'], r['layer_label']) == (gene, layer)]
            recorded = original['strata'][stratum]['candidates']
            rows = state['strata'].setdefault(stratum, {'candidates': {}})
            if not set(rows['candidates']) <= set(recorded):
                raise ValueError('independent checkpoint candidate scope differs')
            if set(rows['candidates']) == set(recorded):
                continue
            inputs, ledger, charged = inputs_for_training(records, source, nominal)
            if ledger != original['strata'][stratum]['opened_training_files'] or charged != original['strata'][stratum]['decoded_command_float64_bytes']:
                raise ValueError('independent command reader scope or accounting differs')
            rows.update(opened_training_files=ledger, decoded_command_float64_bytes=charged)
            for key in sorted(recorded, key=int):
                if key in rows['candidates']:
                    continue
                expected = recorded[key]
                rows['candidates'][key] = verify_candidate(expected['parameters'], records, inputs,
                    expected, protocol['replay']['warmup_s'])
                verify_source_pins(pins)
                save(path, state)
                print(stratum, 'independently retained', len(rows['candidates']), '/', len(recorded), flush=True)
        bodies = verify_training_bodies(original, protocol, catalog)
        if bodies != structure['training_body_hashes']:
            raise ValueError('training bodies differ from terminal source audit')
        verify_source_pins(pins)
        passed = all(r['passed'] for s in state['strata'].values() for r in s['candidates'].values())
        state.update(status='all-candidate-forward-feature-loss-reconstruction-retained',
                     training_body_hashes=bodies,
                     all_reconstructions_agree=passed, held_out_access_authorized=False,
                     selected_numerical_controls_independently_verified=False,
                     biological_validation_complete=False, network_promotion=False)
        save(path, state)


if __name__ == '__main__':
    main()
