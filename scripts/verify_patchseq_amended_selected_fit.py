"""Terminal-gated selected-map cross-engine checks; never opens held-out data."""

from __future__ import annotations

import fcntl
import json
from importlib.metadata import version
from pathlib import Path

from audit_patchseq_fit_checkpoint import audit
from patchseq_amended_selected_agreement import selected_agreement
from patchseq_hh_compiled_driver import prepare_build
from patchseq_independent_fit_inputs import inputs_for_training
from patchseq_verification_routes import require_backend
from verify_patchseq_fit_checkpoint_structure import (
    digest,
    read_yaml,
    terminal_checkpoint,
    verify_source_pins,
    verify_training_bodies,
)
from verify_patchseq_isolated_fit_replay import save

REG = Path('docs/validation-results/post2008-patchseq-amended-selected-verification-registration-1473.yaml')


def require_completed_replay(independent, original, sha, protocol):
    if (independent.get('status') != 'all-candidate-forward-feature-loss-reconstruction-retained'
        or independent.get('input_checkpoint_sha256') != sha
        or independent.get('all_reconstructions_agree') is not True
        or independent.get('held_out_access_authorized') is not False
        or set(independent['strata']) != set(protocol['strata'])):
        raise ValueError('complete agreeing independent table prerequisite missing')
    for stratum in protocol['strata']:
        rows = independent['strata'][stratum]['candidates']
        if (set(rows) != set(original['strata'][stratum]['candidates'])
            or not all(row['passed'] is True for row in rows.values())):
            raise ValueError('independent candidate verification incomplete or disagreeing')


def selected_summary(original, results, expected_keys):
    if set(results) != expected_keys:
        raise ValueError('selected training sweep support incomplete')
    agreement = all(r['all_reconstruction_checks_pass'] is True for r in results.values())
    numerical = agreement and all(r['independent_numerical_controls']['passed'] is True
                                 for r in results.values())
    selected = original['selected_candidate']
    training = original['candidates'][str(selected)]['all_training_observable_gates_pass']
    if type(training) is not bool:
        raise ValueError('training fit claim must be explicit boolean')
    return {'selected_candidate': selected, 'all_selected_reconstructions_agree': agreement,
            'all_selected_numerical_controls_pass': numerical,
            'sealed_training_observable_gates_pass': training,
            'training_and_numerical_eligible': agreement and numerical and training,
            'held_out_access_authorized': False, 'network_promotion': False}


def main(registration=REG):
    reg = read_yaml(registration)
    protocol, seal = read_yaml(reg['protocol']), read_yaml(reg['seal'])
    if any(version(package) != expected for package, expected in protocol['runtime'].items()):
        raise ValueError('selected verification runtime differs')
    catalog = read_yaml(protocol['training_catalog'])
    structure = read_yaml(reg['structure_result'])
    independent_before = digest(reg['independent_checkpoint'])
    independent = json.loads(Path(reg['independent_checkpoint']).read_text())
    require_backend(reg, independent)
    source = read_yaml('docs/validation-results/post2008-patchseq-fitting-target-registration-1429.yaml')
    nominal = read_yaml(source['nominal'])
    output = Path(reg['output_directory'])
    with terminal_checkpoint(protocol['output']) as (original, sha):
        if structure['checkpoint_sha256'] != sha:
            raise ValueError('structure audit belongs to different main output')
        require_completed_replay(independent, original, sha, protocol)
        audit(original, protocol, catalog, seal)
        verify_source_pins(original['source_hashes'])
        verify_source_pins(structure['source_hashes'])
        verify_source_pins(independent['source_hashes'])
        # A live independent verifier can still be publishing its terminal snapshot.
        independent_lock = Path(reg['independent_checkpoint']).parent/'verifier.lock'
        with independent_lock.open('r') as verification_lock:
            fcntl.flock(verification_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            independent_sha = digest(reg['independent_checkpoint'])
            if independent_sha != independent_before:
                raise ValueError('independent checkpoint changed during prerequisite inspection')
            output.mkdir(parents=True, exist_ok=True)
            with (output/'verifier.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                code = (Path(registration), Path(__file__), Path('scripts/patchseq_amended_selected_agreement.py'),
                        Path('scripts/patchseq_verification_routes.py'))
                pins = {str(p): digest(p) for p in code}
                _, build = prepare_build(output/'build')
                pins[build['binary_path']] = build['binary_sha256']
                pins[str(output/'build'/'build.yaml')] = digest(output/'build'/'build.yaml')
                pins.update(original['source_hashes'])
                pins.update(structure['source_hashes'])
                pins.update(independent['source_hashes'])
                pins[reg['independent_checkpoint']] = independent_sha
                pins[reg['structure_result']] = digest(reg['structure_result'])
                path = output/'checkpoint.json'
                state = json.loads(path.read_text()) if path.exists() else {
                    'input_checkpoint_sha256': sha, 'source_hashes': pins, 'strata': {}}
                if state['input_checkpoint_sha256'] != sha or state['source_hashes'] != pins:
                    raise ValueError('selected verifier checkpoint code or inputs differ')
                if not set(state['strata']) <= set(protocol['strata']):
                    raise ValueError('selected checkpoint stratum scope differs')
                for stratum in protocol['strata']:
                    gene, layer = stratum.split('-')
                    records = [r for r in catalog['records'] if r['partition'] == 'fitting'
                               and (r['source_group'], r['layer_label']) == (gene, layer)]
                    inputs, ledger, charged = inputs_for_training(records, source, nominal)
                    reference = original['strata'][stratum]
                    if ledger != reference['opened_training_files'] or charged != reference['decoded_command_float64_bytes']:
                        raise ValueError('selected command reader accounting differs')
                    parameter = reference['candidates'][str(reference['selected_candidate'])]['parameters']
                    rows = state['strata'].setdefault(stratum, {'sweeps': {}})
                    expected = {json.dumps(key) for key in inputs}
                    if not set(rows['sweeps']) <= expected:
                        raise ValueError('selected checkpoint sweep scope differs')
                    for key, (command, rate, holding, epoch) in inputs.items():
                        case = json.dumps(key)
                        if case in rows['sweeps']:
                            continue
                        rows['sweeps'][case] = selected_agreement(parameter, command, rate, holding,
                            epoch, reference['numerical_controls'][case])
                        verify_source_pins(pins)
                        save(path, state)
                        print(stratum, 'selected sweeps retained', len(rows['sweeps']), '/', len(inputs), flush=True)
                    rows['summary'] = selected_summary(reference, rows['sweeps'], expected)
                    save(path, state)
                if verify_training_bodies(original, protocol, catalog) != structure['training_body_hashes']:
                    raise ValueError('source bodies changed during selected verification')
                verify_source_pins(pins)
                state.update(status='selected-training-waveform-feature-and-numerical-checks-retained',
                             held_out_access_authorized=False, biological_validation_complete=False,
                             network_promotion=False, frozen_baseline_modified=False)
                save(path, state)


if __name__ == '__main__':
    main()
