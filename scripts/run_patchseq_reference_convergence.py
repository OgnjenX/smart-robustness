"""Checkpointed fixed-input reference study; never fits or changes production."""

from __future__ import annotations

import fcntl
import json
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path

import numpy as np
from patchseq_adaptive_reference import replay as adaptive_replay
from patchseq_fit_objective import score_sweep
from patchseq_hh_compiled_driver import compiled_driver
from patchseq_independent_fit_inputs import inputs_for_training
from patchseq_response_features import characterize
from patchseq_selected_fit_agreement import feature_agreement
from run_patchseq_isolated_fit import checkpoint, fitting_records, read_yaml
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-reference-convergence-registration-1483.yaml')


def qualification(comparisons):
    required = ('DOP853-medium/DOP853-tight', 'Radau-medium/Radau-tight',
                'DOP853-tight/Radau-tight')
    return all(name in comparisons and comparisons[name]['passed'] is True for name in required)


def main():
    reg = read_yaml(REG)
    for name in ('original', 'predecessor'):
        path = reg['original_checkpoint'] if name == 'original' else reg['predecessor']
        if digest(path) != reg[name+'_sha256']:
            raise ValueError('fixed diagnostic prerequisite changed')
    original = json.loads(Path(reg['original_checkpoint']).read_text())
    previous = json.loads(Path(reg['predecessor']).read_text())
    if previous.get('status') != 'fixed-input-numerical-localization-retained-no-production-change':
        raise ValueError('previous diagnostic not terminal')
    pins = dict(previous['source_hashes'])
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('prior source chain changed')
    protocol = read_yaml('docs/validation-results/post2008-patchseq-isolated-fit-registration-1436.yaml')
    if any(version(k) != v for k, v in protocol['runtime'].items()):
        raise ValueError('diagnostic runtime differs')
    if digest(protocol['training_catalog']) != protocol['training_catalog_sha256']:
        raise ValueError('training catalog changed')
    for name in ('patchseq_adaptive_reference.py', 'patchseq_independent_fit_inputs.py',
                 'patchseq_hh_compiled_driver.py', 'patchseq_fit_objective.py',
                 'patchseq_response_features.py', 'patchseq_selected_fit_agreement.py'):
        p = Path('scripts')/name; pins[str(p)] = digest(p)
    pins.update({str(REG): digest(REG), str(Path(__file__)): digest(__file__),
                 reg['predecessor']: reg['predecessor_sha256'],
                 reg['original_checkpoint']: reg['original_sha256']})
    catalog = read_yaml(protocol['training_catalog'])
    records = fitting_records(catalog, reg['stratum'])
    targets = {(r['source_batch'], r['slot'], s['sweep_number']): s['features']
               for r in records for s in r['sweeps'] if s['state'] == 'fitting-response-characterized'}
    parameter = original['strata'][reg['stratum']]['candidates'][str(reg['candidate_index'])]['parameters']
    source = read_yaml('docs/validation-results/post2008-patchseq-fitting-target-registration-1429.yaml')
    output = Path(reg['output']); output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for path, mode in ((output/'runner.lock', fcntl.LOCK_EX),
                           (Path(reg['original_checkpoint']).parent/'runner.lock', fcntl.LOCK_SH),
                           (Path('results/patchseq-compiled-fit-1467/runner.lock'), fcntl.LOCK_SH)):
            lock = stack.enter_context(path.open('a'))
            fcntl.flock(lock, mode | fcntl.LOCK_NB)
        compiled, build = stack.enter_context(compiled_driver(output/'build'))
        pins.update({build['binary_path']: build['binary_sha256'],
                     str(output/'build'/'build.yaml'): digest(output/'build'/'build.yaml')})
        path = output/'checkpoint.json'
        state = json.loads(path.read_text()) if path.exists() else {
            'source_hashes': pins, 'compiled_build': build, 'parameters': parameter, 'cases': {},
            'production_protocol_changed': False, 'held_out_response_features_read': False,
            'fitting_resumed': False, 'scientific_goal_complete': False}
        if state['source_hashes'] != pins or state['compiled_build'] != build or state['parameters'] != parameter:
            raise ValueError('reference checkpoint belongs to different sources or build')
        labels = [m+'-'+level['name'] for m in reg['methods'] for level in reg['accuracy_levels']]
        labels += ['RK4-refinement-'+str(k) for k in reg['compiled_RK4_diagnostic_refinements']]
        expected = {'-'.join(map(str,key))+'/'+label for key in reg['source_keys'] for label in labels}
        if not set(state['cases']) <= expected or len(expected) != reg['case_count']:
            raise ValueError('reference case scope differs')
        inputs, ledger, charged = inputs_for_training(records, source, read_yaml(source['nominal']))
        state.update(opened_training_command_files=ledger, decoded_command_float64_bytes=charged)
        for raw_key in reg['source_keys']:
            key = tuple(raw_key); case = '-'.join(map(str,key))
            command, rate, holding, epoch = inputs[key]
            for label in labels:
                case_id = case+'/'+label
                if case_id in state['cases']:
                    row = state['cases'][case_id]
                    if row['state'] == 'retained' and digest(row['arrays']) != row['array_sha256']:
                        raise ValueError('retained reference arrays changed')
                    continue
                arrays_path = output/(case+'--'+label+'.npz')
                if arrays_path.exists():
                    raise ValueError('orphan prediction arrays retained; refusing overwrite')
                try:
                    if label.startswith('RK4-'):
                        factor = int(label.rsplit('-',1)[1])
                        v = compiled([parameter], np.repeat(command,factor), rate*factor,
                                     holding, 1.)[0][::factor]
                        receipt = {'method': 'compiled-RK4', 'diagnostic_refinement': factor}
                    else:
                        method, level_name = label.split('-')
                        level = next(r for r in reg['accuracy_levels'] if r['name'] == level_name)
                        v, receipt = adaptive_replay(parameter,command,rate,holding,method=method,
                            rtol=level['rtol'],voltage_atol=level['voltage_atol'],gate_atol=level['gate_atol'],
                            max_step_ms=reg['max_step_ms'],maximum_evaluations=reg['maximum_evaluations_per_replay'])
                    features = characterize(v, command, rate, epoch, holding)
                    np.savez_compressed(arrays_path,predicted_mV=v)
                    row = {'state': 'retained', 'features': features,
                           'score': score_sweep(targets[key],features), 'integration_receipt': receipt,
                           'arrays': str(arrays_path), 'array_sha256': digest(arrays_path)}
                except (ValueError, FloatingPointError, OverflowError, RuntimeError) as exc:
                    row = {'state': 'failed-retained-no-replacement', 'error': str(exc)}
                state['cases'][case_id] = row
                if any(digest(p) != h for p, h in pins.items()):
                    raise ValueError('source or predecessor changed during reference study')
                checkpoint(path,state)
                print('Reference diagnostic retained',len(state['cases']),'/',reg['case_count'],case_id,flush=True)
        summaries = {}
        for raw_key in reg['source_keys']:
            case = '-'.join(map(str,raw_key)); comparisons = {}
            pairs = [('DOP853-medium','DOP853-tight'),('Radau-medium','Radau-tight'),
                     ('DOP853-tight','Radau-tight')]
            pairs += [(label, m+'-tight') for label in labels if label.startswith('RK4-') for m in reg['methods']]
            for a, b in pairs:
                left,right = [state['cases'][case+'/'+x] for x in (a,b)]
                result = {'passed': False, 'replays_complete': left['state']==right['state']=='retained'}
                if result['replays_complete']:
                    with np.load(left['arrays'],allow_pickle=False) as x, np.load(right['arrays'],allow_pickle=False) as y:
                        error = float(np.max(np.abs(x['predicted_mV']-y['predicted_mV'])))
                    match = feature_agreement(left['features'],right['features'])
                    loss_match = abs(left['score']['loss']-right['score']['loss']) <= 1e-6+1e-8*abs(right['score']['loss'])
                    result.update(maximum_waveform_difference_mV=error,feature_agreement=match,
                                  loss_agreement=loss_match,passed=error<=1e-5 and match and loss_match)
                comparisons[a+'/'+b] = result
            complete = all(state['cases'][case+'/'+x]['state']=='retained' for x in labels if not x.startswith('RK4-'))
            qualified = complete and qualification(comparisons)
            summaries[case] = {'reference_qualified': qualified,'comparisons':comparisons,
                'classification': 'bounded-reference-convergence-observed-not-production-authorization' if qualified
                else 'reference-not-qualified-numerical-reliability-unresolved'}
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError('source changed before terminal publication')
        state.update(status='fixed-input-reference-convergence-diagnostic-retained',summaries=summaries,
                     production_integration_change_authorized=False,network_promotion=False,
                     frozen_baseline_modified=False)
        checkpoint(path,state)


if __name__ == '__main__':
    main()
