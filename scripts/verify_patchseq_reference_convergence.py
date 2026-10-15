"""Independent terminal measurements/accounting for fixed reference predictions."""

from __future__ import annotations

import fcntl
import json
import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml
from patchseq_independent_fit_loss import observable_loss
from run_patchseq_isolated_fit import fitting_records, prepare_inputs, read_yaml
from verify_patchseq_fitting_targets import reconstruct
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-reference-audit-registration-1485.yaml')
TERMINAL = 'fixed-input-reference-convergence-diagnostic-retained'


def agreement(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(agreement(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(agreement(x, y) for x, y in zip(a, b, strict=True))
    if type(a) is float:
        return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-5+1e-9*abs(a)
    return a == b


def close(a, b):
    return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-6+1e-8*abs(b)


@contextmanager
def terminal(directory):
    directory = Path(directory)
    path = directory/'checkpoint.json'
    with (directory/'runner.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        before = digest(path)
        state = json.loads(path.read_text())
        if state.get('status') != TERMINAL:
            raise ValueError('reference study is not terminal')
        yield state, before
        if digest(path) != before:
            raise ValueError('terminal reference snapshot changed')


def receipt_matches(receipt, label, command, rate, holding, study):
    if label.startswith('RK4-'):
        return receipt == {'method': 'compiled-RK4', 'diagnostic_refinement': int(label.rsplit('-',1)[1])}
    method, level = label.split('-')
    accuracy = next(r for r in study['accuracy_levels'] if r['name'] == level)
    if (receipt['method'] != method or receipt['rtol'] != accuracy['rtol']
        or receipt['voltage_atol_mV'] != accuracy['voltage_atol']
        or receipt['gate_atol'] != accuracy['gate_atol'] or receipt['max_step_ms'] != study['max_step_ms']
        or not 0 < receipt['function_evaluations'] <= study['maximum_evaluations_per_replay']):
        return False
    nominal = command+holding
    warm = round(rate*study['input_contract']['warmup_s'])
    current = np.r_[np.full(warm,nominal[0]),nominal]
    changes = np.flatnonzero(np.diff(current)!=0)+1
    boundaries = [0]+[int(i) for i in changes if i < len(current)-1]+[len(current)-1]
    rows = receipt['segments']
    if len(rows) != len(boundaries)-1:
        return False
    evaluations = 0
    for i, row in enumerate(rows):
        lo, hi = boundaries[i:i+2]
        if (row['first_index'] != lo or row['last_index'] != hi
            or row['constant_current_pA'] != float(current[lo])
            or type(row['nfev']) is not int or row['nfev'] <= 0):
            return False
        evaluations += row['nfev']
    # Radau's finite-difference Jacobian can require extra RHS calls.
    return evaluations <= receipt['function_evaluations']


def compare(a, b):
    if a is None or b is None:
        return {'passed': False, 'replays_complete': False}
    error = float(np.max(np.abs(a['voltage']-b['voltage'])))
    features = agreement(a['features'],b['features'])
    loss = close(a['loss'],b['loss'])
    return {'passed': error<=1e-5 and features and loss,'replays_complete': True,
            'maximum_waveform_difference_mV': error,'feature_agreement': features,'loss_agreement':loss}


def reference_qualified(complete, comparisons):
    required = ('DOP853-medium/DOP853-tight','Radau-medium/Radau-tight','DOP853-tight/Radau-tight')
    return complete and all(name in comparisons and comparisons[name]['passed'] is True for name in required)


def main():
    reg = read_yaml(REG);study = read_yaml(reg['study'])
    output = Path(study['output'])
    pins = {str(REG):digest(REG),reg['study']:digest(reg['study']),str(Path(__file__)):digest(__file__)}
    for name in ('patchseq_independent_fit_loss.py','run_patchseq_isolated_fit.py',
                 'verify_patchseq_fitting_targets.py'):
        p=Path('scripts')/name;pins[str(p)]=digest(p)
    with terminal(output) as (state, sha):
        if any(digest(p)!=h for p,h in state['source_hashes'].items()):
            raise ValueError('reference source or binary hash differs')
        pins.update(state['source_hashes'])
        if digest(study['original_checkpoint'])!=study['original_sha256']:
            raise ValueError('original checkpoint changed')
        original=json.loads(Path(study['original_checkpoint']).read_text())
        expected_parameter=original['strata'][study['stratum']]['candidates'][str(study['candidate_index'])]['parameters']
        if state['parameters']!=expected_parameter:
            raise ValueError('reference parameter vector differs')
        build=state['compiled_build']
        if read_yaml(output/'build'/'build.yaml')!=build or digest(build['binary_path'])!=build['binary_sha256']:
            raise ValueError('compiled build receipt differs')
        protocol=read_yaml('docs/validation-results/post2008-patchseq-isolated-fit-registration-1436.yaml')
        records=fitting_records(read_yaml(protocol['training_catalog']),study['stratum'])
        source=read_yaml('docs/validation-results/post2008-patchseq-fitting-target-registration-1429.yaml')
        inputs, ledger, charged=prepare_inputs(records,source,read_yaml(source['nominal']))
        if ledger!=state['opened_training_command_files'] or charged!=state['decoded_command_float64_bytes']:
            raise ValueError('independent command accounting differs')
        targets={(r['source_batch'],r['slot'],s['sweep_number']):s['features']
                 for r in records for s in r['sweeps'] if s['state']=='fitting-response-characterized'}
        adaptive=[m+'-'+a['name'] for m in study['methods'] for a in study['accuracy_levels']]
        labels=adaptive+['RK4-refinement-'+str(i) for i in study['compiled_RK4_diagnostic_refinements']]
        expected={'-'.join(map(str,k))+'/'+l for k in study['source_keys'] for l in labels}
        if set(state['cases'])!=expected or len(expected)!=study['case_count']:
            raise ValueError('reference fixed case scope incomplete or expanded')
        cases={};summaries={};array_hashes={};raw_all_agree=True
        for raw_key in study['source_keys']:
            key=tuple(raw_key);case='-'.join(map(str,key));command,rate,holding,epoch=inputs[key]
            data={}
            for label in labels:
                case_id=case+'/'+label;row=state['cases'][case_id]
                if row['state']=='failed-retained-no-replacement':
                    if not isinstance(row.get('error'),str):raise ValueError('missing retained failure reason')
                    cases[case_id]={'state':row['state'],'error':row['error']};data[label]=None
                    continue
                if row['state']!='retained':raise ValueError('unknown case state')
                path=output/(case+'--'+label+'.npz')
                if Path(row['arrays'])!=path or digest(path)!=row['array_sha256']:
                    raise ValueError('prediction array scope or digest differs')
                with np.load(path,allow_pickle=False) as archive:
                    if archive.files!=['predicted_mV']:raise ValueError('unexpected array members')
                    voltage=archive['predicted_mV'].copy()
                if voltage.shape!=command.shape or voltage.dtype!=np.float64 or not np.isfinite(voltage).all():
                    raise ValueError('invalid prediction grid/dtype/voltage')
                features=reconstruct(voltage,command,rate,epoch,holding)
                score=observable_loss(targets[key],features)
                checks={'features':agreement(features,row['features']),
                        'loss':close(score['loss'],row['score']['loss']),
                        'score_gates':score['gates']==row['score']['gates'],
                        'integration_receipt':receipt_matches(row['integration_receipt'],label,command,rate,holding,study)}
                raw_all_agree=raw_all_agree and all(checks.values())
                cases[case_id]={'state':'retained','checks':checks,'independent_features':features,
                                'independent_score':score}
                array_hashes[str(path)]=row['array_sha256']
                data[label]={'voltage':voltage,'features':features,'loss':score['loss']}
            pairs=[('DOP853-medium','DOP853-tight'),('Radau-medium','Radau-tight'),('DOP853-tight','Radau-tight')]
            pairs +=[(l,m+'-tight') for l in labels if l.startswith('RK4-') for m in study['methods']]
            comparisons={a+'/'+b:compare(data[a],data[b]) for a,b in pairs}
            complete=all(data[l] is not None for l in adaptive)
            qualified=reference_qualified(complete,comparisons)
            saved=state['summaries'][case]
            same=agreement(comparisons,saved['comparisons']) and saved['reference_qualified']==qualified
            raw_all_agree=raw_all_agree and same
            summaries[case]={'all_adaptive_cases_complete':complete,'reference_qualified':qualified,
                             'comparisons':comparisons,'stored_summary_agrees':same}
        if set(state['summaries'])!=set(summaries):raise ValueError('summary scope differs')
        if any(digest(p)!=h for p,h in state['source_hashes'].items()):raise ValueError('source changed during audit')
        report={'status':'independent-reference-array-feature-score-and-receipt-audit-retained',
                'checkpoint_sha256':sha,'source_hashes':pins,'prediction_array_sha256':array_hashes,
                'all_independent_checks_agree':raw_all_agree,'cases':cases,'summaries':summaries,
                'solver_trajectories_resimulated':False,'internal_gate_history_independently_observed':False,
                'production_change_authorized':False,'fitting_resumed':False,
                'held_out_response_features_read':False,'scientific_goal_complete':False}
    # Publish only after terminal snapshot stability is confirmed on lock exit.
    destination=Path(reg['output']);destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x') as handle:yaml.safe_dump(report,handle,sort_keys=False)
    print('Independent terminal reference audit retained; agreement:',raw_all_agree,flush=True)


if __name__=='__main__':
    main()
