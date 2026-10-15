"""Fixed-candidate diagnostic replays; no fitting, repairs or held-out data access."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path

import numpy as np
from patchseq_fit_objective import score_sweep, training_loss
from patchseq_fit_search import replay as brian_replay
from patchseq_hh_compiled_driver import compiled_driver
from patchseq_independent_fit_loss import observable_loss, reconstruct_training_loss
from patchseq_isolated_hh import Parameters
from patchseq_response_features import characterize
from patchseq_selected_fit_agreement import feature_agreement
from run_patchseq_isolated_fit import checkpoint, fitting_records, prepare_inputs, read_yaml
from verify_patchseq_fitting_targets import reconstruct
from verify_patchseq_isolated_fit_replay import close
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-mismatch-localization-registration-1478.yaml')


def differing_fields(a, b):
    return {k: {'primary': a[k], 'independent': b[k]} for k in a
            if not feature_agreement(a[k], b[k])}


def main():
    reg = read_yaml(REG)
    for name in ('original', 'failed'):
        if digest(reg[name+'_checkpoint']) != reg[name+'_sha256']:
            raise ValueError('diagnostic input snapshot changed')
    original = json.loads(Path(reg['original_checkpoint']).read_text())
    failed = json.loads(Path(reg['failed_checkpoint']).read_text())
    pins = dict(original['source_hashes'])
    pins.update(failed['source_hashes'])
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('original or failed-transition source changed')
    names = ('patchseq_fit_objective.py', 'patchseq_fit_search.py',
             'patchseq_hh_compiled_driver.py', 'patchseq_independent_fit_loss.py',
             'patchseq_response_features.py', 'patchseq_selected_fit_agreement.py',
             'verify_patchseq_fitting_targets.py')
    for name in names:
        p = Path('scripts')/name
        pins[str(p)] = digest(p)
    pins.update({str(REG): digest(REG), str(Path(__file__)): digest(__file__),
                 reg['original_checkpoint']: reg['original_sha256'],
                 reg['failed_checkpoint']: reg['failed_sha256']})
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('diagnostic source changed')
    protocol = read_yaml(reg['protocol'])
    catalog = read_yaml(protocol['training_catalog'])
    records = fitting_records(catalog, reg['stratum'])
    row = original['strata'][reg['stratum']]['candidates'][str(reg['candidate_index'])]
    source = read_yaml('docs/validation-results/post2008-patchseq-fitting-target-registration-1429.yaml')
    nominal = read_yaml(source['nominal'])
    output = Path(reg['output'])
    output.mkdir(parents=True, exist_ok=True)
    with (output/'runner.lock').open('a') as lock, (Path(reg['original_checkpoint']).parent/'runner.lock').open('r') as old, (Path(reg['failed_checkpoint']).parent/'runner.lock').open('r') as amended:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(old, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(amended, fcntl.LOCK_SH | fcntl.LOCK_NB)
        path = output/'checkpoint.json'
        if path.exists():
            raise ValueError('diagnostic evidence already exists; inspect it rather than overwrite')
        inputs, ledger, charged = prepare_inputs(records, source, nominal)
        targets = {(r['source_batch'], r['slot'], s['sweep_number']): s['features']
                   for r in records for s in r['sweeps'] if s['state'] == 'fitting-response-characterized'}
        state = {'source_hashes': pins, 'candidate_index': reg['candidate_index'],
                 'stratum': reg['stratum'], 'original_loss': row['loss'],
                 'opened_training_files': ledger, 'decoded_command_float64_bytes': charged,
                 'stage_A': {}, 'held_out_response_features_read': False,
                 'fitting_resumed': False, 'source_modified': False}
        primary, independent = {}, {}
        with compiled_driver(output/'build') as (compiled, build):
            state['compiled_build'] = build
            pins[build['binary_path']] = build['binary_sha256']
            pins[str(output/'build'/'build.yaml')] = digest(output/'build'/'build.yaml')
            for key, (command, rate, holding, epoch) in inputs.items():
                v = compiled([row['parameters']], command, rate, holding, protocol['replay']['warmup_s'])[0]
                a, b = characterize(v, command, rate, epoch, holding), reconstruct(v, command, rate, epoch, holding)
                primary[key], independent[key] = a, b
                state['stage_A'][json.dumps(key)] = {
                    'differing_features': differing_fields(a, b),
                    'primary_features': a, 'independent_features': b,
                    'primary_scorer_primary_features': score_sweep(targets[key], a),
                    'independent_scorer_primary_features': observable_loss(targets[key], a),
                    'primary_scorer_independent_features': score_sweep(targets[key], b),
                    'independent_scorer_independent_features': observable_loss(targets[key], b)}
                checkpoint(path, state)
            state['stage_A_aggregate'] = {}
            for label, predictions in (('primary_features', primary), ('independent_features', independent)):
                for name, scorer in (('primary_scorer', training_loss), ('independent_scorer', reconstruct_training_loss)):
                    result = scorer(records, predictions)
                    state['stage_A_aggregate'][label+'/'+name] = {
                        'loss': result['loss'], 'group_losses': result['group_losses'],
                        'agrees_with_original_loss': close(result['loss'], row['loss'])}
            checkpoint(path, state)
            print('Stage A retained: all fixed candidate training sweeps characterized both ways', flush=True)
            key = min(inputs)
            command, rate, holding, epoch = inputs[key]
            try:
                c = compiled([row['parameters']], command, rate, holding, protocol['replay']['warmup_s'])[0]
                b = brian_replay([Parameters(**row['parameters'])], command, rate, holding, protocol['replay']['warmup_s'])[0]
                error = np.abs(c-b)
                outside = np.flatnonzero(error > 1e-5)
                bf = characterize(b, command, rate, epoch, holding)
                cf = characterize(c, command, rate, epoch, holding)
                arrays = output/'stage-B-predictions.npz'
                np.savez_compressed(arrays, brian_mV=b, compiled_mV=c, command_pA=command)
                state['stage_B'] = {'source_key': list(key), 'rate_hz': rate,
                    'maximum_waveform_difference_mV': float(np.max(error)),
                    'first_sample_outside_tolerance': int(outside[0]) if len(outside) else None,
                    'brian_features': bf, 'compiled_features': cf,
                    'brian_extractor_disagreement': differing_fields(bf, reconstruct(b, command, rate, epoch, holding)),
                    'cross_engine_differing_features': differing_fields(bf, cf),
                    'brian_primary_loss': score_sweep(targets[key], bf),
                    'brian_independent_loss': observable_loss(targets[key], reconstruct(b, command, rate, epoch, holding)),
                    'compiled_primary_loss': score_sweep(targets[key], cf),
                    'compiled_independent_loss': observable_loss(targets[key], reconstruct(c, command, rate, epoch, holding)),
                    'prediction_arrays_sha256': digest(arrays)}
            except (ValueError, FloatingPointError, OverflowError) as exc:
                state['stage_B'] = {'source_key': list(key), 'error': str(exc), 'replacement_sweep_used': False}
            if any(digest(p) != h for p, h in pins.items()):
                raise ValueError('source or evidence changed during diagnostic')
            state.update(status='fixed-candidate-mismatch-diagnostic-retained-not-a-fitting-result',
                         tolerance_changed=False, frozen_baseline_modified=False)
            checkpoint(path, state)
            print('Fixed diagnostic complete; fitters remain stopped', flush=True)


if __name__ == '__main__':
    main()
