"""Checkpointed fitting-only screen; never opens held-out recording files."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from importlib.metadata import version
from pathlib import Path

import yaml
from patchseq_fit_objective import training_loss
from patchseq_fit_search import (
    candidate_dict,
    candidates,
    numerical_controls,
    replay,
    selected_candidate,
)
from patchseq_response_features import characterize
from run_patchseq_waveform_QC import read_array
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-isolated-fit-registration-1436.yaml')


def read_yaml(path):
    return yaml.load(Path(path).read_text(), Loader=yaml.CSafeLoader)


def fitting_records(catalog, stratum):
    """Partition filter must precede receipt/body resolution and first file open."""
    group, layer = stratum.split('-')
    return [r for r in catalog['records'] if r['partition'] == 'fitting'
            and (r['source_group'], r['layer_label']) == (group, layer)]


def prepare_inputs(records, source_reg, nominal):
    if not records or any(r['partition'] != 'fitting' for r in records):
        raise ValueError('held-out or empty source access forbidden')
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    import h5py
    batches = {b['name']: b for b in source_reg['batches']}
    inputs, opened, budget = {}, [], [0]
    for record in records:
        wanted = {s['sweep_number']: s['features'] for s in record['sweeps']
                  if s['state'] == 'fitting-response-characterized'}
        if not wanted:
            continue
        batch, slot = record['source_batch'], record['slot']
        sources = batches[batch]
        selection = next(s for s in read_yaml(sources['selection'])['slots'] if s['slot'] == slot)
        if (selection['partition'] != 'fitting' or any(selection[k] != record[k] for k in
            ('source_group', 'layer_label', 'frozen_group_id', 'source_cell_specimen_id'))):
            raise ValueError('frozen training identity differs')
        pair = next(s for s in read_yaml(sources['pairing'])['records'] if s['slot'] == slot)
        wave = next(s for s in read_yaml(sources['waveform_QC'])['records'] if s['slot'] == slot)
        pairs = {s['sweep_number']: s for s in pair['sweeps']}
        waves = {s['sweep_number']: s for s in wave['sweeps']}
        series = {s['path']: s for s in pair['series']}
        context = next(s for s in nominal['records'] if (s['source_batch'], s['slot']) == (batch, slot))
        contexts = {s['sweep_number']: s for s in context['sweeps']}
        item = next(s for s in read_yaml(sources['acquisition'])['receipts'] if s['slot'] == slot)
        if digest(item['path']) != item['sha256']:
            raise ValueError('source receipt changed')
        receipt = read_yaml(item['path'])
        body = Path(receipt['complete_body_path'])
        before = digest(body)
        if before != selection['expected_SHA256'] or before != receipt['complete_file_SHA256']:
            raise ValueError('source recording changed')
        opened.append({'source_batch': batch, 'slot': slot, 'partition': 'fitting'})
        with h5py.File(body, 'r') as file:
            for number, target in wanted.items():
                p, epoch, c = pairs[number], waves[number], contexts[number]
                if not (p['metadata_pair_pass'] and epoch['sweep_QC_pass']
                        and c['nominal_context_supported']):
                    raise ValueError('source support changed')
                roles = {series[path]['role']: series[path] for path in p['series_paths']}
                info = roles['stimulus']
                command = read_array(file, info, h5py, budget, 16*1024**3)
                command *= info['conversion']*1e12
                rate, holding = roles['response']['rate'], c['nominal_holding_offset_pA']
                if (command[0] != target['source_command_baseline_pA']
                    or command[epoch['step_start_index']] != target['source_command_step_pA']
                    or holding != target['nominal_holding_offset_pA']
                    or (epoch['step_stop_index']-epoch['step_start_index'])/rate != target['step_duration_s']):
                    raise ValueError('source command or epoch differs from verified catalog')
                key = (batch, slot, number)
                if key in inputs:
                    raise ValueError('duplicate source sweep')
                inputs[key] = (command, rate, holding, epoch)
        if digest(body) != before:
            raise ValueError('recording changed during command replay preparation')
    return inputs, opened, budget[0]


def checkpoint(path, state):
    temporary = path.with_suffix('.json.pending')
    with temporary.open('w') as handle:
        json.dump(state, handle, sort_keys=True, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_rows(rows, table):
    for key, row in rows.items():
        if (key not in {str(i) for i in range(len(table))}
            or row.get('parameters') != candidate_dict(table[int(key)])
            or row.get('state') != 'scored'):
            raise ValueError('checkpoint candidate identity differs')


def score_batch(parameters, records, inputs, warmup):
    predictions = [{} for _ in parameters]
    errors = [[] for _ in parameters]
    for key, (command, rate, holding, epoch) in inputs.items():
        # A failing candidate is isolated so it cannot poison its batch neighbors.
        try:
            voltage = replay(parameters, command, rate, holding, warmup)
            for index, v in enumerate(voltage):
                predictions[index][key] = characterize(v, command, rate, epoch, holding)
        except (ValueError, FloatingPointError, OverflowError) as batch_error:
            for index, p in enumerate(parameters):
                try:
                    v = replay([p], command, rate, holding, warmup)[0]
                    predictions[index][key] = characterize(v, command, rate, epoch, holding)
                except (ValueError, FloatingPointError, OverflowError) as exc:
                    predictions[index][key] = None
                    errors[index].append({'source_key': list(key), 'error': str(exc),
                                          'batch_error': str(batch_error)})
    rows = []
    for p, prediction, failures in zip(parameters, predictions, errors, strict=True):
        result = training_loss(records, prediction)
        rows.append({'state': 'scored', 'parameters': candidate_dict(p), 'loss': result['loss'],
                     'group_losses': result['group_losses'], 'simulation_failures': failures,
                     'all_training_observable_gates_pass': result['all_applicable_gates_pass']})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    reg = read_yaml(REG)
    if any(version(package) != expected for package, expected in reg['runtime'].items()):
        raise ValueError('sealed runtime differs')
    if Path(args.output) != Path(reg['output']):
        raise ValueError('output must be the sealed checkpoint directory')
    catalog, verified = read_yaml(reg['training_catalog']), read_yaml(reg['verified_catalog'])
    if (digest(reg['training_catalog']) != reg['training_catalog_sha256']
        or digest(reg['verified_catalog']) != reg['verified_catalog_sha256']
        or verified['status'] != 'all-training-response-features-and-held-out-no-access-boundaries-independently-reconstructed'):
        raise ValueError('verified training prerequisites changed')
    pins = {Path(p): h for p, h in verified['source_hashes'].items()}
    pins.update({REG: digest(REG), Path(__file__): digest(__file__)})
    for name in ('patchseq_isolated_hh', 'patchseq_fit_search', 'patchseq_fit_objective', 'patchseq_response_features'):
        p = Path('scripts')/(name+'.py'); pins[p] = digest(p)
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('sealed dependency changed')
    source_reg = read_yaml('docs/validation-results/post2008-patchseq-fitting-target-registration-1429.yaml')
    nominal = read_yaml(source_reg['nominal'])
    table = candidates(reg['search'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output/'checkpoint.json'
        sealed = {'source_hashes': {str(p): h for p, h in pins.items()},
                  'candidate_table': [candidate_dict(p) for p in table]}
        state = json.loads(path.read_text()) if path.exists() else {**sealed, 'strata': {}}
        if any(state.get(k) != v for k, v in sealed.items()):
            raise ValueError('checkpoint does not belong to sealed protocol')
        for stratum in reg['strata']:
            rows = state['strata'].setdefault(stratum, {'candidates': {}})
            validate_rows(rows['candidates'], table)
            records = fitting_records(catalog, stratum)
            if len(rows['candidates']) == len(table) and rows.get('numerical_controls_complete') is True:
                rows['selected_candidate'] = selected_candidate(
                    {int(i): r for i, r in rows['candidates'].items()}, len(table))
                checkpoint(path, state)
                continue
            inputs, opened, budget = prepare_inputs(records, source_reg, nominal)
            rows.update(opened_training_files=opened, decoded_command_float64_bytes=budget)
            width = reg['search']['candidate_batch_maximum']
            for start in range(0, len(table), width):
                ids = list(range(start, min(start+width, len(table))))
                if all(str(i) in rows['candidates'] for i in ids):
                    continue
                if any(str(i) in rows['candidates'] for i in ids):
                    raise ValueError('partial batch checkpoint not supported')
                results = score_batch([table[i] for i in ids], records, inputs, reg['replay']['warmup_s'])
                rows['candidates'].update({str(i): r for i, r in zip(ids, results, strict=True)})
                if any(digest(p) != h for p, h in pins.items()):
                    raise ValueError('source changed during fitting')
                checkpoint(path, state)
                print(stratum, 'retained candidates', len(rows['candidates']), '/', len(table), flush=True)
            chosen = selected_candidate({int(i): r for i, r in rows['candidates'].items()}, len(table))
            rows['selected_candidate'] = chosen
            checkpoint(path, state)
            controls = rows.setdefault('numerical_controls', {})
            if chosen is not None:
                for key, (command, rate, holding, epoch) in inputs.items():
                    case = json.dumps(key)
                    if case in controls:
                        continue
                    try:
                        controls[case] = numerical_controls(table[chosen], command, rate, holding,
                                                           epoch, reg['replay']['warmup_s'])
                    except (ValueError, FloatingPointError, OverflowError) as exc:
                        controls[case] = {'passed': False, 'error': str(exc)}
                    checkpoint(path, state)
            rows['numerical_controls_complete'] = len(controls) == len(inputs) if chosen is not None else False
            rows['all_numerical_controls_pass'] = rows['numerical_controls_complete'] and all(
                c['passed'] for c in controls.values())
            checkpoint(path, state)
        state.update(status='training-search-and-selected-numerical-controls-retained-independent-verification-pending',
                     held_out_response_features_read=False, network_promotion=False, frozen_baseline_modified=False)
        checkpoint(path, state)


if __name__ == '__main__':
    main()
