"""Independent fitting command decoder; forbids held-out and response arrays."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from verify_patchseq_fitting_targets import read
from verify_patchseq_whole_file_integrity import digest


def metadata(path):
    return yaml.load(Path(path).read_text(), Loader=yaml.CSafeLoader)


def command_array(file, descriptor, h5py):
    if descriptor['role'] != 'stimulus':
        raise ValueError('independent fit reader forbids response decoding')
    return read(file, descriptor, h5py)


def inputs_for_training(records, source, nominal):
    # Check every record before resolving even the first receipt/body path.
    if not records or any(r['partition'] != 'fitting' for r in records):
        raise ValueError('independent reader forbids held-out or empty input')
    if len({(r['source_group'], r['layer_label']) for r in records}) != 1:
        raise ValueError('independent reader forbids class/layer pooling')
    if len({(r['source_batch'], r['slot']) for r in records}) != len(records):
        raise ValueError('duplicate source recording')
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    import h5py
    batches = {b['name']: b for b in source['batches']}
    contexts = {(r['source_batch'], r['slot']): r for r in nominal['records']}
    decoded, ledger, byte_count = {}, [], 0
    for record in records:
        wanted = {s['sweep_number']: s['features'] for s in record['sweeps']
                  if s['state'] == 'fitting-response-characterized'}
        if not wanted:
            continue
        batch, slot = record['source_batch'], record['slot']
        config = batches[batch]
        selection = next(r for r in metadata(config['selection'])['slots'] if r['slot'] == slot)
        if selection['partition'] != 'fitting' or any(selection[k] != record[k] for k in
            ('source_group', 'layer_label', 'frozen_group_id', 'source_cell_specimen_id')):
            raise ValueError('independent source identity differs')
        quality = next(r for r in metadata(config['independent_quality'])['records'] if r['slot'] == slot)
        if quality['mandatory_gate'] != 'pass':
            raise ValueError('independent recording quality gate not passing')
        pairing = next(r for r in metadata(config['pairing'])['records'] if r['slot'] == slot)
        waveform = next(r for r in metadata(config['waveform_QC'])['records'] if r['slot'] == slot)
        pairs = {s['sweep_number']: s for s in pairing['sweeps']}
        epochs = {s['sweep_number']: s for s in waveform['sweeps']}
        descriptors = {s['path']: s for s in pairing['series']}
        case = contexts[(batch, slot)]
        if case['source_cell_specimen_id'] != record['source_cell_specimen_id']:
            raise ValueError('independent nominal identity differs')
        currents = {s['sweep_number']: s for s in case['sweeps']}
        receipt_ref = next(r for r in metadata(config['acquisition'])['receipts'] if r['slot'] == slot)
        if digest(receipt_ref['path']) != receipt_ref['sha256']:
            raise ValueError('independent acquisition receipt changed')
        receipt = metadata(receipt_ref['path'])
        body = Path(receipt['complete_body_path'])
        before = digest(body)
        if before != selection['expected_SHA256'] or before != receipt['complete_file_SHA256']:
            raise ValueError('independent recording body changed')
        ledger.append({'source_batch': batch, 'slot': slot, 'partition': 'fitting'})
        with h5py.File(body, 'r') as file:
            for number, target in wanted.items():
                pair, epoch, current = pairs[number], epochs[number], currents[number]
                if not (pair['metadata_pair_pass'] and epoch['sweep_QC_pass']
                        and current['nominal_context_supported']):
                    raise ValueError('independent source support differs')
                roles = {descriptors[path]['role']: descriptors[path] for path in pair['series_paths']}
                if set(roles) != {'response', 'stimulus'} or len(pair['series_paths']) != 2:
                    raise ValueError('independent pair roles differ')
                if byte_count+8*roles['stimulus']['data_shape'][0] > 16*1024**3:
                    raise ValueError('independent command decoding budget exceeded')
                command, charge = command_array(file, roles['stimulus'], h5py)
                byte_count += charge
                rate, holding = roles['response']['rate'], current['nominal_holding_offset_pA']
                if (command[0] != target['source_command_baseline_pA']
                    or command[epoch['step_start_index']] != target['source_command_step_pA']
                    or holding != target['nominal_holding_offset_pA']
                    or (epoch['step_stop_index']-epoch['step_start_index'])/rate != target['step_duration_s']):
                    raise ValueError('independent command/catalog disagreement')
                key = (batch, slot, number)
                if key in decoded:
                    raise ValueError('duplicate independent command sweep')
                decoded[key] = (command, rate, holding, epoch)
        if digest(body) != before:
            raise ValueError('independent recording changed during read')
    return decoded, ledger, byte_count
