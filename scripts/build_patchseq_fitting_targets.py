"""Fitting-only response characterization: held-out files are never opened."""

from __future__ import annotations

import os
from importlib.metadata import version
from pathlib import Path

import yaml
from patchseq_response_features import characterize
from run_patchseq_waveform_QC import read_array
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-fitting-target-registration-1429.yaml')


def plan(slot, pairs, wave, quality, nominal):
    if slot['partition'] not in ('fitting', 'held-out'):
        raise ValueError('unknown frozen source partition')
    if (slot['slot'] != pairs['slot'] or slot['slot'] != wave['slot'] or slot['slot'] != quality['slot']
        or slot['slot'] != nominal['slot'] or slot['source_cell_specimen_id'] != pairs['source_cell_specimen_id']
        or slot['source_cell_specimen_id'] != wave['source_cell_specimen_id']
        or slot['source_cell_specimen_id'] != nominal['source_cell_specimen_id']):
        raise ValueError('source case identity differs')
    qc = {s['sweep_number']: s for s in wave['sweeps']}
    context = {s['sweep_number']: s for s in nominal['sweeps']}
    numbers = [s['sweep_number'] for s in pairs['sweeps']]
    if len(numbers) != len(set(numbers)) or len(qc) != len(wave['sweeps']) or len(context) != len(nominal['sweeps']) or set(numbers) != set(qc) or set(numbers) != set(context):
        raise ValueError('sweep scope differs')
    if slot['partition'] == 'held-out':
        return [{'sweep_number': n, 'state': 'held-out-not-characterized'} for n in numbers]
    outputs = []
    for number in numbers:
        if quality['mandatory_gate'] != 'pass':
            state = 'recording-quality-not-passing'
        elif qc[number]['sweep_QC_pass'] is not True:
            state = 'waveform-QC-not-passing'
        elif context[number]['nominal_context_supported'] is not True:
            state = 'nominal-context-unsupported'
        else:
            state = 'eligible-fitting-response'
        outputs.append({'sweep_number': number, 'state': state})
    return outputs


def characterize_case(slot, pairs, wave, quality, nominal, reader):
    outputs = plan(slot, pairs, wave, quality, nominal)
    by_pair = {s['sweep_number']: s for s in pairs['sweeps']}
    by_wave = {s['sweep_number']: s for s in wave['sweeps']}
    by_context = {s['sweep_number']: s for s in nominal['sweeps']}
    for row in outputs:
        if row['state'] != 'eligible-fitting-response':
            continue
        n = row['sweep_number']
        try:
            row['features'] = reader(by_pair[n], by_wave[n], by_context[n])
            row['state'] = 'fitting-response-characterized'
        except (ValueError, TypeError, OSError) as exc:
            row.update(state='fitting-response-unresolved', error=str(exc))
    return outputs


def main():
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: digest(REG), Path(__file__): digest(__file__),
            Path('scripts/patchseq_response_features.py'): digest('scripts/patchseq_response_features.py'),
            Path('scripts/run_patchseq_waveform_QC.py'): '63617dbe160137c99094c502cf5abeed9940be8d8e85cd3b2649291cc475ea47'}
    for key in ('nominal', 'independent_nominal'):
        pins[Path(reg[key])] = reg[key+'_sha256']
    for batch in reg['batches']:
        for key in ('selection', 'pairing', 'waveform_QC', 'independent_waveform_QC', 'independent_quality', 'acquisition'):
            pins[Path(batch[key])] = batch[key+'_sha256']
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('fitting source changed')
    verified_nominal = yaml.load(Path(reg['independent_nominal']).read_text(), Loader=yaml.CSafeLoader)
    if verified_nominal['status'] != 'all-4880-nominal-offsets-source-consistency-and-unknowns-independently-reconstructed':
        raise ValueError('nominal prerequisite missing')
    pins.update({Path(p): h for p, h in verified_nominal['source_hashes'].items()})
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('verified source changed')
    if any(version(package) != reg['access'][key].split('==')[-1] for package, key in
           [('h5py', 'parser'), ('scipy', 'scipy'), ('numpy', 'numpy')]):
        raise ValueError('sealed dependency differs')
    nominal = yaml.load(Path(reg['nominal']).read_text(), Loader=yaml.CSafeLoader)['records']
    contexts = {(r['source_batch'], r['slot']): r for r in nominal}
    if len(contexts) != len(nominal) or len(nominal) != 96:
        raise ValueError('nominal combined scope differs')
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    import h5py
    output = Path(reg['output'])
    output.parent.mkdir()
    records, budget, opened = [], [0], []
    for batch in reg['batches']:
        data = {key: yaml.load(Path(batch[key]).read_text(), Loader=yaml.CSafeLoader)
                for key in ('selection', 'pairing', 'waveform_QC', 'independent_waveform_QC', 'independent_quality', 'acquisition')}
        if (data['independent_waveform_QC']['status'] != 'all-array-QC-outcomes-independently-reconstructed'
            or data['independent_quality']['status'] != 'all-recording-quality-arrays-measurements-and-mandatory-gates-independently-reconstructed'):
            raise ValueError('independent quality prerequisite missing')
        slots = data['selection']['slots']
        pairs = {r['slot']: r for r in data['pairing']['records']}
        waves = {r['slot']: r for r in data['waveform_QC']['records']}
        health = {r['slot']: r for r in data['independent_quality']['records']}
        receipts = {r['slot']: r for r in data['acquisition']['receipts']}
        if any(len(collection) != 48 for collection in (slots, pairs, waves, health, receipts)):
            raise ValueError('fixed source scope differs')
        for slot in slots:
            number = slot['slot']
            case_context = contexts[(batch['name'], number)]
            work = plan(slot, pairs[number], waves[number], health[number], case_context)
            entry = {key: slot[key] for key in ('slot', 'source_group', 'layer_label', 'partition',
                                               'frozen_group_id', 'source_cell_specimen_id')}
            entry['source_batch'] = batch['name']
            if any(row['state'] == 'eligible-fitting-response' for row in work):
                # Partition and quality gates precede the first HDF5 open.
                if slot['partition'] != 'fitting':
                    raise ValueError('held-out response access forbidden')
                item = receipts[number]
                if digest(item['path']) != item['sha256']:
                    raise ValueError('source receipt changed')
                receipt = yaml.safe_load(Path(item['path']).read_text())
                body = Path(receipt['complete_body_path'])
                before = digest(body)
                if before != slot['expected_SHA256'] or before != receipt['complete_file_SHA256']:
                    raise ValueError('recording changed')
                opened.append({'source_batch': batch['name'], 'slot': number, 'partition': slot['partition']})
                series = {s['path']: s for s in pairs[number]['series']}
                with h5py.File(body, 'r') as file:
                    def reader(pair, epoch, context, series=series, file=file):
                        if not pair['metadata_pair_pass']:
                            raise ValueError('pair not passing')
                        roles = {series[p]['role']: series[p] for p in pair['series_paths']}
                        values = {}
                        for role in ('response', 'stimulus'):
                            info = roles[role]
                            values[role] = read_array(file, info, h5py, budget, reg['access']['maximum_total_float64_bytes'])
                            values[role] *= info['conversion']*(1000 if role == 'response' else 1e12)
                        result = characterize(values['response'], values['stimulus'], roles['response']['rate'],
                                              epoch, context['nominal_holding_offset_pA'])
                        if (result['source_command_baseline_pA'] != epoch['command_baseline_pA']
                            or result['source_command_step_pA'] != epoch['command_step_pA']
                            or result['source_command_delta_pA'] != epoch['command_delta_pA']):
                            raise ValueError('recorded current and cached-QC disagreement')
                        return result
                    entry['sweeps'] = characterize_case(slot, pairs[number], waves[number], health[number], case_context, reader)
                if digest(body) != before:
                    raise ValueError('recording changed during training characterization')
            else:
                entry['sweeps'] = work
            records.append(entry)
    if len({(r['source_batch'], r['slot']) for r in records}) != 96 or any(digest(p) != h for p, h in pins.items()):
        raise ValueError('scope or source changed')
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'fitting-only-response-targets-retained-independent-check-pending',
                       'source_hashes': {str(p): h for p, h in pins.items()}, 'records': records,
                       'opened_response_files': opened, 'charged_float64_bytes': budget[0],
                       'held_out_response_features_read': False, 'model_fitting_performed': False,
                       'frozen_baseline_modified': False}, handle, sort_keys=False)
    print('Training-only response features retained; held-out response files never opened')


if __name__ == '__main__':
    main()
