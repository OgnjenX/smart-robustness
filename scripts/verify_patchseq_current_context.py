"""Independent forward-run current context checks; no mapping collector import."""

from __future__ import annotations

import copy
import math
import re
from pathlib import Path

import numpy as np
import yaml
from verify_patchseq_contiguous_mapping import last_block
from verify_patchseq_setting_mapping import check, require
from verify_patchseq_whole_file_integrity import digest, same

REG = Path('docs/validation-results/post2008-patchseq-current-context-mapping-registration-1420.yaml')


def verify_context(table, data, pair, metadata, saved):
    columns = table['column_candidates']
    sweep = pair['sweep_number']
    if any(len(columns.get(name, [])) != 1 for name in ('SweepNum', 'EntrySourceType')):
        rows, reason = [], 'ambiguous-sweep-or-mode-column'
    else:
        ids = data[f'column-{columns["SweepNum"][0]:04d}'][:, 0]
        modes = data[f'column-{columns["EntrySourceType"][0]:04d}'][:, 0]
        rows = last_block(ids, modes, sweep)
        reason = 'no-explicit-acquisition-block-legacy-branch-not-ported'
    if not rows:
        same(saved, {'sweep_number': sweep, 'historical_range': {'undefined': reason},
                     'mapping': {'mapping_status': 'unknown', 'reason': reason},
                     'slow_injection': {}, 'holding_state': 'unknown', 'slow_state': 'unknown'})
        return False
    same(saved['historical_range'], {'selected_acquisition_rows': rows,
         'derived_lookup_view_only': True, 'raw_values_modified': False})
    def selected(label):
        matches = columns.get(label, [])
        if len(matches) != 1:
            return {'undefined': 'missing-or-ambiguous-column'}
        col = matches[0]
        array = data[f'column-{col:04d}']
        candidates = [row for row in rows if np.count_nonzero(~np.isnan(array[row]))]
        if not candidates:
            return {'undefined': 'no-non-NaN-vector-in-requested-mode'}
        return {'row': candidates[-1], 'column': col, 'vector': array[candidates[-1]]}
    def at(label, page):
        selected_value = selected(label)
        if 'undefined' in selected_value:
            return selected_value
        val = float(selected_value['vector'][page])
        answer = {'row': selected_value['row'], 'column': selected_value['column'], 'page': page}
        if math.isfinite(val):
            answer['value'] = val
        else:
            answer['undefined'] = 'nan' if math.isnan(val) else 'positive-infinity' if val > 0 else 'negative-infinity'
        return answer
    expected = {'sweep_number': sweep, 'mapping_status': 'unknown', 'fields': {}}
    response = re.fullmatch(r'/acquisition/data_[0-9]+_AD([0-9]+)', pair['response'])
    stimulus = re.fullmatch(r'/stimulus/presentation/data_[0-9]+_DA([0-9]+)', pair['stimulus'])
    channel_fields = ('ADC', 'DAC', 'Clamp Mode', 'Headstage Active')
    channels = {name: selected(name) for name in channel_fields}
    page = None
    if response is None or stimulus is None:
        expected['reason'] = 'unsupported-historical-channel-template'
    elif any('undefined' in item for item in channels.values()):
        expected.update(reason='unresolved-channel-setting', channel_setting_errors={
            name: item['undefined'] for name, item in channels.items() if 'undefined' in item})
    else:
        ad, da = int(response[1]), int(stimulus[1])
        candidates = [p for p in range(8) if (channels['ADC']['vector'][p], channels['DAC']['vector'][p],
                      channels['Clamp Mode']['vector'][p]) == (ad, da, 1)]
        expected.update(ad_channel=ad, da_channel=da, headstage_candidates=candidates,
                        channel_setting_rows={name: item['row'] for name, item in channels.items()})
        if len(candidates) != 1:
            expected['reason'] = 'nonunique-headstage'
        else:
            expected['headstage_page'] = candidates[0]
            if channels['Headstage Active']['vector'][candidates[0]] != 1:
                expected['reason'] = 'inactive-or-unknown-headstage-no-fixup'
            else:
                page = candidates[0]
    if page is None:
        same(saved['mapping'], expected)
        same(saved['slow_injection'], {})
        require(saved['holding_state'] == saved['slow_state'] == 'unknown')
        return False
    # Existing independent checker derives channels and every native/float32
    # export property from actual contiguous source-row slices, not a mask.
    local = copy.deepcopy(saved['mapping'])
    require(local['mapping_status'] == 'mapped')
    inverse = {row: index for index, row in enumerate(rows)}
    local['channel_setting_rows'] = {name: inverse[row] for name, row in local['channel_setting_rows'].items()}
    for entry in local['fields'].values():
        for name in ('native_value', 'enabled_state'):
            if 'row' in entry[name]:
                entry[name]['row'] = inverse[entry[name]['row']]
    proof = check(table, {key: values[rows] for key, values in data.items()}, pair, metadata, local)
    require(proof['mapped'])
    holding = saved['mapping']['fields']['bias_current']
    enabled = at('I-Clamp Holding Enable', page).get('value')
    holding_state = 'disabled' if enabled == 0 else 'enabled' if enabled == 1 else 'unknown'
    if holding['concordance'] != 'agrees':
        holding_state = 'unknown'
    same(saved['holding_state'], holding_state)
    slow = {}
    for label, unit in [('Slow current injection', 'On/Off'), ('Slow current injection level', 'V'),
                        ('Slow current injection settling time', 's')]:
        indices = columns.get(label, [])
        observed = table['column_headers'][indices[0]][1].get('decoded_text') if len(indices) == 1 else None
        slow[label] = {'native_value': at(label, page), 'expected_unit': unit,
                       'observed_unit': observed, 'unit_agrees': observed == unit}
    same(saved['slow_injection'], slow)
    enabled = slow['Slow current injection']
    val = enabled['native_value'].get('value') if enabled['unit_agrees'] else None
    same(saved['slow_state'], 'disabled' if val == 0 else 'enabled' if val == 1 else 'unknown')
    return True


def main():
    reg = yaml.safe_load(REG.read_text())
    result = Path(reg['output'])
    pins = {result: digest(result), Path(__file__): digest(__file__),
            Path('scripts/verify_patchseq_setting_mapping.py'): '7dd0fce9568e4325369df10de239efb1832be20048035b0fa1b6d88ca30e1295',
            Path('scripts/verify_patchseq_contiguous_mapping.py'): digest('scripts/verify_patchseq_contiguous_mapping.py')}
    report = yaml.load(result.read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(p): h for p, h in report['source_hashes'].items()})
    require(all(digest(p) == h for p, h in pins.items()))
    native = yaml.load(Path(reg['numeric']).read_text(), Loader=yaml.CSafeLoader)['records']
    native_index = {(r['source_batch'], r['slot']): r for r in native}
    saved_index = {(r['source_batch'], r['slot']): r for r in report['records']}
    require(len(native_index) == len(saved_index) == len(report['records']) == 96)
    summaries, total = [], 0
    for batch in reg['batches']:
        pairs = yaml.load(Path(batch['pairing']).read_text(), Loader=yaml.CSafeLoader)['records']
        measured = yaml.load(Path(batch['instrument_metadata']).read_text(), Loader=yaml.CSafeLoader)['records']
        require(len(pairs) == len(measured) == 48)
        for source, metrics in zip(pairs, measured, strict=True):
            require(source['slot'] == metrics['slot'] and source['source_cell_specimen_id'] == metrics['source_cell_specimen_id'])
            key = batch['name'], source['slot']
            case, saved = native_index[key], saved_index[key]
            require(saved['source_cell_specimen_id'] == source['source_cell_specimen_id'])
            require(len(case['tables']) == 1 and case['tables'][0]['error'] is None)
            table = case['tables'][0]
            require(digest(table['array_path']) == table['array_sha256'])
            with np.load(table['array_path'], allow_pickle=False) as archive:
                data = {name: archive[name] for name in archive.files}
            roles = {s['path']: s['role'] for s in source['series']}
            metadata = {s['sweep_number']: s['instrument_metadata'] for s in metrics['sweeps']}
            require(len(source['sweeps']) == len(saved['sweeps']) == len(metadata))
            mapped = 0
            for pair, outcome in zip(source['sweeps'], saved['sweeps'], strict=True):
                require(pair['sweep_number'] == outcome['sweep_number'])
                paths = {roles[p]: p for p in pair['series_paths']}
                mapped += verify_context(table, data, {'sweep_number': pair['sweep_number'], **paths},
                                         metadata[pair['sweep_number']], outcome)
            total += len(saved['sweeps'])
            summaries.append({'source_batch': batch['name'], 'slot': source['slot'],
                              'verified_pairs': len(saved['sweeps']), 'mapped': mapped})
    require(len({(r['source_batch'], r['slot']) for r in summaries}) == 96 and total == 4880)
    require(all(digest(p) == h for p, h in pins.items()))
    require(report['current_waveforms_modified'] is False and report['model_fitting_performed'] is False
            and report['specimen_eligibility_established'] is False)
    output = Path(reg['independent_output'])
    output.parent.mkdir()
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'all-4880-current-context-blocks-mappings-and-native-states-independently-verified',
                       'source_hashes': {str(p): h for p, h in pins.items()}, 'records': summaries,
                       'mapping_collector_imported': False, 'current_waveforms_modified': False,
                       'model_fitting_performed': False, 'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('All 4880 current-context mappings independently verified without waveform adjustment')


if __name__ == '__main__':
    main()
