"""Native source context for all sweeps; never alter a command waveform."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from continue_patchseq_contiguous_mapping import lookup_view
from map_patchseq_notebook_settings import field_vector, map_pair, value_at
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-current-context-mapping-registration-1420.yaml')
SLOW = {'Slow current injection': 'On/Off', 'Slow current injection level': 'V',
        'Slow current injection settling time': 's'}


def context(table, data, sweep, response, stimulus, metadata):
    view, block = lookup_view(table, data, sweep)
    if view is None:
        return {'sweep_number': sweep, 'historical_range': block,
                'mapping': {'mapping_status': 'unknown', 'reason': block['undefined']},
                'slow_injection': {}, 'holding_state': 'unknown', 'slow_state': 'unknown'}
    mapped = map_pair(table, view, sweep, response, stimulus, metadata)
    slow, holding_state, slow_state = {}, 'unknown', 'unknown'
    if mapped['mapping_status'] == 'mapped':
        page = mapped['headstage_page']
        holding = mapped['fields']['bias_current']
        if holding['concordance'] == 'agrees':
            enabled = holding['enabled_state'].get('value')
            holding_state = 'disabled' if enabled == 0 else 'enabled' if enabled == 1 else 'unknown'
        for label, unit in SLOW.items():
            native = value_at(field_vector(table, view, label, sweep), page)
            columns = table['column_candidates'].get(label, [])
            observed = table['column_headers'][columns[0]][1].get('decoded_text') if len(columns) == 1 else None
            slow[label] = {'native_value': native, 'expected_unit': unit,
                           'observed_unit': observed, 'unit_agrees': observed == unit}
        enabled = slow['Slow current injection']
        if enabled['unit_agrees']:
            val = enabled['native_value'].get('value')
            slow_state = 'disabled' if val == 0 else 'enabled' if val == 1 else 'unknown'
    return {'sweep_number': sweep, 'historical_range': block, 'mapping': mapped,
            'slow_injection': slow, 'holding_state': holding_state, 'slow_state': slow_state}


def main():
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: digest(REG), Path(__file__): digest(__file__),
            Path('scripts/continue_patchseq_contiguous_mapping.py'): '085520d5859d30a5c794cce5647df490af1f6ddf1bc3c9e2a49e9395312580cd',
            Path('scripts/map_patchseq_notebook_settings.py'): '1af813bbc9a5bd8ced18cd0dcb15ed6c92b225182593cc721ecf551e5ecd67fd',
            Path('scripts/read_patchseq_notebook_numeric.py'): '5aa3f1610f74434f1e5c7db2c94f3c339eaad11e60141f68e78c7a8a58d4ba05'}
    for key in ('numeric', 'independent_numeric'):
        pins[Path(reg[key])] = reg[key+'_sha256']
    for batch in reg['batches']:
        for key in ('pairing', 'independent_pairing', 'instrument_metadata'):
            pins[Path(batch[key])] = batch[key+'_sha256']
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('current context source changed')
    independent = yaml.load(Path(reg['independent_numeric']).read_text(), Loader=yaml.CSafeLoader)
    if independent['status'] != 'all-96-native-current-context-columns-independently-bitwise-reconstructed':
        raise ValueError('numeric prerequisite incomplete')
    pins.update({Path(p): h for p, h in independent['source_hashes'].items()})
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('numeric evidence changed')
    native = yaml.load(Path(reg['numeric']).read_text(), Loader=yaml.CSafeLoader)['records']
    lookup = {(c['source_batch'], c['slot']): c for c in native}
    if len(native) != len(lookup) or len(native) != 96:
        raise ValueError('native case scope differs')
    results, total = [], 0
    for batch in reg['batches']:
        pairs = yaml.load(Path(batch['pairing']).read_text(), Loader=yaml.CSafeLoader)['records']
        measured = yaml.load(Path(batch['instrument_metadata']).read_text(), Loader=yaml.CSafeLoader)['records']
        if len(pairs) != len(measured) or len(pairs) != 48:
            raise ValueError('batch scope differs')
        for pair_case, measured_case in zip(pairs, measured, strict=True):
            if (pair_case['slot'] != measured_case['slot']
                or pair_case['source_cell_specimen_id'] != measured_case['source_cell_specimen_id']):
                raise ValueError('cell identity differs')
            case = lookup[(batch['name'], pair_case['slot'])]
            if len(case['tables']) != 1 or case['tables'][0]['error'] is not None:
                raise ValueError('numeric table unresolved')
            table = case['tables'][0]
            if digest(table['array_path']) != table['array_sha256']:
                raise ValueError('native archive changed')
            roles = {s['path']: s['role'] for s in pair_case['series']}
            metadata = {s['sweep_number']: s['instrument_metadata'] for s in measured_case['sweeps']}
            if len(metadata) != len(pair_case['sweeps']):
                raise ValueError('instrument scope differs')
            with np.load(table['array_path'], allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
            outcomes = []
            for pair in pair_case['sweeps']:
                paths = {roles[p]: p for p in pair['series_paths']}
                outcomes.append(context(table, data, pair['sweep_number'], paths['response'],
                                        paths['stimulus'], metadata[pair['sweep_number']]))
            total += len(outcomes)
            results.append({'source_batch': batch['name'], 'slot': pair_case['slot'],
                            'source_cell_specimen_id': pair_case['source_cell_specimen_id'], 'sweeps': outcomes})
    if total != 4880 or len({(r['source_batch'], r['slot']) for r in results}) != 96:
        raise ValueError('complete sweep scope differs')
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('source changed during mapping')
    output = Path(reg['output'])
    output.parent.mkdir()
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'all-current-context-mapping-outcomes-retained-independent-check-pending',
                       'source_hashes': {str(p): h for p, h in pins.items()}, 'records': results,
                       'current_waveforms_modified': False, 'model_fitting_performed': False,
                       'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('All 4880 source current-context mappings retained; no command changes or fitting')


if __name__ == '__main__':
    main()
