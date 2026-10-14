"""Qualify a documented nominal holding offset, never fill missing current."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-nominal-current-registration-1425.yaml')


def consistency(table, arrays, rows, page, label, expected):
    cols = table['column_candidates'].get(label, [])
    if len(cols) != 1:
        return {'consistent': False, 'reason': 'missing-or-ambiguous-source-column'}
    values = arrays[f'column-{cols[0]:04d}'][rows, page]
    observed = [(row, float(val)) for row, val in zip(rows, values, strict=True) if not np.isnan(val)]
    if not observed:
        return {'consistent': False, 'reason': 'no-observed-source-values'}
    bad = [row for row, val in observed if not math.isfinite(val) or val != expected]
    return {'consistent': not bad, 'observed_source_rows': [row for row, _ in observed],
            'conflicting_or_nonfinite_rows': bad}


def qualify(table, arrays, mapped):
    result = {'sweep_number': mapped['sweep_number'], 'nominal_context_supported': False,
              'nominal_holding_offset_pA': None, 'source_consistency': {}, 'failures': []}
    mapping = mapped['mapping']
    if mapping['mapping_status'] != 'mapped':
        return {**result, 'failures': ['unresolved-source-headstage-mapping']}
    holding = mapping['fields']['bias_current']
    if holding['concordance'] != 'agrees' or mapped['holding_state'] not in ('enabled', 'disabled'):
        return {**result, 'failures': ['unresolved-holding-state-or-export-concordance']}
    enable_columns = table['column_candidates'].get('I-Clamp Holding Enable', [])
    enable_unit = table['column_headers'][enable_columns[0]][1].get('decoded_text') if len(enable_columns) == 1 else None
    if holding['native_unit'] != 'pA' or enable_unit != 'On/Off':
        return {**result, 'failures': ['unsupported-native-holding-or-enable-unit']}
    slow = mapped['slow_injection'].get('Slow current injection', {})
    if mapped['slow_state'] != 'disabled' or slow.get('unit_agrees') is not True:
        return {**result, 'failures': ['enabled-or-unresolved-slow-injection']}
    rows, page = mapped['historical_range']['selected_acquisition_rows'], mapping['headstage_page']
    enabled = holding['enabled_state'].get('value')
    checks = [('I-Clamp Holding Enable', enabled), ('Slow current injection', 0.)]
    if enabled == 1:
        value = holding['native_value'].get('value')
        if type(value) not in (int, float) or not math.isfinite(value):
            return {**result, 'failures': ['missing-or-nonfinite-enabled-holding-level']}
        checks.append(('I-Clamp Holding Level', value))
        offset = value
    elif enabled == 0:
        offset = 0.
    else:
        return {**result, 'failures': ['nonbinary-holding-enable']}
    for label, expected in checks:
        proof = consistency(table, arrays, rows, page, label, expected)
        result['source_consistency'][label] = proof
        if not proof['consistent']:
            result['failures'].append('unsupported-source-consistency:'+label)
    if not result['failures']:
        result.update(nominal_context_supported=True, nominal_holding_offset_pA=float(offset))
    return result


def main():
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: digest(REG), Path(__file__): digest(__file__)}
    for key in ('mapping', 'independent_mapping', 'numeric'):
        pins[Path(reg[key])] = reg[key+'_sha256']
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('registered context source changed')
    verified = yaml.load(Path(reg['independent_mapping']).read_text(), Loader=yaml.CSafeLoader)
    if verified['status'] != 'all-4880-current-context-blocks-mappings-and-native-states-independently-verified':
        raise ValueError('mapping prerequisite incomplete')
    pins.update({Path(p): h for p, h in verified['source_hashes'].items()})
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('verified context changed')
    mapping = yaml.load(Path(reg['mapping']).read_text(), Loader=yaml.CSafeLoader)['records']
    native = yaml.load(Path(reg['numeric']).read_text(), Loader=yaml.CSafeLoader)['records']
    index = {(r['source_batch'], r['slot']): r for r in native}
    if len(mapping) != len(native) or len(index) != 96:
        raise ValueError('combined case scope differs')
    results, total = [], 0
    for case in mapping:
        numeric = index[(case['source_batch'], case['slot'])]
        if len(numeric['tables']) != 1 or numeric['tables'][0]['error'] is not None:
            raise ValueError('source table unresolved')
        table = numeric['tables'][0]
        if digest(table['array_path']) != table['array_sha256']:
            raise ValueError('native archive changed')
        with np.load(table['array_path'], allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        outcomes = [qualify(table, arrays, sweep) for sweep in case['sweeps']]
        results.append({'source_batch': case['source_batch'], 'slot': case['slot'],
                        'source_cell_specimen_id': case['source_cell_specimen_id'], 'sweeps': outcomes})
        total += len(outcomes)
    if len({(r['source_batch'], r['slot']) for r in results}) != 96 or total != 4880:
        raise ValueError('complete scope differs')
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('source changed during qualification')
    output = Path(reg['output'])
    output.parent.mkdir()
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'nominal-current-source-context-outcomes-retained-independent-check-pending',
                       'source_hashes': {str(p): h for p, h in pins.items()}, 'records': results,
                       'new_waveform_reads': 0, 'old_QC_or_archives_modified': False,
                       'model_fitting_performed': False, 'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('All nominal context qualifications retained; no source current or voltage replaced')


if __name__ == '__main__':
    main()
