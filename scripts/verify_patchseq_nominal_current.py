"""Independent source-row consistency and nominal-offset checks, no collector."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml
from verify_patchseq_whole_file_integrity import digest, same

REG = Path('docs/validation-results/post2008-patchseq-nominal-current-registration-1425.yaml')


def reconstruct(table, arrays, mapped):
    expected = {'sweep_number': mapped['sweep_number'], 'nominal_context_supported': False,
                'nominal_holding_offset_pA': None, 'source_consistency': {}, 'failures': []}
    core = mapped['mapping']
    if core['mapping_status'] != 'mapped':
        expected['failures'] = ['unresolved-source-headstage-mapping']
        return expected
    holding = core['fields']['bias_current']
    if holding['concordance'] != 'agrees' or mapped['holding_state'] not in ('enabled', 'disabled'):
        expected['failures'] = ['unresolved-holding-state-or-export-concordance']
        return expected
    indices = table['column_candidates'].get('I-Clamp Holding Enable', [])
    unit = table['column_headers'][indices[0]][1].get('decoded_text') if len(indices) == 1 else None
    if unit != 'On/Off' or holding['native_unit'] != 'pA':
        expected['failures'] = ['unsupported-native-holding-or-enable-unit']
        return expected
    slow = mapped['slow_injection'].get('Slow current injection', {})
    if mapped['slow_state'] != 'disabled' or slow.get('unit_agrees') is not True:
        expected['failures'] = ['enabled-or-unresolved-slow-injection']
        return expected
    enabled = holding['enabled_state'].get('value')
    requirements = [('I-Clamp Holding Enable', enabled), ('Slow current injection', 0.)]
    if enabled == 1:
        offset = holding['native_value'].get('value')
        if type(offset) not in (int, float) or not math.isfinite(offset):
            expected['failures'] = ['missing-or-nonfinite-enabled-holding-level']
            return expected
        requirements.append(('I-Clamp Holding Level', offset))
    elif enabled == 0:
        offset = 0.
    else:
        expected['failures'] = ['nonbinary-holding-enable']
        return expected
    page, rows = core['headstage_page'], mapped['historical_range']['selected_acquisition_rows']
    for label, target in requirements:
        columns = table['column_candidates'].get(label, [])
        if len(columns) != 1:
            proof = {'consistent': False, 'reason': 'missing-or-ambiguous-source-column'}
        else:
            array = arrays[f'column-{columns[0]:04d}']
            observations, bad = [], []
            for row in rows:
                value = float(array[row, page])
                if math.isnan(value):
                    continue  # Missing annotation, not an injected-current zero.
                observations.append(row)
                if not math.isfinite(value) or value != target:
                    bad.append(row)
            proof = ({'consistent': not bad, 'observed_source_rows': observations,
                      'conflicting_or_nonfinite_rows': bad} if observations
                     else {'consistent': False, 'reason': 'no-observed-source-values'})
        expected['source_consistency'][label] = proof
        if proof['consistent'] is not True:
            expected['failures'].append('unsupported-source-consistency:'+label)
    if not expected['failures']:
        expected.update(nominal_context_supported=True, nominal_holding_offset_pA=float(offset))
    return expected


def main():
    reg = yaml.safe_load(REG.read_text())
    result = Path(reg['output'])
    report = yaml.load(result.read_text(), Loader=yaml.CSafeLoader)
    pins = {result: digest(result), Path(__file__): digest(__file__)}
    pins.update({Path(p): h for p, h in report['source_hashes'].items()})
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('nominal context evidence changed')
    mapping = yaml.load(Path(reg['mapping']).read_text(), Loader=yaml.CSafeLoader)['records']
    numeric = yaml.load(Path(reg['numeric']).read_text(), Loader=yaml.CSafeLoader)['records']
    native = {(r['source_batch'], r['slot']): r for r in numeric}
    saved = {(r['source_batch'], r['slot']): r for r in report['records']}
    if len(mapping) != len(native) or len(saved) != len(report['records']) or len(saved) != 96:
        raise ValueError('case scope differs')
    summaries, total = [], 0
    for case in mapping:
        key = case['source_batch'], case['slot']
        outcome, source = saved[key], native[key]
        if outcome['source_cell_specimen_id'] != case['source_cell_specimen_id'] or len(source['tables']) != 1:
            raise ValueError('cell or native scope differs')
        table = source['tables'][0]
        if table['error'] is not None or digest(table['array_path']) != table['array_sha256']:
            raise ValueError('source archive changed or unresolved')
        with np.load(table['array_path'], allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        reconstructed = [reconstruct(table, arrays, sweep) for sweep in case['sweeps']]
        same(reconstructed, outcome['sweeps'])
        total += len(reconstructed)
        summaries.append({'source_batch': key[0], 'slot': key[1], 'verified_sweeps': len(reconstructed),
                          'supported_nominal_contexts': sum(r['nominal_context_supported'] for r in reconstructed)})
    if total != 4880 or len({(r['source_batch'], r['slot']) for r in summaries}) != 96:
        raise ValueError('complete sweep scope differs')
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('source changed during reconstruction')
    if report['new_waveform_reads'] != 0 or any(report[k] is not False for k in
        ('old_QC_or_archives_modified', 'model_fitting_performed', 'specimen_eligibility_established')):
        raise ValueError('claim scope differs')
    output = Path(reg['independent_output'])
    output.parent.mkdir()
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'all-4880-nominal-offsets-source-consistency-and-unknowns-independently-reconstructed',
                       'source_hashes': {str(p): h for p, h in pins.items()}, 'records': summaries,
                       'collector_imported': False, 'new_waveform_reads': 0,
                       'model_fitting_performed': False, 'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('All 4880 nominal offset/state-consistency outcomes independently reconstructed')


if __name__ == '__main__':
    main()
