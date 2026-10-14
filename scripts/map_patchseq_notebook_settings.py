"""Explicit supported notebook mode branch; no health score or fitting."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import yaml
from read_patchseq_notebook_numeric import last_explicit_mode_row
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-setting-mapping-registration-1348.yaml')
FIELDS = {'bias_current': ('I-Clamp Holding Level', 'I-Clamp Holding Enable', 'pA'),
          'bridge_balance': ('Bridge Bal Value', 'Bridge Bal Enable', 'MOhm'),
          'capacitance_compensation': ('Neut Cap Value', 'Neut Cap Enabled', 'pF')}


def field_vector(table, arrays, label, sweep):
    columns = table['column_candidates'].get(label, [])
    if len(columns) != 1:
        return {'undefined': 'missing-or-ambiguous-column'}
    identities = [table['column_candidates'].get(name, []) for name in ('SweepNum', 'EntrySourceType')]
    if any(len(c) != 1 for c in identities):
        return {'undefined': 'ambiguous-sweep-or-mode-column'}
    values = arrays[f'column-{columns[0]:04d}']
    sweeps, modes = (arrays[f'column-{c[0]:04d}'][:, 0] for c in identities)
    selected = last_explicit_mode_row(sweeps, modes, values, sweep, 0)
    if 'undefined' in selected:
        return selected
    return {'row': selected['row'], 'column': columns[0], 'vector': values[selected['row']]}


def value_at(selected, page):
    if 'undefined' in selected:
        return {'undefined': selected['undefined']}
    value = float(selected['vector'][page])
    result = {'row': selected['row'], 'column': selected['column'], 'page': page}
    if math.isfinite(value):
        result['value'] = value
    else:
        result['undefined'] = 'nan' if math.isnan(value) else 'positive-infinity' if value > 0 else 'negative-infinity'
    return result


def map_pair(table, arrays, sweep, response_path, stimulus_path, metadata):
    out = {'sweep_number': sweep, 'mapping_status': 'unknown', 'fields': {}}
    response = re.fullmatch(r'/acquisition/data_[0-9]+_AD([0-9]+)', response_path)
    stimulus = re.fullmatch(r'/stimulus/presentation/data_[0-9]+_DA([0-9]+)', stimulus_path)
    if response is None or stimulus is None:
        return {**out, 'reason': 'unsupported-historical-channel-template'}
    ad, da = int(response[1]), int(stimulus[1])
    names = ('ADC', 'DAC', 'Clamp Mode', 'Headstage Active')
    selected = {n: field_vector(table, arrays, n, sweep) for n in names}
    if any('undefined' in x for x in selected.values()):
        return {**out, 'reason': 'unresolved-channel-setting', 'channel_setting_errors': {
            n: s['undefined'] for n, s in selected.items() if 'undefined' in s}}
    candidates = [p for p in range(8) if selected['ADC']['vector'][p] == ad
                  and selected['DAC']['vector'][p] == da and selected['Clamp Mode']['vector'][p] == 1]
    out.update(ad_channel=ad, da_channel=da, headstage_candidates=candidates,
               channel_setting_rows={n: x['row'] for n, x in selected.items()})
    if len(candidates) != 1:
        return {**out, 'reason': 'nonunique-headstage'}
    page = candidates[0]
    out['headstage_page'] = page
    if selected['Headstage Active']['vector'][page] != 1:
        return {**out, 'reason': 'inactive-or-unknown-headstage-no-fixup'}
    out['mapping_status'] = 'mapped'
    for property_name, (label, enable_label, unit) in FIELDS.items():
        vector = field_vector(table, arrays, label, sweep)
        enable = field_vector(table, arrays, enable_label, sweep)
        native, enabled = value_at(vector, page), value_at(enable, page)
        exported = metadata[property_name]
        entry = {'native_value': native, 'enabled_state': enabled, 'exported_metadata': exported,
                 'native_unit': unit, 'concordance': 'unknown'}
        cols = table['column_candidates'].get(label, [])
        actual_unit = table['column_headers'][cols[0]][1].get('decoded_text') if len(cols) == 1 else None
        if actual_unit != unit:
            entry['reason'] = 'unsupported-native-unit'
        elif enabled.get('value') == 0:
            entry['concordance'] = 'agrees' if exported['present'] is False else 'discordant'
            entry['expected_export'] = 'absent-disabled-not-physical-zero'
        elif enabled.get('value') == 1 and 'value' in native:
            with np.errstate(over='ignore', invalid='ignore'):
                serialized = float(np.float32(native['value']))
            if math.isfinite(serialized):
                entry['expected_float32_value'] = serialized
                entry['concordance'] = 'agrees' if (exported.get('dtype') == 'float32'
                    and exported.get('value') == serialized) else 'discordant'
            else:
                entry['reason'] = 'nonfinite-float32-serialization'
        else:
            entry['reason'] = 'unresolved-enabled-or-finite-value'
        out['fields'][property_name] = entry
    return out


def main():
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: digest(REG), Path(__file__): digest(__file__)}
    for key in ('numeric', 'independent_numeric', 'pairing', 'instrument_metadata'):
        pins[Path(reg[key])] = reg[key+'_sha256']
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('registered inputs changed')
    independent = yaml.load(Path(reg['independent_numeric']).read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(p): h for p, h in independent['source_hashes'].items()})
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('source changed')
    numeric = yaml.load(Path(reg['numeric']).read_text(), Loader=yaml.CSafeLoader)['records']
    pairs = yaml.load(Path(reg['pairing']).read_text(), Loader=yaml.CSafeLoader)['records']
    qc = yaml.load(Path(reg['instrument_metadata']).read_text(), Loader=yaml.CSafeLoader)['records']
    if not len(numeric) == len(pairs) == len(qc) == 48:
        raise ValueError('fixed case scope differs')
    output = Path(reg['output'])
    output.parent.mkdir()
    records = []
    for case, pairing, measured in zip(numeric, pairs, qc, strict=True):
        if case['slot'] != pairing['slot'] or case['slot'] != measured['slot']:
            raise ValueError('case identity differs')
        if len(case['tables']) != 1 or case['tables'][0]['error'] is not None:
            raise ValueError('unresolved notebook device')
        table = case['tables'][0]
        by_path = {s['path']: s for s in pairing['series']}
        meta = {s['sweep_number']: s['instrument_metadata'] for s in measured['sweeps']}
        with np.load(table['array_path'], allow_pickle=False) as arrays:
            outcomes = []
            for pair in pairing['sweeps']:
                paths = {by_path[p]['role']: p for p in pair['series_paths']}
                outcomes.append(map_pair(table, arrays, pair['sweep_number'], paths['response'], paths['stimulus'], meta[pair['sweep_number']]))
        records.append({'slot': case['slot'], 'source_cell_specimen_id': pairing['source_cell_specimen_id'], 'sweeps': outcomes})
    if len({r['slot'] for r in records}) != 48 or any(digest(p) != h for p, h in pins.items()):
        raise ValueError('source or scope changed')
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'explicit-mode-setting-mapping-outcomes-retained-independent-check-pending',
            'source_hashes': {str(p): h for p, h in pins.items()}, 'records': records,
            'health_classification_performed': False, 'model_fitting_performed': False,
            'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('Retained all 2330 sweep/headstage and instrument-concordance outcomes')


if __name__ == '__main__':
    main()
