"""Independent verification of the observed strict branch and mode unknowns.

This checker supports the actual mode-unknown or uniquely mapped outcomes; other
branches require their own assessment. It imports no mapping collector/selector.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import yaml
from verify_patchseq_whole_file_integrity import digest

RESULT = Path('results/patchseq-setting-mapping-1349/manifest.yaml')
OUTPUT = Path('results/patchseq-setting-mapping-independent-1351/manifest.yaml')
FIELDS = {'bias_current': ('I-Clamp Holding Level', 'I-Clamp Holding Enable', 'pA'),
          'bridge_balance': ('Bridge Bal Value', 'Bridge Bal Enable', 'MOhm'),
          'capacitance_compensation': ('Neut Cap Value', 'Neut Cap Enabled', 'pF')}


def require(condition):
    if not condition:
        raise ValueError('strict mapping or its independent evidence differs')


def check(table, data, pair, metadata, saved):
    sweep = pair['sweep_number']
    columns = table['column_candidates']
    require(all(len(columns[label]) == 1 for label in ('SweepNum', 'EntrySourceType')))
    ids = data[f'column-{columns["SweepNum"][0]:04d}'][:, 0]
    modes = data[f'column-{columns["EntrySourceType"][0]:04d}'][:, 0]
    rows = [k for k, value in enumerate(ids) if value == sweep]
    channel_fields = ('ADC', 'DAC', 'Clamp Mode', 'Headstage Active')
    if rows and any(not math.isfinite(modes[k]) or modes[k] not in (0, 1, 2) for k in rows):
        require(saved == {'sweep_number': sweep, 'mapping_status': 'unknown', 'fields': {},
            'reason': 'unresolved-channel-setting', 'channel_setting_errors': {
                n: 'legacy-or-missing-source-mode' for n in channel_fields}})
        return {'mapped': False, 'explicit_acquisition_rows_exist': any(modes[k] == 0 for k in rows)}
    require(bool(rows))
    def select(label):
        require(len(columns[label]) == 1)
        col = columns[label][0]
        values = data[f'column-{col:04d}']
        for row in reversed(rows):
            if modes[row] == 0 and np.count_nonzero(~np.isnan(values[row])):
                return {'row': row, 'column': col, 'vector': values[row]}
        return {'undefined': 'no-non-NaN-vector-in-requested-mode'}
    def at(label, page):
        item = select(label)
        if 'undefined' in item:
            return item
        val = float(item['vector'][page])
        answer = {'row': item['row'], 'column': item['column'], 'page': page}
        if math.isfinite(val):
            answer['value'] = val
        else:
            answer['undefined'] = 'nan' if math.isnan(val) else 'positive-infinity' if val > 0 else 'negative-infinity'
        return answer
    response = re.fullmatch(r'/acquisition/data_[0-9]+_AD([0-9]+)', pair['response'])
    stimulus = re.fullmatch(r'/stimulus/presentation/data_[0-9]+_DA([0-9]+)', pair['stimulus'])
    require(response is not None and stimulus is not None)
    ad, da = int(response[1]), int(stimulus[1])
    channels = {name: select(name) for name in channel_fields}
    require(all('vector' in v for v in channels.values()))
    candidates = [p for p in range(8) if (channels['ADC']['vector'][p], channels['DAC']['vector'][p],
                    channels['Clamp Mode']['vector'][p]) == (ad, da, 1)]
    require(len(candidates) == 1)
    page = candidates[0]
    require(channels['Headstage Active']['vector'][page] == 1)
    expected = {'sweep_number': sweep, 'mapping_status': 'mapped', 'fields': {},
        'ad_channel': ad, 'da_channel': da, 'headstage_candidates': candidates,
        'channel_setting_rows': {name: item['row'] for name, item in channels.items()}, 'headstage_page': page}
    for prop, (label, enabled_label, unit) in FIELDS.items():
        native, enabled, exported = at(label, page), at(enabled_label, page), metadata[prop]
        require(table['column_headers'][columns[label][0]][1]['decoded_text'] == unit)
        entry = {'native_value': native, 'enabled_state': enabled, 'exported_metadata': exported,
                 'native_unit': unit, 'concordance': 'unknown'}
        if enabled.get('value') == 0:
            entry.update(concordance='agrees' if exported['present'] is False else 'discordant',
                         expected_export='absent-disabled-not-physical-zero')
        elif enabled.get('value') == 1 and 'value' in native:
            with np.errstate(over='ignore', invalid='ignore'):
                rounded = float(np.float32(native['value']))
            if math.isfinite(rounded):
                entry['expected_float32_value'] = rounded
                entry['concordance'] = 'agrees' if exported.get('dtype') == 'float32' and exported.get('value') == rounded else 'discordant'
            else:
                entry['reason'] = 'nonfinite-float32-serialization'
        else:
            entry['reason'] = 'unresolved-enabled-or-finite-value'
        expected['fields'][prop] = entry
    require(expected == saved)
    return {'mapped': True, 'explicit_acquisition_rows_exist': any(modes[k] == 0 for k in rows)}


def main():
    pins = {RESULT: '6983192061afe2e6af5da377ea96b83ea28f18f99cc8f85530fa6003719688ff'}
    require(digest(RESULT) == pins[RESULT])
    report = yaml.load(RESULT.read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(p): h for p, h in report['source_hashes'].items()})
    require(all(digest(p) == h for p, h in pins.items()))
    numeric = yaml.load(Path('results/patchseq-notebook-numeric-1345/manifest.yaml').read_text(), Loader=yaml.CSafeLoader)['records']
    pairing = yaml.load(Path('results/patchseq-pairing-1324/manifest.yaml').read_text(), Loader=yaml.CSafeLoader)['records']
    qc = yaml.load(Path('results/patchseq-waveform-QC-1328/manifest.yaml').read_text(), Loader=yaml.CSafeLoader)['records']
    require(len(numeric) == len(pairing) == len(qc) == len(report['records']) == 48)
    summaries = []
    for case, pairs, measured, saved in zip(numeric, pairing, qc, report['records'], strict=True):
        require(case['slot'] == pairs['slot'] == measured['slot'] == saved['slot'])
        require(saved['source_cell_specimen_id'] == pairs['source_cell_specimen_id'])
        require(len(case['tables']) == 1 and case['tables'][0]['error'] is None)
        table = case['tables'][0]
        by_path = {s['path']: s['role'] for s in pairs['series']}
        metadata = {s['sweep_number']: s['instrument_metadata'] for s in measured['sweeps']}
        require(len(pairs['sweeps']) == len(saved['sweeps']))
        outcomes = []
        with np.load(table['array_path'], allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        for pair, outcome in zip(pairs['sweeps'], saved['sweeps'], strict=True):
            paths = {by_path[path]: path for path in pair['series_paths']}
            outcomes.append(check(table, data, {'sweep_number': pair['sweep_number'], **paths},
                                  metadata[pair['sweep_number']], outcome))
        summaries.append({'slot': case['slot'], 'verified_pairs': len(outcomes),
            'mapped': sum(x['mapped'] for x in outcomes), 'unknown': sum(not x['mapped'] for x in outcomes),
            'pairs_with_explicit_acquisition_rows': sum(x['explicit_acquisition_rows_exist'] for x in outcomes)})
    require(len({s['slot'] for s in summaries}) == 48 and all(digest(p) == h for p, h in pins.items()))
    require(report['health_classification_performed'] is False and report['model_fitting_performed'] is False
            and report['specimen_eligibility_established'] is False)
    OUTPUT.parent.mkdir()
    with OUTPUT.open('x') as handle:
        yaml.safe_dump({'status': 'all-observed-strict-mapping-and-mode-unknown-outcomes-independently-verified',
            'verifier_sha256': digest(__file__), 'source_hashes': {str(p): h for p, h in pins.items()},
            'records': summaries, 'health_classification_performed': False,
            'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('Independently verified all 2330 strict mappings and retained mode unknowns')


if __name__ == '__main__':
    main()
