"""Independent forward block enumeration and local-to-source row validation."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml
from verify_patchseq_setting_mapping import check, require
from verify_patchseq_whole_file_integrity import digest

RESULT = Path('results/patchseq-contiguous-mapping-1355/manifest.yaml')
OUTPUT = Path('results/patchseq-contiguous-mapping-independent-1357/manifest.yaml')


def last_block(ids, modes, sweep):
    require(ids.shape == modes.shape and ids.ndim == 1)
    runs, current = [], []
    for row in range(len(ids)):
        if ids[row] == sweep and modes[row] == 0:
            current.append(row)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs[-1] if runs else []


def verify_one(table, data, pair, metadata, saved, prior):
    require(saved['original_strict_outcome_verbatim'] == prior)
    cols = table['column_candidates']
    require(len(cols['SweepNum']) == len(cols['EntrySourceType']) == 1)
    ids = data[f'column-{cols["SweepNum"][0]:04d}'][:, 0]
    modes = data[f'column-{cols["EntrySourceType"][0]:04d}'][:, 0]
    rows = last_block(ids, modes, pair['sweep_number'])
    require(bool(rows))  # Observed all-explicit-block result, not legacy fallback.
    require(saved['historical_range'] == {'selected_acquisition_rows': rows,
            'derived_lookup_view_only': True, 'raw_values_modified': False})
    # Use only original row slices, never a masked sweep-ID view or collector.
    sliced = {name: values[rows, :] for name, values in data.items()}
    local = copy.deepcopy(saved['continuation_outcome'])
    require(local['mapping_status'] == 'mapped')
    inverse = {source: position for position, source in enumerate(rows)}
    require(all(row in inverse for row in local['channel_setting_rows'].values()))
    local['channel_setting_rows'] = {name: inverse[row] for name, row in local['channel_setting_rows'].items()}
    for field in local['fields'].values():
        for name in ('native_value', 'enabled_state'):
            if 'row' in field[name]:
                require(field[name]['row'] in inverse)
                field[name]['row'] = inverse[field[name]['row']]
    proof = check(table, sliced, pair, metadata, local)
    require(proof['mapped'])
    return len(rows)


def main():
    pins = {RESULT: '0c36a1f9bf7fbeb7547b4ecbd9052d4b8fe2246d697bda1c11b3c77ca7b95c08',
        Path('scripts/verify_patchseq_setting_mapping.py'): '7dd0fce9568e4325369df10de239efb1832be20048035b0fa1b6d88ca30e1295'}
    require(all(digest(p) == h for p, h in pins.items()))
    report = yaml.load(RESULT.read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(p): h for p, h in report['source_hashes'].items()})
    require(all(digest(p) == h for p, h in pins.items()))
    def records(path):
        return yaml.load(Path(path).read_text(), Loader=yaml.CSafeLoader)['records']
    original = records('results/patchseq-setting-mapping-1349/manifest.yaml')
    numeric = records('results/patchseq-notebook-numeric-1345/manifest.yaml')
    pairing = records('results/patchseq-pairing-1324/manifest.yaml')
    qc = records('results/patchseq-waveform-QC-1328/manifest.yaml')
    require(len(original) == len(numeric) == len(pairing) == len(qc) == len(report['records']) == 48)
    summaries = []
    for old, numeric_case, pairs, measured, saved in zip(original, numeric, pairing, qc, report['records'], strict=True):
        require(old['slot'] == numeric_case['slot'] == pairs['slot'] == measured['slot'] == saved['slot'])
        require(saved['source_cell_specimen_id'] == old['source_cell_specimen_id'] == pairs['source_cell_specimen_id'])
        require(len(numeric_case['tables']) == 1 and numeric_case['tables'][0]['error'] is None)
        table = numeric_case['tables'][0]
        with np.load(table['array_path'], allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        roles = {s['path']: s['role'] for s in pairs['series']}
        metadata = {s['sweep_number']: s['instrument_metadata'] for s in measured['sweeps']}
        require(len(old['sweeps']) == len(pairs['sweeps']) == len(saved['sweeps']))
        for prior, pair, outcome in zip(old['sweeps'], pairs['sweeps'], saved['sweeps'], strict=True):
            require(prior['sweep_number'] == pair['sweep_number'])
            paths = {roles[path]: path for path in pair['series_paths']}
            verify_one(table, data, {'sweep_number': pair['sweep_number'], **paths},
                       metadata[pair['sweep_number']], outcome, prior)
        summaries.append({'slot': old['slot'], 'verified_pairs': len(saved['sweeps']),
                          'source_blocks_and_all_exported_fields_matched': True})
    require(len({s['slot'] for s in summaries}) == 48 and all(digest(p) == h for p, h in pins.items()))
    for flag in ('raw_archives_modified', 'health_classification_performed', 'model_fitting_performed', 'specimen_eligibility_established'):
        require(report[flag] is False)
    OUTPUT.parent.mkdir()
    with OUTPUT.open('x') as handle:
        yaml.safe_dump({'status': 'all-contiguous-blocks-setting-mappings-and-preserved-strict-outcomes-independently-verified',
            'verifier_sha256': digest(__file__), 'source_hashes': {str(p): h for p, h in pins.items()},
            'records': summaries, 'health_classification_performed': False, 'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('Independently verified all 2330 contiguous-block mappings and original strict outcomes')


if __name__ == '__main__':
    main()
