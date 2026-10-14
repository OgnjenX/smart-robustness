"""Source-defined lookup view; raw archives and original mapper are unchanged."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from map_patchseq_notebook_settings import map_pair
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-contiguous-mapping-registration-1354.yaml')


def lookup_view(table, data, sweep):
    columns = table['column_candidates']
    if any(len(columns.get(name, [])) != 1 for name in ('SweepNum', 'EntrySourceType')):
        return None, {'undefined': 'ambiguous-sweep-or-mode-column'}
    sweep_key = f'column-{columns["SweepNum"][0]:04d}'
    mode_key = f'column-{columns["EntrySourceType"][0]:04d}'
    ids, modes = data[sweep_key][:, 0], data[mode_key][:, 0]
    matches = np.flatnonzero((ids == sweep) & (modes == 0))
    if len(matches) == 0:
        return None, {'undefined': 'no-explicit-acquisition-block-legacy-branch-not-ported'}
    first = last = int(matches[-1])
    for row in matches[-2::-1]:
        if int(row)+1 != first:
            break
        first = int(row)
    identities = data[sweep_key].copy()
    keep = np.zeros(len(ids), dtype=bool)
    keep[first:last+1] = True
    identities[~keep, 0] = np.nan
    view = dict(data)
    view[sweep_key] = identities
    return view, {'selected_acquisition_rows': list(range(first, last+1)),
                  'derived_lookup_view_only': True, 'raw_values_modified': False}


def main():
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: digest(REG), Path(__file__): digest(__file__), Path(reg['source_audit']): digest(reg['source_audit'])}
    for name in ('original_result', 'independent_original'):
        pins[Path(reg[name])] = reg[name+'_sha256']
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('sealed evidence changed')
    independent = yaml.load(Path(reg['independent_original']).read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(p): h for p, h in independent['source_hashes'].items()})
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('source changed')
    original = yaml.load(Path(reg['original_result']).read_text(), Loader=yaml.CSafeLoader)['records']
    numeric = yaml.load(Path('results/patchseq-notebook-numeric-1345/manifest.yaml').read_text(), Loader=yaml.CSafeLoader)['records']
    pairing = yaml.load(Path('results/patchseq-pairing-1324/manifest.yaml').read_text(), Loader=yaml.CSafeLoader)['records']
    qc = yaml.load(Path('results/patchseq-waveform-QC-1328/manifest.yaml').read_text(), Loader=yaml.CSafeLoader)['records']
    if not len(original) == len(numeric) == len(pairing) == len(qc) == 48:
        raise ValueError('scope differs')
    output = Path(reg['output'])
    output.parent.mkdir()
    results = []
    for old, numeric_case, pairs, measured in zip(original, numeric, pairing, qc, strict=True):
        if not old['slot'] == numeric_case['slot'] == pairs['slot'] == measured['slot']:
            raise ValueError('case identity differs')
        if len(numeric_case['tables']) != 1 or numeric_case['tables'][0]['error'] is not None:
            raise ValueError('unresolved notebook table')
        table = numeric_case['tables'][0]
        with np.load(table['array_path'], allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        by_path = {s['path']: s['role'] for s in pairs['series']}
        metadata = {s['sweep_number']: s['instrument_metadata'] for s in measured['sweeps']}
        if len(old['sweeps']) != len(pairs['sweeps']):
            raise ValueError('original sweep scope differs')
        outcomes = []
        for prior, pair in zip(old['sweeps'], pairs['sweeps'], strict=True):
            sweep = pair['sweep_number']
            if prior['sweep_number'] != sweep:
                raise ValueError('sweep identity differs')
            view, selected = lookup_view(table, data, sweep)
            if view is None:
                outcome = {'sweep_number': sweep, 'mapping_status': 'unknown', 'fields': {}, 'reason': selected['undefined']}
            else:
                paths = {by_path[path]: path for path in pair['series_paths']}
                outcome = map_pair(table, view, sweep, paths['response'], paths['stimulus'], metadata[sweep])
            outcomes.append({'original_strict_outcome_verbatim': prior, 'historical_range': selected, 'continuation_outcome': outcome})
        results.append({'slot': old['slot'], 'source_cell_specimen_id': old['source_cell_specimen_id'], 'sweeps': outcomes})
    if len({r['slot'] for r in results}) != 48 or any(digest(p) != h for p, h in pins.items()):
        raise ValueError('source or scope changed')
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'historical-explicit-range-outcomes-retained-independent-check-pending',
            'source_hashes': {str(p): h for p, h in pins.items()}, 'records': results,
            'raw_archives_modified': False, 'health_classification_performed': False,
            'model_fitting_performed': False, 'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('Retained separate contiguous-range mapping outcomes and all original strict outcomes')


if __name__ == '__main__':
    main()
