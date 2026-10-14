"""Read bounded key/unit headers, never numbers, waveforms or script bodies."""

from __future__ import annotations

import os
from importlib.metadata import version
from pathlib import Path

import yaml
from continue_patchseq_notebook_keys import read
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-append-notebook-keys-registration-1411.yaml')


def key_paths(layout):
    if 'undefined' in layout:
        raise ValueError('notebook layout unresolved')
    paths = [r['path'] for r in layout['records'] if r.get('object_type') == 'dataset'
             and r['path'].rsplit('/', 1)[-1] in ('numericalKeys', 'textualKeys')]
    if len(paths) != 2 or len(set(paths)) != 2 or {p.rsplit('/', 1)[-1] for p in paths} != {'numericalKeys', 'textualKeys'}:
        raise ValueError('fixed numerical/textual header scope differs')
    return sorted(paths)


def main():
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: digest(REG), Path(__file__): digest(__file__),
            Path('scripts/continue_patchseq_notebook_keys.py'): 'f2183c92e90dd19e6cd8267565372f6c77c3cb9198d8ba5f1f6c89bd0f1ad568',
            Path('scripts/read_patchseq_exporter_provenance.py'): digest('scripts/read_patchseq_exporter_provenance.py')}
    for key in ('provenance', 'independent_provenance', 'acquisition'):
        pins[Path(reg[key])] = reg[key+'_sha256']
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('registered source changed')
    verified = yaml.load(Path(reg['independent_provenance']).read_text(), Loader=yaml.CSafeLoader)
    if verified['status'] != 'all-48-provenance-tables-script-name-attributes-and-notebook-layouts-exactly-reconstructed':
        raise ValueError('provenance prerequisite missing')
    pins.update({Path(p): h for p, h in verified['source_hashes'].items()})
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('verified source changed')
    provenance = yaml.load(Path(reg['provenance']).read_text(), Loader=yaml.CSafeLoader)
    acquired = yaml.safe_load(Path(reg['acquisition']).read_text())
    receipts = {r['slot']: yaml.safe_load(Path(r['path']).read_text()) for r in acquired['receipts']}
    if len(provenance['records']) != len(receipts) or len(receipts) != 48 or version('h5py') != reg['parser'].split('==')[1]:
        raise ValueError('scope or parser differs')
    os.environ['HDF5_PLUGIN_PRELOAD'] = reg['plugins_disabled']
    import h5py
    results = []
    for case in provenance['records']:
        body = Path(receipts[case['slot']]['complete_body_path'])
        before = digest(body)
        if before != case['recording_sha256'] or before != receipts[case['slot']]['complete_file_SHA256']:
            raise ValueError('recording changed')
        budget, headers, error = [0], [], None
        try:
            paths = key_paths(case['labnotebook_structure'])
            with h5py.File(body, 'r') as file:
                headers = [read(file, path, h5py, budget) for path in paths]
        except (ValueError, TypeError, OSError) as exc:
            error = str(exc)
        if digest(body) != before:
            raise ValueError('recording changed during header read')
        results.append({'slot': case['slot'], 'recording_sha256': before, 'headers': headers,
                        'error': error, 'decoded_text_bytes': budget[0]})
    if len({r['slot'] for r in results}) != 48 or any(digest(p) != h for p, h in pins.items()):
        raise ValueError('source or case scope changed')
    output = Path(reg['output'])
    output.parent.mkdir()
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'append-key-header-outcomes-retained-independent-check-pending',
                       'source_hashes': {str(p): h for p, h in pins.items()}, 'records': results,
                       'notebook_trial_values_read': False, 'waveform_reads': False,
                       'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('All append key/header outcomes retained without trial-value reads')


if __name__ == '__main__':
    main()
