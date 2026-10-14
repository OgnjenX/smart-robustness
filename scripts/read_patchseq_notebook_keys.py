"""Bounded notebook text headers only, not trial values or physiological arrays."""

from __future__ import annotations

import os
from importlib.metadata import version
from pathlib import Path

import yaml
from read_patchseq_exporter_provenance import node_at, text
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-notebook-keys-registration-1338.yaml')


def read_header(file, path, h5py, budget):
    result = {'path': path}
    try:
        if path.rsplit('/', 1)[-1] not in ('numericalKeys', 'textualKeys'):
            raise ValueError('unregistered-value-path')
        node, error = node_at(file, path, h5py)
        if error:
            raise ValueError(error)
        if (not isinstance(node, h5py.Dataset) or node.ndim != 2 or node.shape[0] != 3
            or not 1 <= node.shape[1] <= 512 or node.dtype.kind not in 'OSU' or node.external
            or node.is_virtual or node.id.get_create_plist().get_nfilters()):
            raise ValueError('unsupported-header-storage-shape-or-type')
        result.update(shape=list(node.shape), dtype=str(node.dtype))
        rows = []
        for row in range(3):
            cells = []
            for col in range(node.shape[1]):
                cell, cost = text(node[row, col])
                if cost > 512 or budget[0]+cost > 65536:
                    raise ValueError('decoded-text-byte-bound')
                budget[0] += cost
                cells.append(cell)
            rows.append(cells)
        result['rows'] = rows
    except (ValueError, TypeError, OSError) as exc:
        result['undefined'] = str(exc)
    return result


def main():
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: digest(REG), Path(__file__): digest(__file__),
            Path('scripts/read_patchseq_exporter_provenance.py'): digest('scripts/read_patchseq_exporter_provenance.py')}
    for key in ('provenance', 'independent_provenance', 'acquisition'):
        pins[Path(reg[key])] = reg[key+'_sha256']
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('registered input changed')
    independent = yaml.load(Path(reg['independent_provenance']).read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(p): h for p, h in independent['source_hashes'].items()})
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('source changed')
    provenance = yaml.load(Path(reg['provenance']).read_text(), Loader=yaml.CSafeLoader)
    acquisition = yaml.safe_load(Path(reg['acquisition']).read_text())
    receipts = {r['slot']: yaml.safe_load(Path(r['path']).read_text()) for r in acquisition['receipts']}
    if len(provenance['records']) != 48 or version('h5py') != reg['scope']['parser'].split('==')[1]:
        raise ValueError('scope or parser changed')
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    import h5py
    output = Path(reg['output'])
    output.parent.mkdir()
    results = []
    for entry in provenance['records']:
        body = Path(receipts[entry['slot']]['complete_body_path'])
        if digest(body) != entry['recording_sha256']:
            raise ValueError('recording changed')
        paths = [x['path'] for x in entry['labnotebook_structure']['records']
                 if x.get('object_type') == 'dataset' and x['path'].rsplit('/', 1)[-1] in ('numericalKeys', 'textualKeys')]
        budget = [0]
        with h5py.File(body, 'r') as file:
            headers = [read_header(file, path, h5py, budget) for path in paths]
        if digest(body) != entry['recording_sha256']:
            raise ValueError('recording changed during header read')
        results.append({'slot': entry['slot'], 'recording_sha256': entry['recording_sha256'],
                        'headers': headers, 'decoded_text_bytes': budget[0]})
    if len({r['slot'] for r in results}) != 48 or any(digest(p) != h for p, h in pins.items()):
        raise ValueError('source or scope changed')
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'all-header-outcomes-retained-independent-check-pending',
            'source_hashes': {str(p): h for p, h in pins.items()}, 'records': results,
            'notebook_trial_values_read': False, 'waveform_reads': False,
            'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('Retained bounded notebook field/unit headers for all 48 recordings')


if __name__ == '__main__':
    main()
