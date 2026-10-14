"""Separate standard-filter header read preserving the original rejection."""

from __future__ import annotations

import os
from importlib.metadata import version
from pathlib import Path

import yaml
from read_patchseq_exporter_provenance import node_at, text
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-notebook-keys-continuation-registration-1340.yaml')


def table(file, path, h5py):
    if path.rsplit('/', 1)[-1] not in ('numericalKeys', 'textualKeys'):
        raise ValueError('unregistered-value-path')
    node, error = node_at(file, path, h5py)
    if error:
        raise ValueError(error)
    if (not isinstance(node, h5py.Dataset) or node.ndim != 2 or node.shape[0] != 3
        or not 1 <= node.shape[1] <= 512 or node.dtype.kind not in 'OSU' or node.external or node.is_virtual):
        raise ValueError('unsupported-header-storage-shape-or-type')
    props = node.id.get_create_plist()
    filters = [props.get_filter(k)[0] for k in range(props.get_nfilters())]
    if any(f not in (1, 2, 3) for f in filters):
        raise ValueError('unsupported-plugin-filter')
    return node, filters


def read(file, path, h5py, budget):
    node, filters = table(file, path, h5py)
    rows = []
    for row in range(3):
        values = []
        for col in range(node.shape[1]):
            value, size = text(node[row, col])
            if size > 512 or budget[0]+size > 65536:
                raise ValueError('decoded-text-byte-bound')
            budget[0] += size
            values.append(value)
        rows.append(values)
    return {'path': path, 'shape': list(node.shape), 'dtype': str(node.dtype),
            'filter_IDs': filters, 'rows': rows}


def main():
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: digest(REG), Path(__file__): digest(__file__),
            Path(reg['original_result']): reg['original_result_sha256']}
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('sealed original outcome changed')
    original = yaml.load(Path(reg['original_result']).read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(p): h for p, h in original['source_hashes'].items()})
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('original source changed')
    original_reg = yaml.safe_load(Path(reg['original_registration']).read_text())
    acquired = yaml.safe_load(Path(original_reg['acquisition']).read_text())
    receipts = {x['slot']: yaml.safe_load(Path(x['path']).read_text()) for x in acquired['receipts']}
    if len(original['records']) != 48 or version('h5py') != reg['parser'].split('==')[1]:
        raise ValueError('scope or parser differs')
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    import h5py
    output = Path(reg['output'])
    output.parent.mkdir()
    # Validate the original metadata-only rejection across every file BEFORE
    # reading any header value. Do not pick a successful subset.
    evidence = []
    for item in original['records']:
        body = Path(receipts[item['slot']]['complete_body_path'])
        if digest(body) != item['recording_sha256']:
            raise ValueError('recording changed')
        checks = []
        with h5py.File(body, 'r') as file:
            if len(item['headers']) != 2:
                raise ValueError('unexpected key scope')
            for header in item['headers']:
                if header != {'path': header['path'], 'undefined': 'unsupported-header-storage-shape-or-type'}:
                    raise ValueError('unexpected original rejection')
                n, filters = table(file, header['path'], h5py)
                if not filters:
                    raise ValueError('original filter rejection not reproduced')
                checks.append({'path': header['path'], 'shape': list(n.shape), 'dtype': str(n.dtype), 'filter_IDs': filters})
        evidence.append({'slot': item['slot'], 'original_storage_rejection_reconstructed': checks})
    results = []
    for item, proof in zip(original['records'], evidence, strict=True):
        body = Path(receipts[item['slot']]['complete_body_path'])
        if digest(body) != item['recording_sha256']:
            raise ValueError('recording changed before header read')
        budget, error, headers = [0], None, []
        with h5py.File(body, 'r') as file:
            try:
                headers = [read(file, h['path'], h5py, budget) for h in item['headers']]
            except (ValueError, TypeError, OSError) as exc:
                error = str(exc)
        if digest(body) != item['recording_sha256']:
            raise ValueError('recording changed during read')
        results.append({**proof, 'recording_sha256': item['recording_sha256'],
                        'original_outcome_verbatim': item, 'headers': headers,
                        'error': error, 'decoded_text_bytes': budget[0]})
    if len({x['slot'] for x in results}) != 48 or any(digest(p) != h for p, h in pins.items()):
        raise ValueError('source or scope changed')
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'standard-filter-header-outcomes-retained-independent-check-pending',
            'source_hashes': {str(p): h for p, h in pins.items()}, 'records': results,
            'notebook_trial_values_read': False, 'waveform_reads': False,
            'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('Original 96 storage rejections retained; separate bounded header outcomes recorded')


if __name__ == '__main__':
    main()
