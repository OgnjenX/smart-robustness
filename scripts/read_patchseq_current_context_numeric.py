"""Supplemental bounded native current context, preserving all source coordinates."""

from __future__ import annotations

import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml
from read_patchseq_exporter_provenance import node_at
from read_patchseq_notebook_numeric import extract
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-current-context-numeric-registration-1416.yaml')


def extract_context(file, header, fields, h5py, budget):
    path = header['path'].rsplit('/', 1)[0]+'/numericalValues'
    node, error = node_at(file, path, h5py)
    if error or not isinstance(node, h5py.Dataset) or node.dtype != np.dtype('float64'):
        raise ValueError('unsupported-native-float64-context-storage')
    return extract(file, header, fields, h5py, budget)


def main():
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: digest(REG), Path(__file__): digest(__file__),
            Path('scripts/read_patchseq_notebook_numeric.py'): '5aa3f1610f74434f1e5c7db2c94f3c339eaad11e60141f68e78c7a8a58d4ba05',
            Path('scripts/read_patchseq_exporter_provenance.py'): digest('scripts/read_patchseq_exporter_provenance.py')}
    inputs = []
    for batch in reg['batches']:
        for key in ('headers', 'independent_headers', 'acquisition'):
            pins[Path(batch[key])] = batch[key+'_sha256']
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError('registered numeric context changed')
        headers = yaml.load(Path(batch['headers']).read_text(), Loader=yaml.CSafeLoader)
        verified = yaml.load(Path(batch['independent_headers']).read_text(), Loader=yaml.CSafeLoader)
        expected = ('all-96-headers-and-preserved-storage-rejections-independently-reconstructed'
                    if batch['name'] == 'original' else 'all-96-append-key-headers-independently-reconstructed')
        if verified['status'] != expected:
            raise ValueError('independent header gate missing')
        pins.update({Path(p): h for p, h in verified['source_hashes'].items()})
        acquisition = yaml.safe_load(Path(batch['acquisition']).read_text())
        receipts = {r['slot']: yaml.safe_load(Path(r['path']).read_text()) for r in acquisition['receipts']}
        if len(headers['records']) != len(verified['records']) or len(receipts) != 48 or len(headers['records']) != 48:
            raise ValueError('fixed batch scope differs')
        inputs.append((batch['name'], headers['records'], receipts))
    if any(digest(p) != h for p, h in pins.items()) or version('h5py') != reg['scope']['parser'].split('==')[1]:
        raise ValueError('source or parser changed')
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    import h5py
    output = Path(reg['output'])
    output.parent.mkdir()
    records, budget = [], [0]
    for name, headers, receipts in inputs:
        for header_case in headers:
            body = Path(receipts[header_case['slot']]['complete_body_path'])
            before = digest(body)
            if before != header_case['recording_sha256'] or before != receipts[header_case['slot']]['complete_file_SHA256']:
                raise ValueError('recording changed')
            numeric = [h for h in header_case['headers'] if h['path'].endswith('/numericalKeys')]
            if len(numeric) != 1:
                raise ValueError('numerical key scope differs')
            tables = []
            with h5py.File(body, 'r') as file:
                for header in numeric:
                    try:
                        arrays, info = extract_context(file, header, reg['scope']['fields'], h5py, budget)
                        if info['source_dtype'] != 'float64':
                            raise ValueError('native source dtype differs')
                        path = output.parent/f'{name}-slot-{header_case["slot"]:03d}.npz'
                        with path.open('xb') as handle:
                            np.savez_compressed(handle, **arrays)
                        tables.append({**info, 'array_path': str(path), 'array_sha256': digest(path), 'error': None})
                    except (ValueError, TypeError, OSError) as exc:
                        tables.append({'header_path': header['path'], 'error': str(exc)})
            if digest(body) != before:
                raise ValueError('recording changed during extraction')
            records.append({'source_batch': name, 'slot': header_case['slot'], 'recording_sha256': before, 'tables': tables})
    if len({(r['source_batch'], r['slot']) for r in records}) != 96 or any(digest(p) != h for p, h in pins.items()):
        raise ValueError('source or combined scope changed')
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'supplemental-native-current-context-retained-independent-check-pending',
                       'source_hashes': {str(p): h for p, h in pins.items()}, 'records': records,
                       'charged_float64_bytes': budget[0], 'old_archives_modified': False,
                       'model_fitting_performed': False, 'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('All 96 supplemental numeric context archives retained; no sweep mapping or fit')


if __name__ == '__main__':
    main()
