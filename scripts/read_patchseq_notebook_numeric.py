"""Archive only whitelisted numeric notebook columns with source coordinates."""

from __future__ import annotations

import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml
from read_patchseq_exporter_provenance import node_at
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-notebook-numeric-registration-1344.yaml')


def extract(file, header, fields, h5py, budget):
    path = header['path'].rsplit('/', 1)[0]+'/numericalValues'
    if not header['path'].endswith('/numericalKeys'):
        raise ValueError('unregistered-numeric-value-path')
    node, error = node_at(file, path, h5py)
    if error:
        raise ValueError(error)
    if (not isinstance(node, h5py.Dataset) or node.ndim != 3 or not 1 <= node.shape[0] <= 20000
        or not 1 <= node.shape[1] <= 512 or node.shape[2] != 9 or node.dtype.kind not in 'iuf'
        or node.shape[1] != header['shape'][1] or node.is_virtual or node.external):
        raise ValueError('unsupported-notebook-numeric-storage')
    props = node.id.get_create_plist()
    filters = [props.get_filter(k)[0] for k in range(props.get_nfilters())]
    if any(k not in (1, 2, 3) for k in filters):
        raise ValueError('unsupported-plugin-filter')
    columns = {name: [k for k, value in enumerate(header['rows'][0]) if value.get('decoded_text') == name]
               for name in fields}
    indices = sorted({k for values in columns.values() for k in values})
    size = len(indices)*node.shape[0]*9*8
    if budget[0]+size > 268435456:
        raise ValueError('aggregate-numeric-byte-budget')
    budget[0] += size
    arrays = {f'column-{k:04d}': np.asarray(node[:, k, :], dtype=np.float64) for k in indices}
    metadata = {'path': path, 'shape': list(node.shape), 'source_dtype': str(node.dtype), 'filter_IDs': filters,
        'column_candidates': columns, 'selected_columns': indices,
        'column_headers': {k: [header['rows'][r][k] for r in range(3)] for k in indices},
        'decoded_float64_bytes': size}
    return arrays, metadata


def last_explicit_mode_row(sweeps, modes, values, sweep, mode):
    """Strict finite-mode branch only; unknown modes never get a heuristic."""
    if sweeps.ndim != 1 or modes.shape != sweeps.shape or values.shape != (len(sweeps), 9):
        raise ValueError('selector shapes')
    rows = np.flatnonzero(sweeps == sweep)
    if not len(rows):
        return {'undefined': 'sweep-absent'}
    if np.any(~np.isfinite(modes[rows])) or not np.isin(modes[rows], [0, 1, 2]).all():
        return {'undefined': 'legacy-or-missing-source-mode'}
    candidates = [int(r) for r in rows if modes[r] == mode and (~np.isnan(values[r])).any()]
    return {'row': candidates[-1]} if candidates else {'undefined': 'no-non-NaN-vector-in-requested-mode'}


def main():
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: digest(REG), Path(__file__): digest(__file__)}
    for key in ('headers', 'independent_headers', 'acquisition'):
        pins[Path(reg[key])] = reg[key+'_sha256']
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('registered inputs changed')
    independent = yaml.load(Path(reg['independent_headers']).read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(p): h for p, h in independent['source_hashes'].items()})
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('original source changed')
    if version('h5py') != reg['scope']['parser'].split('==')[1]:
        raise ValueError('parser differs')
    headers = yaml.load(Path(reg['headers']).read_text(), Loader=yaml.CSafeLoader)['records']
    acquisition = yaml.safe_load(Path(reg['acquisition']).read_text())
    receipts = {x['slot']: yaml.safe_load(Path(x['path']).read_text()) for x in acquisition['receipts']}
    if len(headers) != len(receipts) or len(headers) != 48:
        raise ValueError('case scope differs')
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    import h5py
    output = Path(reg['output'])
    output.parent.mkdir()
    records, budget = [], [0]
    for item in headers:
        body = Path(receipts[item['slot']]['complete_body_path'])
        if digest(body) != item['recording_sha256']:
            raise ValueError('recording changed')
        tables = []
        with h5py.File(body, 'r') as file:
            for index, header in enumerate(item['headers']):
                if not header['path'].endswith('/numericalKeys'):
                    continue
                try:
                    arrays, metadata = extract(file, header, reg['scope']['fields'], h5py, budget)
                    target = output.parent/f'slot-{item["slot"]:03d}-table-{index}.npz'
                    with target.open('xb') as handle:
                        np.savez_compressed(handle, **arrays)
                    tables.append({**metadata, 'array_path': str(target), 'array_sha256': digest(target), 'error': None})
                except (ValueError, TypeError, OSError) as exc:
                    tables.append({'header_path': header['path'], 'error': str(exc)})
        if digest(body) != item['recording_sha256']:
            raise ValueError('recording changed during numeric extraction')
        records.append({'slot': item['slot'], 'recording_sha256': item['recording_sha256'], 'tables': tables})
    if len({r['slot'] for r in records}) != 48 or any(digest(p) != h for p, h in pins.items()):
        raise ValueError('source or case scope changed')
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'native-numeric-column-archives-retained-independent-check-pending',
            'source_hashes': {str(p): h for p, h in pins.items()}, 'records': records,
            'charged_float64_bytes': budget[0], 'raw_nine_page_values_preserved': True,
            'health_classification_performed': False, 'model_fitting_performed': False,
            'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('Retained whitelisted native notebook columns for all 48 files; no health classification or fit')


if __name__ == '__main__':
    main()
