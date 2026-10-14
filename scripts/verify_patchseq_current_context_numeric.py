"""Independent bitwise source-to-archive numeric column reconstruction."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml
from verify_patchseq_whole_file_integrity import digest

RESULT = Path('results/patchseq-current-context-numeric-1417/manifest.yaml')
OUTPUT = Path('results/patchseq-current-context-numeric-independent-1418/manifest.yaml')


def require(condition):
    if not condition:
        raise ValueError('numeric archive scope, storage, values or provenance differ')


def equal_bits(saved, original):
    require(saved.dtype == np.dtype('float64') and saved.shape == original.shape)
    require(saved.tobytes(order='C') == np.asarray(original, dtype=np.float64).tobytes(order='C'))


def main():
    pins = {RESULT: digest(RESULT), Path(__file__): digest(__file__)}
    report = yaml.load(RESULT.read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(p): h for p, h in report['source_hashes'].items()})
    require(all(digest(p) == h for p, h in pins.items()))
    reg = yaml.safe_load(Path('docs/validation-results/post2008-patchseq-current-context-numeric-registration-1416.yaml').read_text())
    sources = {}
    for batch in reg['batches']:
        headers = yaml.load(Path(batch['headers']).read_text(), Loader=yaml.CSafeLoader)['records']
        acquired = yaml.safe_load(Path(batch['acquisition']).read_text())
        receipts = {r['slot']: yaml.safe_load(Path(r['path']).read_text()) for r in acquired['receipts']}
        require(len(headers) == len(receipts) == 48)
        for header in headers:
            sources[(batch['name'], header['slot'])] = (header, receipts[header['slot']])
    require(len(sources) == len(report['records']) == 96)
    require({(r['source_batch'], r['slot']) for r in report['records']} == set(sources))
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    import h5py
    summaries, charged, archive_pins = [], 0, {}
    for case in report['records']:
        source, receipt = sources[(case['source_batch'], case['slot'])]
        body = Path(receipt['complete_body_path'])
        require(digest(body) == case['recording_sha256'] == source['recording_sha256'] == receipt['complete_file_SHA256'])
        expected_headers = [h for h in source['headers'] if h['path'].endswith('/numericalKeys')]
        require(len(case['tables']) == len(expected_headers) == 1)
        with h5py.File(body, 'r') as file:
            for table, header in zip(case['tables'], expected_headers, strict=True):
                require(table['error'] is None)
                path = header['path'].rsplit('/', 1)[0]+'/numericalValues'
                require(table['path'] == path)
                node = file
                for component in path.strip('/').split('/'):
                    require(isinstance(node, h5py.Group) and type(node.get(component, getlink=True)) is h5py.HardLink)
                    node = node[component]
                require(isinstance(node, h5py.Dataset) and node.ndim == 3 and 0 < node.shape[0] <= 20000
                        and 0 < node.shape[1] <= 512 and node.shape[1] == header['shape'][1] and node.shape[2] == 9
                        and node.dtype == np.dtype('float64') and not node.is_virtual and not node.external)
                properties = node.id.get_create_plist()
                filters = [properties.get_filter(k)[0] for k in range(properties.get_nfilters())]
                require(all(k in (1, 2, 3) for k in filters))
                require(table['shape'] == list(node.shape) and table['source_dtype'] == str(node.dtype)
                        and table['filter_IDs'] == filters)
                labels = [c.get('decoded_text') for c in header['rows'][0]]
                memberships = {name: [k for k, label in enumerate(labels) if label == name] for name in reg['scope']['fields']}
                indices = sorted({k for cols in memberships.values() for k in cols})
                require(table['column_candidates'] == memberships and table['selected_columns'] == indices)
                require(table['column_headers'] == {k: [header['rows'][r][k] for r in range(3)] for k in indices})
                cost = len(indices)*node.shape[0]*9*8
                require(cost == table['decoded_float64_bytes'])
                charged += cost
                require(charged <= 268435456)
                archive = Path(table['array_path'])
                require(archive.parent == RESULT.parent and digest(archive) == table['array_sha256'])
                archive_pins[archive] = table['array_sha256']
                with np.load(archive, allow_pickle=False) as arrays:
                    require(set(arrays.files) == {f'column-{k:04d}' for k in indices})
                    for k in indices:
                        equal_bits(arrays[f'column-{k:04d}'], node[:, k, :])
        require(digest(body) == case['recording_sha256'])
        summaries.append({'source_batch': case['source_batch'], 'slot': case['slot'],
                          'verified_columns': len(indices), 'raw_value_bits_matched': True})
    require(len({(r['source_batch'], r['slot']) for r in summaries}) == 96 and charged == report['charged_float64_bytes'])
    require(all(digest(p) == h for p, h in {**pins, **archive_pins}.items()))
    require(report['old_archives_modified'] is False and report['model_fitting_performed'] is False
            and report['specimen_eligibility_established'] is False)
    OUTPUT.parent.mkdir()
    with OUTPUT.open('x') as handle:
        yaml.safe_dump({'status': 'all-96-native-current-context-columns-independently-bitwise-reconstructed',
                       'source_hashes': {str(p): h for p, h in {**pins, **archive_pins}.items()}, 'records': summaries,
                       'charged_float64_bytes': charged, 'old_archives_modified': False,
                       'model_fitting_performed': False, 'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('All 96 native current-context archives independently bitwise reconstructed')


if __name__ == '__main__':
    main()
