"""Independently verify standard-filter key headers and retained rejections."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from verify_patchseq_whole_file_integrity import digest

RESULT = Path('results/patchseq-append-notebook-keys-1412/manifest.yaml')
OUTPUT = Path('results/patchseq-append-notebook-keys-independent-1413/manifest.yaml')


def require(condition):
    if not condition:
        raise ValueError('independent header scope, storage, values or provenance differ')


def verify(file, header, h5py):
    path = header['path']
    require(path.rsplit('/', 1)[-1] in ('numericalKeys', 'textualKeys'))
    node = file
    for name in path.strip('/').split('/'):
        require(isinstance(node, h5py.Group) and type(node.get(name, getlink=True)) is h5py.HardLink)
        node = node[name]
    require(isinstance(node, h5py.Dataset) and node.ndim == 2 and node.shape[0] == 3
            and 0 < node.shape[1] <= 512 and node.dtype.kind in 'OSU' and not node.is_virtual and not node.external)
    props = node.id.get_create_plist()
    filters = [props.get_filter(k)[0] for k in range(props.get_nfilters())]
    require(all(k in (1, 2, 3) for k in filters))
    rows = [[None]*node.shape[1] for _ in range(3)]
    total = 0
    for col in range(node.shape[1]):
        for row in range(3):
            value = node[row, col]
            if hasattr(value, 'shape'):
                require(value.shape in ((), (1,)))
                value = value.item()
            require(type(value) in (str, bytes))
            raw = value.encode('utf-8') if type(value) is str else value
            require(len(raw) <= 512)
            total += len(raw)
            cell = {'raw_type': type(value).__name__, 'raw_hex': raw.hex()}
            try:
                cell['decoded_text'] = raw.decode('utf-8')
            except UnicodeDecodeError:
                cell['undefined'] = 'non-UTF8'
            rows[row][col] = cell
    actual = {'path': path, 'shape': list(node.shape), 'dtype': str(node.dtype), 'filter_IDs': filters, 'rows': rows}
    require(actual == header)
    return total


def main():
    pins = {RESULT: digest(RESULT), Path(__file__): digest(__file__)}
    report = yaml.load(RESULT.read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(p): h for p, h in report['source_hashes'].items()})
    require(all(digest(p) == h for p, h in pins.items()))
    reg = yaml.safe_load(Path('docs/validation-results/post2008-patchseq-append-notebook-keys-registration-1411.yaml').read_text())
    provenance = yaml.load(Path(reg['provenance']).read_text(), Loader=yaml.CSafeLoader)
    acquired = yaml.safe_load(Path(reg['acquisition']).read_text())
    receipts = {x['slot']: yaml.safe_load(Path(x['path']).read_text()) for x in acquired['receipts']}
    require(len(report['records']) == len(provenance['records']) == len(receipts) == 48)
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    import h5py
    summaries = []
    for saved, source in zip(report['records'], provenance['records'], strict=True):
        require(saved['slot'] == source['slot'] and saved['error'] is None)
        body = Path(receipts[saved['slot']]['complete_body_path'])
        require(digest(body) == saved['recording_sha256'] == source['recording_sha256'])
        paths = sorted(r['path'] for r in source['labnotebook_structure']['records']
                       if r.get('object_type') == 'dataset' and r['path'].endswith(('/numericalKeys', '/textualKeys')))
        require(len(paths) == len(set(paths)) == len(saved['headers']) == 2)
        require({p.rsplit('/', 1)[-1] for p in paths} == {'numericalKeys', 'textualKeys'})
        require(paths == [h['path'] for h in saved['headers']])
        with h5py.File(body, 'r') as file:
            total = sum(verify(file, header, h5py) for header in saved['headers'])
        require(total == saved['decoded_text_bytes'] and total <= 65536)
        require(digest(body) == saved['recording_sha256'])
        summaries.append({'slot': saved['slot'], 'verified_headers': 2, 'decoded_text_bytes': total})
    require(len({r['slot'] for r in summaries}) == 48 and all(digest(p) == h for p, h in pins.items()))
    require(report['notebook_trial_values_read'] is False and report['waveform_reads'] is False)
    require(report['specimen_eligibility_established'] is False)
    OUTPUT.parent.mkdir()
    with OUTPUT.open('x') as handle:
        yaml.safe_dump({'status': 'all-96-append-key-headers-independently-reconstructed',
                       'source_hashes': {str(p): h for p, h in pins.items()}, 'records': summaries,
                       'notebook_trial_values_read': False, 'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('All 96 append key/unit headers independently reconstructed')


if __name__ == '__main__':
    main()
