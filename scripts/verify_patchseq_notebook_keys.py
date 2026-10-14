"""Independently verify standard-filter key headers and retained rejections."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from verify_patchseq_whole_file_integrity import digest

RESULT = Path('results/patchseq-notebook-keys-continuation-1341/manifest.yaml')
OUTPUT = Path('results/patchseq-notebook-keys-independent-1342/manifest.yaml')


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
    require(bool(filters) and all(k in (1, 2, 3) for k in filters))
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
    pins = {RESULT: 'b306314d8ca2f916300dac5733cd6a47ba67f417a95819b8a567ddc7b2e56835'}
    require(digest(RESULT) == pins[RESULT])
    report = yaml.load(RESULT.read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(p): h for p, h in report['source_hashes'].items()})
    require(all(digest(p) == h for p, h in pins.items()))
    original = yaml.load(Path('results/patchseq-notebook-keys-1339/manifest.yaml').read_text(), Loader=yaml.CSafeLoader)
    acquired = yaml.safe_load(Path('results/patchseq-intrinsic-acquisition-1314/manifest.yaml').read_text())
    receipts = {x['slot']: yaml.safe_load(Path(x['path']).read_text()) for x in acquired['receipts']}
    require(len(report['records']) == len(original['records']) == len(receipts) == 48)
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    import h5py
    summaries = []
    for saved, old in zip(report['records'], original['records'], strict=True):
        require(saved['original_outcome_verbatim'] == old and saved['slot'] == old['slot'] and saved['error'] is None)
        receipt = receipts[saved['slot']]
        body = Path(receipt['complete_body_path'])
        require(digest(body) == saved['recording_sha256'] == receipt['complete_file_SHA256'])
        require(len(saved['headers']) == len(old['headers']) == len(saved['original_storage_rejection_reconstructed']) == 2)
        total = 0
        with h5py.File(body, 'r') as file:
            for header, rejected, storage in zip(saved['headers'], old['headers'], saved['original_storage_rejection_reconstructed'], strict=True):
                require(rejected == {'path': header['path'], 'undefined': 'unsupported-header-storage-shape-or-type'})
                require(storage == {k: header[k] for k in ('path', 'shape', 'dtype', 'filter_IDs')})
                total += verify(file, header, h5py)
        require(total <= 65536 and total == saved['decoded_text_bytes'] and digest(body) == saved['recording_sha256'])
        summaries.append({'slot': saved['slot'], 'verified_headers': 2, 'decoded_text_bytes': total})
    require(len({s['slot'] for s in summaries}) == 48 and all(digest(p) == h for p, h in pins.items()))
    require(report['notebook_trial_values_read'] is False and report['waveform_reads'] is False
            and report['specimen_eligibility_established'] is False)
    OUTPUT.parent.mkdir()
    with OUTPUT.open('x') as handle:
        yaml.safe_dump({'status': 'all-96-headers-and-preserved-storage-rejections-independently-reconstructed',
            'verifier_sha256': digest(__file__), 'source_hashes': {str(p): h for p, h in pins.items()},
            'records': summaries, 'notebook_trial_values_read': False, 'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('Independently reconstructed all 96 key headers and retained original storage rejections')


if __name__ == '__main__':
    main()
