"""Independent strict reconstruction of successful bounded provenance records."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from verify_patchseq_whole_file_integrity import digest

RESULT = Path('results/patchseq-exporter-provenance-1334/manifest.yaml')
OUTPUT = Path('results/patchseq-exporter-provenance-independent-1335/manifest.yaml')


def require(value, message):
    if not value:
        raise ValueError(message)


def locate(file, path, h5py):
    node = file
    for part in path.strip('/').split('/'):
        require(isinstance(node, h5py.Group) and type(node.get(part, getlink=True)) is h5py.HardLink,
                'missing or non-hard provenance path')
        node = node[part]
    return node


def decode(value):
    if hasattr(value, 'shape'):
        require(value.shape in ((), (1,)), 'text shape')
        value = value.item()
    require(type(value) in (bytes, str), 'text type')
    data = value if type(value) is bytes else value.encode('utf-8')
    require(len(data) <= 4096, 'text bound')
    out = {'raw_type': type(value).__name__, 'raw_hex': data.hex()}
    try:
        out['decoded_text'] = data.decode('utf-8')
    except UnicodeDecodeError:
        out['undefined'] = 'non-UTF8'
    return out, len(data)


def reconstruct(file, h5py):
    table = locate(file, '/general/generated_by', h5py)
    require(isinstance(table, h5py.Dataset) and table.ndim == 2 and table.shape[1] == 2
            and 0 < table.shape[0] <= 32 and table.dtype.kind in 'OSU'
            and not table.is_virtual and not table.external and table.id.get_create_plist().get_nfilters() == 0,
            'unsupported provenance table')
    rows, total = [], 0
    for row in range(table.shape[0]):
        cells = []
        for col in range(2):
            cell, length = decode(table[row, col])
            total += length
            require(total <= 32768, 'total text bound')
            cells.append(cell)
        rows.append(cells)
    script = locate(file, '/general/source_script', h5py)
    require('file_name' in script.attrs, 'missing script-name attribute')
    info = script.attrs.get_id('file_name')
    require(info.shape in ((), (1,)) and info.get_storage_size() <= 4096, 'attribute bound')
    filename, _ = decode(script.attrs['file_name'])
    root = '/general/labnotebook'
    pending, layout = [(root, locate(file, root, h5py), 0)], []
    while pending:
        path, node, depth = pending.pop()
        require(len(layout) < 256, 'notebook object bound')
        if node is None:
            layout.append({'path': path, 'undefined': 'non-hard-link-not-followed'})
        elif isinstance(node, h5py.Dataset):
            layout.append({'path': path, 'object_type': 'dataset',
                'shape': None if node.shape is None else list(node.shape), 'dtype': str(node.dtype)})
        elif isinstance(node, h5py.Group):
            layout.append({'path': path, 'object_type': 'group'})
            if depth < 2:
                for name in reversed(sorted(node.keys())):
                    require(len(name.encode('utf-8')) <= 1024, 'name bound')
                    pending.append((path+'/'+name, node[name] if type(node.get(name, getlink=True)) is h5py.HardLink else None, depth+1))
        else:
            raise TypeError('unsupported notebook type')
    return {'generated_by': {'shape': list(table.shape), 'dtype': str(table.dtype), 'rows': rows, 'decoded_bytes': total},
            'source_script_file_name': filename, 'labnotebook_structure': {'records': layout}}


def main():
    pins = {RESULT: 'a090c2b7aabe58b9fe1811de74008dd59518f0a93dc1f62e12e90bfc20ca7cae'}
    require(digest(RESULT) == pins[RESULT], 'sealed result changed')
    report = yaml.load(RESULT.read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(p): h for p, h in report['source_hashes'].items()})
    require(all(digest(p) == h for p, h in pins.items()), 'source changed')
    reg = yaml.safe_load(Path('docs/validation-results/post2008-patchseq-exporter-provenance-registration-1333.yaml').read_text())
    acquired = yaml.safe_load(Path(reg['acquisition']).read_text())
    receipts = {x['slot']: yaml.safe_load(Path(x['path']).read_text()) for x in acquired['receipts']}
    require(len(report['records']) == len(receipts) == 48, 'scope differs')
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    import h5py
    summaries = []
    for saved in report['records']:
        receipt = receipts[saved['slot']]
        body = Path(receipt['complete_body_path'])
        require(digest(body) == receipt['complete_file_SHA256'] == saved['recording_sha256'], 'body differs')
        with h5py.File(body, 'r') as file:
            actual = {'slot': saved['slot'], 'recording_sha256': saved['recording_sha256'], **reconstruct(file, h5py)}
        require(actual == saved, 'independent descriptors differ')
        require(digest(body) == saved['recording_sha256'], 'body changed during read')
        summaries.append({'slot': saved['slot'], 'exact_metadata_match': True})
    require(len({x['slot'] for x in summaries}) == 48, 'duplicate slot')
    require(all(digest(p) == h for p, h in pins.items()), 'source changed during reconstruction')
    require(report['waveform_or_notebook_values_read'] is False and report['script_bodies_read'] is False
            and report['specimen_eligibility_established'] is False, 'scope claim differs')
    OUTPUT.parent.mkdir()
    with OUTPUT.open('x') as handle:
        yaml.safe_dump({'status': 'all-48-provenance-tables-script-name-attributes-and-notebook-layouts-exactly-reconstructed',
            'verifier_sha256': digest(__file__), 'source_hashes': {str(p): h for p, h in pins.items()},
            'records': summaries, 'waveform_or_notebook_values_read': False, 'script_bodies_read': False,
            'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('Independently reconstructed targeted provenance metadata for all 48 recordings')


if __name__ == '__main__':
    main()
