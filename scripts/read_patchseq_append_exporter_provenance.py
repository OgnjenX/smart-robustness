"""Targeted provenance text and names-only notebook layout, never code execution."""

from __future__ import annotations

import os
from importlib.metadata import version
from pathlib import Path

import yaml
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-append-exporter-provenance-registration-1407.yaml')


def node_at(file, path, h5py):
    obj = file
    for part in path.strip('/').split('/'):
        if not isinstance(obj, h5py.Group):
            return None, 'parent-not-group'
        link = obj.get(part, getlink=True)
        if link is None:
            return None, 'absent'
        if type(link) is not h5py.HardLink:
            return None, 'non-hard-link-not-followed'
        obj = obj[part]
    return obj, None


def text(value):
    if hasattr(value, 'shape'):
        if value.shape not in ((), (1,)):
            raise ValueError('non-scalar-text')
        value = value.item()
    if type(value) not in (str, bytes):
        raise ValueError('not-text')
    raw = value.encode('utf-8') if type(value) is str else value
    if len(raw) > 4096:
        raise ValueError('text-byte-limit')
    result = {'raw_type': type(value).__name__, 'raw_hex': raw.hex()}
    try:
        result['decoded_text'] = raw.decode('utf-8')
    except UnicodeDecodeError:
        result['undefined'] = 'non-UTF8'
    return result, len(raw)


def inspect(file, h5py):
    out = {}
    table, error = node_at(file, '/general/generated_by', h5py)
    if error:
        out['generated_by'] = {'undefined': error}
    else:
        info = {'shape': None if table.shape is None else list(table.shape), 'dtype': str(table.dtype)} if isinstance(table, h5py.Dataset) else {}
        try:
            if (not isinstance(table, h5py.Dataset) or table.ndim != 2 or table.shape[1] != 2
                or not 1 <= table.shape[0] <= 32 or table.is_virtual or table.external
                or table.id.get_create_plist().get_nfilters() or table.dtype.kind not in 'OSU'):
                raise ValueError('unsupported-provenance-table-storage')
            values, total = [], 0
            for row in range(table.shape[0]):
                pair = []
                for column in range(2):
                    value, size = text(table[row, column])
                    total += size
                    if total > 32768:
                        raise ValueError('total-provenance-text-byte-limit')
                    pair.append(value)
                values.append(pair)
            info['rows'] = values
            info['decoded_bytes'] = total
        except (ValueError, TypeError, OSError) as exc:
            info['undefined'] = str(exc)
        out['generated_by'] = info
    script, error = node_at(file, '/general/source_script', h5py)
    if error:
        out['source_script_file_name'] = {'undefined': error}
    elif 'file_name' not in script.attrs:
        out['source_script_file_name'] = {'undefined': 'absent-attribute'}
    else:
        try:
            aid = script.attrs.get_id('file_name')
            if aid.shape not in ((), (1,)) or aid.get_storage_size() > 4096:
                raise ValueError('unsupported-attribute-shape-or-storage')
            out['source_script_file_name'], _ = text(script.attrs['file_name'])
        except (ValueError, TypeError, OSError) as exc:
            out['source_script_file_name'] = {'undefined': str(exc)}
    notebook, error = node_at(file, '/general/labnotebook', h5py)
    layout = []
    if error:
        out['labnotebook_structure'] = {'undefined': error}
    else:
        def visit(obj, path, depth):
            if len(layout) >= 256:
                raise ValueError('notebook-object-limit')
            if isinstance(obj, h5py.Group):
                layout.append({'path': path, 'object_type': 'group'})
                if depth < 2:
                    for name in sorted(obj.keys()):
                        if len(name.encode('utf-8')) > 1024:
                            raise ValueError('notebook-name-byte-limit')
                        target = path+'/'+name
                        if type(obj.get(name, getlink=True)) is h5py.HardLink:
                            visit(obj[name], target, depth+1)
                        else:
                            if len(layout) >= 256:
                                raise ValueError('notebook-object-limit')
                            layout.append({'path': target, 'undefined': 'non-hard-link-not-followed'})
            elif isinstance(obj, h5py.Dataset):
                layout.append({'path': path, 'object_type': 'dataset',
                    'shape': None if obj.shape is None else list(obj.shape), 'dtype': str(obj.dtype)})
            else:
                raise TypeError('unsupported-notebook-object')
        try:
            visit(notebook, '/general/labnotebook', 0)
            out['labnotebook_structure'] = {'records': layout}
        except (ValueError, TypeError, OSError) as exc:
            out['labnotebook_structure'] = {'records': layout, 'undefined': str(exc)}
    return out


def main():
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: digest(REG), Path(__file__): digest(__file__)}
    for key in ('acquisition', 'independent_QC'):
        pins[Path(reg[key])] = reg[key+'_sha256']
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('registered input changed')
    independent = yaml.load(Path(reg['independent_QC']).read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(p): h for p, h in independent['source_hashes'].items()})
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('source changed')
    if version('h5py') != reg['scope']['parser'].split('==')[1]:
        raise ValueError('parser differs')
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    import h5py
    acquisition = yaml.safe_load(Path(reg['acquisition']).read_text())
    records = []
    if len(acquisition['receipts']) != 48:
        raise ValueError('fixed 48-case scope differs')
    output = Path(reg['output'])
    output.parent.mkdir()
    for item in acquisition['receipts']:
        receipt = yaml.safe_load(Path(item['path']).read_text())
        body = Path(receipt['complete_body_path'])
        before = digest(body)
        if before != receipt['complete_file_SHA256']:
            raise ValueError('recording changed')
        with h5py.File(body, 'r') as file:
            records.append({'slot': item['slot'], 'recording_sha256': before, **inspect(file, h5py)})
        if digest(body) != before:
            raise ValueError('recording changed during metadata read')
    if any(digest(p) != h for p, h in pins.items()) or len({r['slot'] for r in records}) != 48:
        raise ValueError('source or scope changed')
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'targeted-provenance-outcomes-retained-independent-check-pending',
            'source_hashes': {str(p): h for p, h in pins.items()}, 'records': records,
            'waveform_or_notebook_values_read': False, 'script_bodies_read': False,
            'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('Retained exporter declarations and names-only notebook layout for all 48 recordings')


if __name__ == '__main__':
    main()
