"""Bounded series metadata inventory with no waveform/reference-vector reads."""

from __future__ import annotations

import math
import os
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import yaml
from verify_patchseq_whole_file_integrity import digest, same

REG = Path('docs/validation-results/post2008-patchseq-trace-schema-registration-1318.yaml')
REG_HASH = '59408cad9a2c9cf8128f0cedfec2543c845e16e973b1424c18b38e55ac6d793f'


def scalar_descriptor(value, maximum):
    if hasattr(value, 'shape'):
        if value.shape not in ((), (1,)):
            return {'raw_type': type(value).__name__, 'undefined': 'non-scalar'}
        value = value.item()
    kind = type(value).__name__
    if type(value) is bytes:
        if len(value) > maximum:
            return {'raw_type': kind, 'undefined': 'decoded-size-limit-exceeded'}
        try:
            text = value.decode('utf-8')
        except UnicodeDecodeError:
            return {'raw_type': kind, 'raw_hex': value.hex(), 'undefined': 'non-UTF8'}
        return {'raw_type': kind, 'raw_hex': value.hex(), 'decoded_text': text}
    if type(value) is str:
        return {'raw_type': kind, 'value': value} if len(value.encode()) <= maximum else {'raw_type': kind, 'undefined': 'decoded-size-limit-exceeded'}
    if type(value) in (bool, int, float):
        if type(value) is float and not math.isfinite(value):
            return {'raw_type': kind, 'undefined': 'nan' if math.isnan(value) else 'positive-infinity' if value > 0 else 'negative-infinity'}
        return {'raw_type': kind, 'value': value}
    return {'raw_type': kind, 'undefined': 'unsupported-type'}


def inventory(file, policy, h5py, records, reads):
    maximum = policy['maximum_scalar_or_attribute_decoded_bytes']
    def append(record):
        if len(records) >= policy['maximum_structural_objects_per_file']:
            raise ValueError('registered structural-object bound exceeded')
        records.append(record)

    def names(keys):
        values = sorted(keys)
        if any(len(name.encode()) > policy['maximum_name_utf8_bytes'] for name in values):
            raise ValueError('registered structural-name bound exceeded')
        return values

    def attributes(node, permitted, path):
        result = {}
        for key in permitted:
            if key not in node.attrs:
                result[key] = {'present': False}
                continue
            info = node.attrs.get_id(key)
            descriptor = {'present': True, 'shape': None if info.shape is None else list(info.shape), 'dtype': str(info.dtype)}
            if info.shape not in ((), (1,)) or info.get_storage_size() > maximum:
                descriptor['undefined'] = 'unsupported-shape-or-storage-bound'
            else:
                reads['attributes'].append(path+'@'+key)
                descriptor['scalar'] = scalar_descriptor(node.attrs[key], maximum)
            result[key] = descriptor
        return result

    def walk(path, node, depth):
        record = {'path': path, 'group_depth': depth, 'link_type': 'HardLink', 'attribute_names': names(node.attrs.keys())}
        if isinstance(node, h5py.Group):
            record.update(object_type='group', attributes=attributes(node, policy['allowed_group_attribute_values'], path))
            append(record)
            for name in names(node.keys()):
                child_path = path.rstrip('/')+'/'+name
                link = node.get(name, getlink=True)
                if type(link) is not h5py.HardLink:
                    append({'path': child_path, 'link_type': type(link).__name__, 'unsupported': 'non-hard-link-never-followed'})
                    continue
                child = node[name]
                if isinstance(child, h5py.Group) and depth >= policy['maximum_group_depth_below_root']:
                    append({'path': child_path, 'link_type': 'HardLink', 'unsupported': 'group-depth-bound'})
                    continue
                walk(child_path, child, depth+1 if isinstance(child, h5py.Group) else depth)
        elif isinstance(node, h5py.Dataset):
            record.update(object_type='dataset', shape=None if node.shape is None else list(node.shape),
                dtype=str(node.dtype), chunks=None if node.chunks is None else list(node.chunks), compression=node.compression,
                attributes=attributes(node, policy['allowed_dataset_attribute_values'], path))
            if path.split('/')[-1] in policy['allowed_scalar_dataset_names']:
                if (node.shape not in ((), (1,)) or node.compression is not None or node.is_virtual
                    or node.external or node.id.get_storage_size() > maximum):
                    record['scalar_metadata'] = {'undefined': 'unsupported-shape-or-storage'}
                else:
                    reads['datasets'].append(path)
                    record['scalar_metadata'] = scalar_descriptor(node[()], maximum)
            append(record)
        else:
            raise TypeError('unsupported HDF5 object')

    for root_path in policy['read_only_roots']:
        cursor = file
        for part in root_path.strip('/').split('/'):
            if not isinstance(cursor, h5py.Group):
                append({'path': root_path, 'unsupported': 'root-parent-not-group'})
                break
            link = cursor.get(part, getlink=True)
            if link is None:
                append({'path': root_path, 'present': False})
                break
            if type(link) is not h5py.HardLink:
                append({'path': root_path, 'link_type': type(link).__name__, 'unsupported': 'non-hard-root-never-followed'})
                break
            cursor = cursor[part]
        else:
            walk(root_path, cursor, 0)


def main():
    if digest(REG) != REG_HASH:
        raise ValueError('sealed trace-schema registration changed')
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: REG_HASH, Path(__file__): digest(__file__)}
    for key in ('acquisition', 'independent_acquisition', 'subset'):
        pins[Path(reg[key])] = reg[key+'_sha256']
    pins[Path('scripts/verify_patchseq_whole_file_integrity.py')] = '22e40726389126ff7a875794138145d222ed5b87515e87a2ec9bccadce78c057'
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('registered input changed')
    independent = yaml.load(Path(reg['independent_acquisition']).read_text(), Loader=yaml.CSafeLoader)
    if (independent['status'] != 'all-selected-acquisition-accounting-and-identity-outcomes-independently-verified'
        or independent['summary']['content_and_identity_passes'] != 48):
        raise ValueError('independent acquisition gate incomplete')
    pins.update({Path(p): value for p, value in independent['source_hashes'].items()})
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('verified recording evidence changed')
    acquired = yaml.safe_load(Path(reg['acquisition']).read_text())
    slots = yaml.safe_load(Path(reg['subset']).read_text())['slots']
    same(slots, acquired['selected_slots'])
    if len(slots) != 48 or version('h5py') != reg['parser_dependency'].split('==')[1]:
        raise ValueError('source scope or registered parser differs')
    os.environ['HDF5_PLUGIN_PRELOAD'] = reg['HDF5_plugin_preload']
    import h5py
    output = Path(reg['output'])
    output.parent.mkdir()
    reports = []
    for slot in slots:
        matches = [r for r in acquired['receipts'] if r['slot'] == slot['slot']]
        if len(matches) != 1 or digest(matches[0]['path']) != matches[0]['sha256']:
            raise ValueError('immutable slot receipt differs')
        receipt = yaml.safe_load(Path(matches[0]['path']).read_text())
        binary = Path(receipt['complete_body_path'])
        if digest(binary) != slot['expected_SHA256']:
            raise ValueError('complete recording changed')
        records, reads, error = [], {'attributes': [], 'datasets': []}, None
        try:
            with h5py.File(binary, 'r') as file:
                inventory(file, reg['scope'], h5py, records, reads)
        except (ValueError, TypeError, OSError) as exc:
            error = {'type': type(exc).__name__, 'message': str(exc)}
        if digest(binary) != slot['expected_SHA256']:
            raise ValueError('recording changed during metadata inventory')
        reports.append({'slot': slot['slot'], 'source_cell_specimen_id': slot['source_cell_specimen_id'],
            'status': 'schema-metadata-inventoried' if error is None else 'schema-metadata-incomplete-retained',
            'records': records, 'logical_metadata_reads': reads, 'error': error,
            'waveform_or_timestamp_vectors_read': False, 'specimen_eligibility_established': False})
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('source changed during inventory')
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'all-selected-schema-metadata-outcomes-retained-independent-verification-pending',
            'source_hashes': {str(p): value for p, value in pins.items()}, 'records': reports,
            'finished_at_utc': datetime.now(UTC).isoformat(), 'network_requests': 0,
            'waveform_or_timestamp_vectors_read': False, 'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('Fixed 48-file trace metadata inventory retained; no waveform samples read')


if __name__ == '__main__':
    main()
