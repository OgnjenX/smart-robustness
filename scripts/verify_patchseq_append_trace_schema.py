"""Independent append metadata reconstruction; never waveform access."""

from __future__ import annotations

import math
import os
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import yaml
from verify_patchseq_whole_file_integrity import digest, same

REG = Path('docs/validation-results/post2008-patchseq-append-trace-schema-registration-1380.yaml')
RESULT = Path('results/patchseq-append-trace-schema-1381/manifest.yaml')
OUTPUT = Path('results/patchseq-append-trace-schema-independent-1382/manifest.yaml')


def decode(raw, maximum):
    if hasattr(raw, 'shape'):
        if raw.shape not in ((), (1,)):
            return {'raw_type': type(raw).__name__, 'undefined': 'non-scalar'}
        raw = raw.item()
    kind = type(raw).__name__
    if type(raw) in (str, bytes):
        length = len(raw.encode('utf-8')) if type(raw) is str else len(raw)
        if length > maximum:
            return {'raw_type': kind, 'undefined': 'decoded-size-limit-exceeded'}
        if type(raw) is str:
            return {'raw_type': kind, 'value': raw}
        result = {'raw_type': kind, 'raw_hex': raw.hex()}
        try:
            result['decoded_text'] = raw.decode('utf-8')
        except UnicodeDecodeError:
            result['undefined'] = 'non-UTF8'
        return result
    if type(raw) in (bool, int, float):
        if type(raw) is float and not math.isfinite(raw):
            flag = 'nan' if math.isnan(raw) else 'negative-infinity' if raw < 0 else 'positive-infinity'
            return {'raw_type': kind, 'undefined': flag}
        return {'raw_type': kind, 'value': raw}
    return {'raw_type': kind, 'undefined': 'unsupported-type'}


def reconstruct(file, policy, h5py, records, reads):
    bound = policy['maximum_scalar_or_attribute_decoded_bytes']
    def bounded_names(keys):
        values = sorted(keys)
        if any(len(k.encode('utf-8')) > policy['maximum_name_utf8_bytes'] for k in values):
            raise ValueError('structural name limit')
        return values

    def attrs(node, path, permitted):
        result = {}
        for name in permitted:
            if name not in node.attrs:
                result[name] = {'present': False}
                continue
            info = node.attrs.get_id(name)
            entry = {'present': True, 'shape': None if info.shape is None else list(info.shape), 'dtype': str(info.dtype)}
            if info.shape in ((), (1,)) and info.get_storage_size() <= bound:
                reads['attributes'].append(path+'@'+name)
                entry['scalar'] = decode(node.attrs[name], bound)
            else:
                entry['undefined'] = 'unsupported-shape-or-storage-bound'
            result[name] = entry
        return result

    for root in policy['read_only_roots']:
        node, pending = file, []
        for part in root.strip('/').split('/'):
            if not isinstance(node, h5py.Group):
                pending.append(('record', {'path': root, 'unsupported': 'root-parent-not-group'}))
                break
            link = node.get(part, getlink=True)
            if link is None:
                pending.append(('record', {'path': root, 'present': False}))
                break
            if type(link) is not h5py.HardLink:
                pending.append(('record', {'path': root, 'link_type': type(link).__name__, 'unsupported': 'non-hard-root-never-followed'}))
                break
            node = node[part]
        else:
            pending.append(('node', root, node, 0))
        while pending:
            task = pending.pop()
            if task[0] == 'record':
                descriptor = task[1]
            else:
                _, path, node, depth = task
                descriptor = {'path': path, 'group_depth': depth, 'link_type': 'HardLink', 'attribute_names': bounded_names(node.attrs.keys())}
                if isinstance(node, h5py.Group):
                    descriptor.update(object_type='group', attributes=attrs(node, path, policy['allowed_group_attribute_values']))
                    for name in reversed(bounded_names(node.keys())):
                        child_path = path.rstrip('/')+'/'+name
                        link = node.get(name, getlink=True)
                        if type(link) is not h5py.HardLink:
                            pending.append(('record', {'path': child_path, 'link_type': type(link).__name__, 'unsupported': 'non-hard-link-never-followed'}))
                            continue
                        child = node[name]
                        is_group = isinstance(child, h5py.Group)
                        if is_group and depth >= policy['maximum_group_depth_below_root']:
                            pending.append(('record', {'path': child_path, 'link_type': 'HardLink', 'unsupported': 'group-depth-bound'}))
                        else:
                            pending.append(('node', child_path, child, depth+1 if is_group else depth))
                elif isinstance(node, h5py.Dataset):
                    descriptor.update(object_type='dataset', shape=None if node.shape is None else list(node.shape),
                        dtype=str(node.dtype), chunks=None if node.chunks is None else list(node.chunks), compression=node.compression,
                        attributes=attrs(node, path, policy['allowed_dataset_attribute_values']))
                    if path.rsplit('/', 1)[1] in policy['allowed_scalar_dataset_names']:
                        if (node.shape in ((), (1,)) and node.compression is None and not node.is_virtual
                            and not node.external and node.id.get_storage_size() <= bound):
                            reads['datasets'].append(path)
                            descriptor['scalar_metadata'] = decode(node[()], bound)
                        else:
                            descriptor['scalar_metadata'] = {'undefined': 'unsupported-shape-or-storage'}
                else:
                    raise TypeError('unsupported HDF5 metadata object')
            if len(records) >= policy['maximum_structural_objects_per_file']:
                raise ValueError('structural object limit')
            records.append(descriptor)


def main():
    pins = {REG: 'c38c8254c48814b4f7785d56c2f4fe1d0f87d513695ae4f5e3f6328c95c0e21c',
        RESULT: digest(RESULT),
        Path('scripts/inventory_patchseq_append_trace_schema.py'): 'cc03841c3221f396d84c1392312367329968fb31f56a4839e8e4d4928ff10265'}
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('sealed trace metadata evidence changed')
    reg = yaml.safe_load(REG.read_text())
    report = yaml.load(RESULT.read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(p): value for p, value in report['source_hashes'].items()})
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('verified recording or source changed')
    if (report['status'] != 'all-selected-schema-metadata-outcomes-retained-independent-verification-pending'
        or report['waveform_or_timestamp_vectors_read'] is not False or report['network_requests'] != 0
        or report['specimen_eligibility_established'] is not False):
        raise ValueError('trace metadata scope differs')
    acquired = yaml.safe_load(Path(reg['acquisition']).read_text())
    slots = yaml.safe_load(Path(reg['subset']).read_text())['slots']
    if len(report['records']) != 48 or version('h5py') != reg['parser_dependency'].split('==')[1]:
        raise ValueError('source scope or registered parser differs')
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    import h5py
    summaries = []
    for expected, slot in zip(report['records'], slots, strict=True):
        same(expected['slot'], slot['slot'])
        same(expected['source_cell_specimen_id'], slot['source_cell_specimen_id'])
        if expected['status'] != 'schema-metadata-inventoried' or expected['error'] is not None:
            raise ValueError('incomplete metadata needs a distinct verification path')
        receipt_items = [r for r in acquired['receipts'] if r['slot'] == slot['slot']]
        if len(receipt_items) != 1:
            raise ValueError('slot receipt not unique')
        receipt = yaml.safe_load(Path(receipt_items[0]['path']).read_text())
        body = Path(receipt['complete_body_path'])
        if digest(body) != slot['expected_SHA256']:
            raise ValueError('complete recording changed')
        records, reads = [], {'attributes': [], 'datasets': []}
        with h5py.File(body, 'r') as file:
            reconstruct(file, reg['scope'], h5py, records, reads)
        same(records, expected['records'])
        same(reads, expected['logical_metadata_reads'])
        if expected['waveform_or_timestamp_vectors_read'] is not False or expected['specimen_eligibility_established'] is not False:
            raise ValueError('per-case metadata claim scope differs')
        if digest(body) != slot['expected_SHA256']:
            raise ValueError('recording changed during independent metadata reconstruction')
        summaries.append({'slot': slot['slot'], 'structural_objects': len(records),
            'scalar_dataset_reads': len(reads['datasets']), 'attribute_reads': len(reads['attributes'])})
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('evidence changed during reconstruction')
    OUTPUT.parent.mkdir()
    with OUTPUT.open('x') as handle:
        yaml.safe_dump({'status': 'all-selected-trace-metadata-and-access-whitelists-independently-reconstructed',
            'verifier_sha256': digest(__file__), 'source_hashes': {str(p): value for p, value in pins.items()},
            'records': summaries, 'verified_at_utc': datetime.now(UTC).isoformat(), 'metadata_collector_imported': False,
            'network_requests': 0, 'waveform_or_timestamp_vectors_read': False, 'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('All 48 trace metadata descriptors and permitted scalar reads independently reconstructed')


if __name__ == '__main__':
    main()
