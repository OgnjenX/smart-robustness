"""Discover metadata structure from saved ranges; never read dataset values."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import yaml
from verify_patchseq_NWB_metadata_pilot import SavedRanges, sha

REG = Path('docs/validation-results/post2008-patchseq-NWB-structure-registration-1271.yaml')


def describe(file, policy, h5py, records):
    def bounded(names):
        names = sorted(names)
        if any(len(name.encode('utf-8')) > policy['maximum_name_utf8_bytes'] for name in names):
            raise ValueError('structural name bound exceeded')
        return names

    seen = {}

    def append(record):
        if len(records) >= policy['maximum_unique_object_paths']:
            raise ValueError('structural record bound exceeded')
        records.append(record)

    def inspect(path, obj, depth):
        if path in seen and seen[path] <= depth:
            return
        first = path not in seen
        seen[path] = depth
        attrs = bounded(obj.attrs.keys())
        if len(attrs) > policy['maximum_attribute_names_per_object']:
            raise ValueError('attribute name count exceeded')
        record = {'path': path, 'link_type': 'HardLink', 'attribute_names': attrs}
        if isinstance(obj, h5py.Dataset):
            record.update(object_type='dataset', dataset_shape=None if obj.shape is None else list(obj.shape),
                          dataset_dtype=str(obj.dtype))
        elif isinstance(obj, h5py.Group):
            record['object_type'] = 'group'
        else:
            raise TypeError('unsupported metadata object')
        if first:
            append(record)
        if isinstance(obj, h5py.Group) and depth < policy['maximum_descendant_depth_per_root']:
            for name in bounded(obj.keys()):
                child = path.rstrip('/')+'/'+name
                link = obj.get(name, getlink=True)
                if name in policy['excluded_descent'] or type(link) is not h5py.HardLink:
                    append({'path': child, 'link_type': type(link).__name__, 'dereferenced': False})
                    continue
                inspect(child, obj[name], depth+1)

    # Root children are names/link descriptors only. Do not open data containers.
    for name in bounded(file.keys()):
        append({'path': '/'+name, 'link_type': type(file.get(name, getlink=True)).__name__,
                        'root_name_only': True})
    for path in sorted(policy['inspect_metadata_roots']):
        obj, present = file, True
        for name in path.strip('/').split('/'):
            if not isinstance(obj, h5py.Group):
                raise TypeError('metadata root parent is not a group')
            link = obj.get(name, getlink=True)
            if link is None:
                append({'path': path, 'present': False})
                present = False
                break
            if type(link) is not h5py.HardLink:
                append({'path': path, 'link_type': type(link).__name__, 'dereferenced': False})
                present = False
                break
            obj = obj[name]
        if present:
            inspect(path, obj, 0)
    if len(records) > policy['maximum_unique_object_paths']:
        raise ValueError('structural record bound exceeded')


def main():
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: sha(REG), Path(__file__): sha(__file__)}
    for key in ('pilot_result', 'independent_replay', 'offline_reader'):
        pins[Path(reg[key])] = reg[key+'_sha256']
    if any(sha(p) != digest for p, digest in pins.items()):
        raise ValueError('registered input hash differs')
    pilot = yaml.load(Path(reg['pilot_result']).read_text(), Loader=yaml.CSafeLoader)
    replay = yaml.load(Path(reg['independent_replay']).read_text(), Loader=yaml.CSafeLoader)
    if replay['status'] != 'bounded-NWB-ranges-and-metadata-independently-replayed-offline':
        raise ValueError('independent offline gate missing')
    pins.update({Path(p): digest for p, digest in replay['source_hashes'].items()})
    if any(sha(p) != digest for p, digest in pins.items()):
        raise ValueError('independently verified source changed')
    ranges = []
    for receipt in pilot['HTTP_receipts']:
        if receipt['kind'] == 'RANGE':
            # Receipt spans are independently checked by the pinned replay.
            ranges.append((receipt['start'], Path(receipt['body_path']).read_bytes()))
    if len(ranges) != replay['range_count'] or sum(len(b) for _, b in ranges) != replay['range_bytes']:
        raise ValueError('saved range inventory differs')
    if version('h5py') != reg['parser_dependency'].split('==')[1]:
        raise ValueError('registered parser version differs')
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    import h5py
    output = Path(reg['output'])
    if output.parent.exists():
        raise FileExistsError('structural outcome already exists; no overwrite or rerun')
    output.parent.mkdir()
    records, error, spans = [], None, []
    with SavedRanges(pilot['asset']['size'], ranges) as stream:
        try:
            with h5py.File(stream, 'r') as file:
                describe(file, reg['structure_policy'], h5py, records)
        except (ValueError, TypeError, OSError) as exc:
            error = {'type': type(exc).__name__, 'message': str(exc)}
        spans = stream.read_spans
    if any(sha(p) != digest for p, digest in pins.items()):
        raise ValueError('source changed during structural inspection')
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'offline-structure-complete' if error is None else 'offline-structure-incomplete',
            'source_hashes': {str(p): digest for p, digest in pins.items()}, 'asset': pilot['asset'],
            'records': records, 'error': error, 'offline_read_spans': spans,
            'completed_at_utc': datetime.now(UTC).isoformat(), 'network_requests': 0,
            'dataset_values_read': False, 'attribute_values_read': False,
            'physiological_arrays_read': False, 'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('Offline structural inspection complete' if error is None else 'Offline structural inspection incomplete; no fallback')


if __name__ == '__main__':
    main()
