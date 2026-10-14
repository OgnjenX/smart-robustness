"""Independent hard-link scalar reconstruction from immutable offline cache."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import yaml
from verify_patchseq_NWB_metadata_pilot import SavedRanges, equal, sha

RESULT = Path('results/patchseq-NWB-general-session-1280/manifest.yaml')
OUTPUT = Path('results/patchseq-NWB-general-session-independent-1281/manifest.yaml')


def independent_scalar(file, h5py):
    node = file
    for name in ('general', 'session_id'):
        if not isinstance(node, h5py.Group):
            raise TypeError('identity parent is not a group')
        link = node.get(name, getlink=True)
        if link is None:
            return {'present': False}
        if type(link) is not h5py.HardLink:
            raise TypeError('identity path not hard linked')
        node = node[name]
    if not isinstance(node, h5py.Dataset):
        raise TypeError('identity object not a dataset')
    if (node.shape not in ((), (1,)) or node.compression is not None or node.is_virtual
        or node.external or node.id.get_storage_size() > 4096):
        raise ValueError('identity storage unapproved')
    value = node[()]
    if hasattr(value, 'shape'):
        value = value.item()
    if type(value) is bytes:
        if len(value) > 4096:
            raise ValueError('identity decoded size exceeded')
        scalar = {'raw_type': 'bytes', 'raw_hex': value.hex(), 'decoded_text': value.decode('utf-8')}
    elif type(value) is str:
        if len(value.encode()) > 4096:
            raise ValueError('identity decoded size exceeded')
        scalar = {'raw_type': 'str', 'value': value}
    else:
        raise TypeError('independent check requires text; no numeric conversion')
    return {'present': True, 'shape': list(node.shape), 'dtype': str(node.dtype), 'scalar': scalar}


def main():
    pins = {RESULT: '871b1cfda6560119d096c3400487464d2d4d99e4bc4332b941fd82fd4c8ceb3e',
            Path('scripts/read_patchseq_NWB_general_session.py'): '74d7e1122e0cfccce0a5c107d83d8cc6edef113b45c637bd9fbaf0afce07b2e5'}
    if any(sha(p) != digest for p, digest in pins.items()):
        raise ValueError('scalar evidence changed')
    report = yaml.safe_load(RESULT.read_text())
    pins.update({Path(p): digest for p, digest in report['source_hashes'].items()})
    if any(sha(p) != digest for p, digest in pins.items()):
        raise ValueError('cache source changed')
    if (report['status'] != 'cached-session-read-complete-independent-verification-pending'
        or report['network_requests'] != 0 or report['new_bytes'] != 0
        or report['pinned_sources_unchanged'] is not True or report['physiological_arrays_read'] is not False
        or report['specimen_eligibility_established'] is not False):
        raise ValueError('scalar scope differs')
    ranges = []
    for path in (Path('results/patchseq-NWB-metadata-pilot-1267/manifest.yaml'),
                 Path('results/patchseq-NWB-structure-range-1275/manifest.yaml')):
        if path not in pins:
            raise ValueError('manifest not pinned')
        source = yaml.safe_load(path.read_text())
        equal(source['asset'], report['asset'])
        for receipt in source['HTTP_receipts']:
            if receipt['kind'] == 'RANGE':
                body = Path(receipt['body_path'])
                if body not in pins or sha(body) != pins[body]:
                    raise ValueError('body not independently pinned')
                ranges.append((receipt['start'], body.read_bytes()))
    if len(ranges) != 38 or sum(len(b) for _, b in ranges) != 2490368:
        raise ValueError('cache inventory differs')
    if version('h5py') != '3.12.1':
        raise ValueError('registered parser differs')
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    import h5py
    with SavedRanges(report['asset']['size'], ranges) as stream:
        with h5py.File(stream, 'r') as file:
            record = independent_scalar(file, h5py)
        spans = stream.read_spans
    inspection = {'root_attributes': {}, 'datasets': {'/general/session_id': record},
                  'logical_dataset_reads': ['/general/session_id'] if record['present'] else []}
    equal(inspection, report['inspection'])
    expected = report['asset']['path'].split('_ses-')[1].removesuffix('_icephys.nwb')
    scalar = record.get('scalar', {})
    value = scalar.get('decoded_text') if scalar.get('raw_type') == 'bytes' else scalar.get('value')
    match = value == expected if value is not None else None
    equal(match, report['literal_path_session_match'])
    if any(sha(p) != digest for p, digest in pins.items()):
        raise ValueError('input changed during verification')
    OUTPUT.parent.mkdir()
    with OUTPUT.open('x') as handle:
        yaml.safe_dump({'status': 'general-session-scalar-independently-reconstructed',
            'verifier_sha256': sha(__file__), 'source_hashes': {str(p): digest for p, digest in pins.items()},
            'inspection': inspection, 'literal_path_session_match': match, 'offline_read_spans': spans,
            'verified_at_utc': datetime.now(UTC).isoformat(), 'scalar_collector_imported': False,
            'network_requests': 0, 'new_bytes': 0, 'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('Candidate session scalar independently reconstructed from cached bytes')


if __name__ == '__main__':
    main()
