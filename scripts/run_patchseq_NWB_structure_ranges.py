"""Sealed bounded retrieval for names-only HDF5 metadata discovery."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import yaml
from inspect_patchseq_NWB_structure import describe
from patchseq_bounded_NWB import BoundedReader, select_url
from run_patchseq_NWB_metadata_pilot import Transport, sha

REG = Path('docs/validation-results/post2008-patchseq-NWB-structure-range-registration-1274.yaml')


def seed(reader, pilot, expected):
    blocks = {}
    width = reader.limits['cache_block_bytes']
    for receipt in pilot['HTTP_receipts']:
        if receipt['kind'] != 'RANGE':
            continue
        path = Path(receipt['body_path'])
        data = path.read_bytes()
        start, end = receipt['start'], receipt['end']
        if (sha(path) != receipt['body_sha256'] or start % width or len(data) != end-start+1
            or end != min(start+width, reader.size)-1 or start//width in blocks):
            raise ValueError('invalid or changed seed block')
        blocks[start//width] = data
    if len(blocks) != expected['blocks'] or sum(len(b) for b in blocks.values()) != expected['bytes']:
        raise ValueError('seed inventory differs')
    reader.cache.update(blocks)


def main():
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: sha(REG), Path(__file__): sha(__file__)}
    for key in ('pilot', 'independent_replay', 'prior_structure', 'structural_registration',
                'structure_runner', 'range_reader', 'transport_runner', 'offline_reader'):
        pins[Path(reg[key])] = reg[key+'_sha256']
    if any(sha(path) != digest for path, digest in pins.items()):
        raise ValueError('registered input changed')
    replay = yaml.load(Path(reg['independent_replay']).read_text(), Loader=yaml.CSafeLoader)
    if replay['status'] != 'bounded-NWB-ranges-and-metadata-independently-replayed-offline':
        raise ValueError('independent pilot prerequisite missing')
    pins.update({Path(path): digest for path, digest in replay['source_hashes'].items()})
    if any(sha(path) != digest for path, digest in pins.items()):
        raise ValueError('independently checked input changed')
    pilot = yaml.load(Path(reg['pilot']).read_text(), Loader=yaml.CSafeLoader)
    prior = yaml.safe_load(Path(reg['prior_structure']).read_text())
    if prior['status'] != 'offline-structure-incomplete' or prior['asset'] != pilot['asset']:
        raise ValueError('prior terminal structural outcome differs')
    if version('h5py') != reg['parser_dependency'].split('==')[1]:
        raise ValueError('registered parser differs')
    original = Path('results/patchseq-archive-identity-metadata-1239')/pilot['asset']['asset_id']/'response.json'
    if original not in pins:
        raise ValueError('original locator not pinned')
    url, _ = select_url(json.loads(original.read_bytes()), reg['URL_resolution'])
    policy = yaml.safe_load(Path(reg['structural_registration']).read_text())['structure_policy']
    root = Path(reg['output_directory'])
    root.mkdir()  # Exclusive: no duplicate/restart/overwrite of a terminal run.
    transport = Transport(root, url, pilot['asset'], reg)
    stream = BoundedReader(pilot['asset']['size'], reg['HTTP_range_limits'], transport.fetch, transport.check_deadline)
    report = {'status': 'bounded-structural-acquisition-incomplete', 'asset': pilot['asset'],
              'source_hashes': {str(path): digest for path, digest in pins.items()},
              'started_at_utc': datetime.now(UTC).isoformat(), 'records': [],
              'dataset_values_read': False, 'attribute_values_read': False,
              'physiological_arrays_read': False, 'specimen_eligibility_established': False}
    try:
        seed(stream, pilot, reg['seed_cache'])
        etags = {r['header_fields']['etag'][0] for r in pilot['HTTP_receipts'] if r['kind'] == 'RANGE'}
        if len(etags) != 1:
            raise ValueError('seed ETag ambiguous')
        transport.head()
        if transport.etag != etags.pop():
            raise ValueError('object changed since pilot')
        os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
        import h5py
        with h5py.File(stream, 'r') as file:
            describe(file, policy, h5py, report['records'])
        transport.check_deadline()
        report['status'] = 'bounded-structural-acquisition-complete-independent-replay-pending'
    except (OSError, ValueError, TypeError, KeyError, TimeoutError, subprocess.TimeoutExpired) as exc:
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        stream.close()
        report.update(HTTP_receipts=transport.records, new_range_requests=stream.requests,
                      charged_new_range_bytes=stream.downloaded,
                      combined_cache_bytes=sum(len(b) for b in stream.cache.values()),
                      finished_at_utc=datetime.now(UTC).isoformat(),
                      pinned_sources_unchanged=all(sha(p) == digest for p, digest in pins.items()))
        with (root/'manifest.yaml').open('x') as handle:
            yaml.safe_dump(report, handle, sort_keys=False)
    print(report['status'])


if __name__ == '__main__':
    main()
