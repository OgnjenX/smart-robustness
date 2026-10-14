"""Sealed cached-only read of one preregistered session metadata scalar."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import yaml
from verify_patchseq_NWB_metadata_pilot import SavedRanges, replay, sha

REG = Path('docs/validation-results/post2008-patchseq-NWB-general-session-registration-1279.yaml')
REG_HASH = '63ee2640e1122e1b8c9575ce2d633665092e04fd121f01eaa2041f54aed2793c'


def comparison(entry, expected):
    scalar = entry.get('scalar', {})
    text = (scalar.get('decoded_text') if scalar.get('raw_type') == 'bytes'
            else scalar.get('value') if scalar.get('raw_type') == 'str' else None)
    return text == expected if text is not None else None


def main():
    if sha(REG) != REG_HASH:
        raise ValueError('sealed scalar registration changed')
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: REG_HASH, Path(__file__): sha(__file__)}
    for key in ('independent_structure', 'independent_verifier', 'offline_reader'):
        pins[Path(reg[key])] = reg[key+'_sha256']
    if any(sha(p) != digest for p, digest in pins.items()):
        raise ValueError('registered prerequisite changed')
    verified = yaml.safe_load(Path(reg['independent_structure']).read_text())
    if verified['status'] != 'partial-structure-and-range-receipts-independently-verified':
        raise ValueError('independent structure gate missing')
    pins.update({Path(p): digest for p, digest in verified['source_hashes'].items()})
    if any(sha(p) != digest for p, digest in pins.items()):
        raise ValueError('previously verified input changed')
    manifests = [Path('results/patchseq-NWB-metadata-pilot-1267/manifest.yaml'),
                 Path('results/patchseq-NWB-structure-range-1275/manifest.yaml')]
    if any(p not in pins for p in manifests):
        raise ValueError('range manifests not independently pinned')
    inputs = [yaml.safe_load(p.read_text()) for p in manifests]
    if inputs[0]['asset'] != inputs[1]['asset']:
        raise ValueError('cache asset mismatch')
    ranges = []
    for report in inputs:
        for receipt in report['HTTP_receipts']:
            if receipt['kind'] == 'RANGE':
                path = Path(receipt['body_path'])
                if path not in pins or pins[path] != receipt['body_sha256']:
                    raise ValueError('cache block not independently verified')
                ranges.append((receipt['start'], path.read_bytes()))
    if len(ranges) != 38 or sum(len(data) for _, data in ranges) != 2490368:
        raise ValueError('registered cache inventory differs')
    if version('h5py') != reg['parser_dependency'].split('==')[1]:
        raise ValueError('registered parser differs')
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    import h5py
    output = Path(reg['output'])
    output.parent.mkdir()
    result = {'status': 'cached-session-read-incomplete', 'asset': inputs[0]['asset'],
        'source_hashes': {str(p): digest for p, digest in pins.items()},
        'network_requests': 0, 'new_bytes': 0, 'physiological_arrays_read': False,
        'specimen_eligibility_established': False, 'literal_path_session_match': None}
    with SavedRanges(inputs[0]['asset']['size'], ranges) as stream:
        try:
            inspection = replay(stream, reg['logical_metadata_reads'], h5py)
            result['inspection'] = inspection
            expected = inputs[0]['asset']['path'].split('_ses-')[1].removesuffix('_icephys.nwb')
            result['literal_path_session_match'] = comparison(inspection['datasets']['/general/session_id'], expected)
            result['status'] = 'cached-session-read-complete-independent-verification-pending'
        except (ValueError, TypeError, OSError) as exc:
            result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        result['offline_read_spans'] = stream.read_spans
    result['pinned_sources_unchanged'] = all(sha(p) == digest for p, digest in pins.items())
    result['finished_at_utc'] = datetime.now(UTC).isoformat()
    with output.open('x') as handle:
        yaml.safe_dump(result, handle, sort_keys=False)
    print(result['status'])


if __name__ == '__main__':
    main()
