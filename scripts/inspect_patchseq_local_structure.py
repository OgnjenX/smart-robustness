"""Sealed names-only structural continuation on a verified local binary."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import yaml
from inspect_patchseq_NWB_structure import describe
from verify_patchseq_NWB_metadata_pilot import sha

REG = Path('docs/validation-results/post2008-patchseq-local-structure-registration-1289.yaml')
REG_HASH = 'b9c85f4ca2eeb6abc07a0033251891846d650bd90e3c81c76ca49281dd4ba8c9'


def main():
    if sha(REG) != REG_HASH:
        raise ValueError('sealed local structural registration changed')
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: REG_HASH, Path(__file__): sha(__file__)}
    for key in ('independent_integrity', 'integrity_verifier', 'complete_binary', 'structural_registration', 'structural_reader'):
        pins[Path(reg[key])] = reg[key+'_sha256']
    if any(sha(p) != value for p, value in pins.items()):
        raise ValueError('registered input changed')
    verified = yaml.safe_load(Path(reg['independent_integrity']).read_text())
    if (verified['status'] != 'complete-file-author-checksum-and-cache-offsets-independently-verified'
        or verified['complete_file_SHA256'] != reg['complete_binary_sha256']
        or verified['complete_body_path'] != reg['complete_binary']):
        raise ValueError('independent content gate differs')
    pins.update({Path(p): value for p, value in verified['source_hashes'].items()})
    if any(sha(p) != value for p, value in pins.items()):
        raise ValueError('verified source changed')
    binary = Path(reg['complete_binary'])
    if binary.stat().st_size != reg['complete_binary_bytes']:
        raise ValueError('verified byte size differs')
    if version('h5py') != reg['parser_dependency'].split('==')[1]:
        raise ValueError('registered parser differs')
    policy = yaml.safe_load(Path(reg['structural_registration']).read_text())['structure_policy']
    os.environ['HDF5_PLUGIN_PRELOAD'] = reg['HDF5_plugin_preload']
    import h5py
    output = Path(reg['output'])
    output.parent.mkdir()
    records, error = [], None
    try:
        with h5py.File(binary, 'r') as file:
            describe(file, policy, h5py, records)
    except (ValueError, TypeError, OSError) as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
    unchanged = all(sha(p) == value for p, value in pins.items())
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'local-structure-complete-independent-verification-pending' if error is None else 'local-structure-incomplete',
            'source_hashes': {str(p): value for p, value in pins.items()}, 'records': records, 'error': error,
            'finished_at_utc': datetime.now(UTC).isoformat(), 'pinned_sources_unchanged': unchanged,
            'dataset_values_read': False, 'attribute_values_read': False,
            'network_requests': 0, 'new_bytes': 0, 'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('Local structural scope complete; independent reconstruction pending' if error is None else 'Local structural scope incomplete; outcome retained')


if __name__ == '__main__':
    main()
