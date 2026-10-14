"""Reconstruct local structure using the separately implemented traversal."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import yaml
from verify_patchseq_NWB_structure_ranges import reconstruct
from verify_patchseq_whole_file_integrity import digest, same

REG = Path('docs/validation-results/post2008-patchseq-local-structure-registration-1289.yaml')
RESULT = Path('results/patchseq-local-structure-1290/manifest.yaml')
OUTPUT = Path('results/patchseq-local-structure-independent-1291/manifest.yaml')


def main():
    pins = {REG: 'b9c85f4ca2eeb6abc07a0033251891846d650bd90e3c81c76ca49281dd4ba8c9',
        RESULT: '958bef541c45a8568932157ee0de815e0cde695bdf5a53d676269e0ff97626ae',
        Path('scripts/inspect_patchseq_local_structure.py'): '4cfa40444b449e354bcaa3abf864b77ab24850545c689a1c0cf330b41815255a',
        Path('scripts/verify_patchseq_NWB_structure_ranges.py'): 'fcb99fddb2a492444239f0c97a08f7613a85463bf513ea2f0c6cd5789a9e3c03',
        Path('scripts/verify_patchseq_whole_file_integrity.py'): '22e40726389126ff7a875794138145d222ed5b87515e87a2ec9bccadce78c057'}
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('pinned local structure evidence changed')
    reg, report = yaml.safe_load(REG.read_text()), yaml.safe_load(RESULT.read_text())
    pins.update({Path(p): value for p, value in report['source_hashes'].items()})
    pins[Path('scripts/verify_patchseq_NWB_metadata_pilot.py')] = 'e1d4d3482106bc56d13114a25e95ac884ff96b600851556e54e27e5a97b69087'
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('pinned source changed')
    if (report['status'] != 'local-structure-complete-independent-verification-pending'
        or report['error'] is not None or report['pinned_sources_unchanged'] is not True
        or report['network_requests'] != 0 or report['new_bytes'] != 0
        or any(report[k] is not False for k in ('dataset_values_read', 'attribute_values_read', 'specimen_eligibility_established'))):
        raise ValueError('local structural scope differs')
    policy = yaml.safe_load(Path(reg['structural_registration']).read_text())['structure_policy']
    if version('h5py') != reg['parser_dependency'].split('==')[1]:
        raise ValueError('registered parser differs')
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    import h5py
    binary = Path(reg['complete_binary'])
    if binary.stat().st_size != reg['complete_binary_bytes']:
        raise ValueError('binary size differs')
    records = []
    with h5py.File(binary, 'r') as file:
        reconstruct(file, policy, h5py, records)
    same(records, report['records'])
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('sources changed during independent reconstruction')
    OUTPUT.parent.mkdir()
    with OUTPUT.open('x') as handle:
        yaml.safe_dump({'status': 'local-structural-scope-independently-reconstructed',
            'verifier_sha256': digest(__file__), 'source_hashes': {str(p): value for p, value in pins.items()},
            'records': records, 'record_count': len(records), 'verified_at_utc': datetime.now(UTC).isoformat(),
            'structural_collector_imported': False, 'network_requests': 0, 'new_bytes': 0,
            'dataset_values_read': False, 'attribute_values_read': False,
            'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('All local structural records independently reconstructed; no values read')


if __name__ == '__main__':
    main()
