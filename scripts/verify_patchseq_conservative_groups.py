"""Independent source-key projection, graph traversal and split reconstruction."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml
from verify_patchseq_whole_file_integrity import digest, same

REG = Path('docs/validation-results/post2008-patchseq-conservative-split-registration-1298.yaml')
RESULT = Path('results/patchseq-conservative-split-1299/manifest.yaml')
OUTPUT = Path('results/patchseq-conservative-split-independent-1301/manifest.yaml')


def source_projection(references, july):
    indexed = {}
    for row in july['July_rows']:
        indexed.setdefault(str(row['canonical_ids']['ephys_session_id']), []).append(row)
    records = []
    for asset in references['assets']:
        directory, filename = asset['path'].split('/')
        subject, session = filename.removesuffix('_icephys.nwb').split('_ses-')
        if not directory.startswith('sub-') or subject != directory or not filename.endswith('_icephys.nwb'):
            raise ValueError('invalid source path')
        subject = subject[4:]
        if any(not t.isascii() or not t.isdecimal() or t.startswith('0') for t in (subject, session)):
            raise ValueError('noncanonical source path tokens')
        candidates = indexed.get(session, [])
        keys = {'asset:'+asset['asset_id'], 'archive_subject:'+subject}
        unresolved = []
        if len(candidates) != 1 or asset['complete_external_file_reference_agreement'] is not True:
            unresolved.append('session-or-file-reference-not-unique-and-complete')
        for row in candidates:
            for field, namespace in [('cell_specimen_id', 'author_cell'), ('donor_id', 'author_donor')]:
                raw, canonical = row['raw_fields'][field], row['canonical_ids'][field]
                if (type(raw) is not str or not raw.isascii() or not raw.isdecimal()
                    or raw.startswith('0') or type(canonical) is not int or raw != str(canonical)):
                    unresolved.append(f'noncanonical-or-missing:{row["source_row_number"]}:{field}')
                else:
                    keys.add(namespace+':'+raw)
            if row['raw_fields']['ephys_session_id'] != session or type(row['canonical_ids']['ephys_session_id']) is not int:
                unresolved.append(f'raw-session-disagreement:{row["source_row_number"]}')
        records.append({'asset_id': asset['asset_id'], 'file_path': asset['path'], 'group_keys': sorted(keys),
            'July_candidate_source_rows': [r['source_row_number'] for r in candidates], 'unresolved': unresolved})
    return records


def graph_components(records, domain):
    graph = {}
    for record in records:
        anchor = 'asset:'+record['asset_id']
        if anchor not in record['group_keys']:
            raise ValueError('asset node absent')
        for node in record['group_keys']:
            graph.setdefault(anchor, set()).add(node)
            graph.setdefault(node, set()).add(anchor)
    unseen, node_assignment, groups = set(graph), {}, []
    while unseen:
        pending, visited = [min(unseen)], set()
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            pending.extend(graph[node]-visited)
        unseen.difference_update(visited)
        keys = sorted(visited)
        identifier = hashlib.sha256(json.dumps(keys, ensure_ascii=True, separators=(',', ':')).encode('utf-8')).hexdigest()
        hash_hex = hashlib.sha256((domain+'\x00'+identifier).encode('utf-8')).hexdigest()
        partition = 'fitting' if int(hash_hex[:16], 16) < (2*2**64)//3 else 'held-out'
        assets = sorted(node[len('asset:'):] for node in keys if node.startswith('asset:'))
        groups.append({'group_id': identifier, 'group_keys': keys, 'partition': partition, 'asset_ids': assets})
        for node in keys:
            node_assignment[node] = (identifier, partition)
    outputs = []
    for record in records:
        identifier, partition = node_assignment['asset:'+record['asset_id']]
        if any(node_assignment[node] != (identifier, partition) for node in record['group_keys']):
            raise ValueError('shared source-key leakage')
        outputs.append({**record, 'group_id': identifier, 'partition': partition, 'specimen_eligibility_established': False})
    return outputs, sorted(groups, key=lambda g: g['group_id'])


def assert_no_leakage(records):
    assignments = {}
    assets = set()
    for record in records:
        if record['asset_id'] in assets:
            raise ValueError('duplicate asset membership')
        assets.add(record['asset_id'])
        for key in record['group_keys']:
            assignment = (record['group_id'], record['partition'])
            if key in assignments and assignments[key] != assignment:
                raise ValueError('shared source-key crosses group or partition')
            assignments[key] = assignment


def main():
    pins = {REG: '7301888920c16954a390c0ba3a60f4b8ce5cd0007af4201d6ac1a94e464dd2e6',
        RESULT: 'abc6a0a9859524df090d2ab00dda7bb24404104b13a435aea8a70fc47745608d',
        Path('scripts/partition_patchseq_source_groups.py'): '3e5927312ba6dbc076adbddcf0f81836ffa266ff05e095aec7edd3a53788331e'}
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('sealed partition source changed')
    reg, report = yaml.safe_load(REG.read_text()), yaml.load(RESULT.read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(p): value for p, value in report['source_hashes'].items()})
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('registered metadata changed')
    if (report['status'] != 'conservative-source-group-partition-frozen-independent-verification-pending'
        or report['network_requests'] != 0
        or any(report[k] is not False for k in ('physical_donor_equivalence_established', 'mapping_confidence_assigned',
                                               'physiological_arrays_read', 'specimen_eligibility_established'))):
        raise ValueError('partition claim scope differs')
    references = yaml.load(Path(reg['file_references']).read_text(), Loader=yaml.CSafeLoader)
    july = yaml.load(Path(reg['July']).read_text(), Loader=yaml.CSafeLoader)
    records, groups = graph_components(source_projection(references, july), reg['partition']['domain'])
    same(records, report['records'])
    same(groups, report['groups'])
    assert_no_leakage(report['records'])
    summary = {'records': len(records), 'groups': len(groups),
        'fitting_groups': sum(g['partition'] == 'fitting' for g in groups),
        'held_out_groups': sum(g['partition'] == 'held-out' for g in groups),
        'unresolved_records': sum(bool(r['unresolved']) for r in records)}
    same(summary, report['summary'])
    if len(records) != 4435 or any(digest(p) != value for p, value in pins.items()):
        raise ValueError('source scope or hashes changed')
    OUTPUT.parent.mkdir()
    with OUTPUT.open('x') as handle:
        yaml.safe_dump({'status': 'conservative-source-groups-and-partitions-independently-reconstructed',
            'verifier_sha256': digest(__file__), 'source_hashes': {str(p): value for p, value in pins.items()},
            'summary': summary, 'all_records_groups_hashes_and_partitions_match': True,
            'shared_key_cross_partition_leakage': 0, 'verified_at_utc': datetime.now(UTC).isoformat(),
            'partition_collector_imported': False, 'network_requests': 0,
            'physical_donor_equivalence_established': False, 'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('All 4435 memberships and frozen group partitions independently verified; no shared-key leakage')


if __name__ == '__main__':
    main()
