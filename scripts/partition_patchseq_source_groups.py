"""Freeze conservative metadata components; no biological identity inference."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import yaml
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-conservative-split-registration-1298.yaml')


def components(records, domain):
    parent = {}
    def root(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    for record in records:
        keys = sorted(set(record['group_keys']))
        if not keys:
            raise ValueError('empty component record')
        for key in keys:
            parent[root(key)] = root(keys[0])
    groups = {}
    for key in sorted(parent):
        groups.setdefault(root(key), []).append(key)
    identifiers, outputs = {}, []
    for keys in sorted(groups.values()):
        identifier = hashlib.sha256(json.dumps(keys, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
        token = hashlib.sha256((domain+'\0'+identifier).encode()).digest()
        partition = 'fitting' if int.from_bytes(token[:8], 'big') < (2*(1 << 64))//3 else 'held-out'
        outputs.append({'group_id': identifier, 'group_keys': keys, 'partition': partition,
                        'asset_ids': sorted(r['asset_id'] for r in records if root(r['group_keys'][0]) == root(keys[0]))})
        for key in keys:
            identifiers[key] = (identifier, partition)
    assigned = [{**r, 'group_id': identifiers[r['group_keys'][0]][0], 'partition': identifiers[r['group_keys'][0]][1],
                 'specimen_eligibility_established': False} for r in records]
    return assigned, sorted(outputs, key=lambda g: g['group_id'])


def project(references, july):
    records = []
    for asset in references['assets']:
        keys, unresolved = ['asset:'+asset['asset_id']], []
        path = asset['path']
        match = re.fullmatch(r'sub-([1-9][0-9]*)/sub-([1-9][0-9]*)_ses-([1-9][0-9]*)_icephys\.nwb', path)
        if match is None or match[1] != match[2]:
            raise ValueError('invalid registered source path')
        keys.append('archive_subject:'+match[1])
        rows = [r for r in july['July_rows'] if str(r['canonical_ids']['ephys_session_id']) == match[3]]
        if len(rows) != 1 or asset['complete_external_file_reference_agreement'] is not True:
            unresolved.append('session-or-file-reference-not-unique-and-complete')
        for row in rows:
            for field, namespace in [('cell_specimen_id', 'author_cell'), ('donor_id', 'author_donor')]:
                raw, canonical = row['raw_fields'][field], row['canonical_ids'][field]
                if type(raw) is not str or re.fullmatch('[1-9][0-9]*', raw) is None or type(canonical) is not int or raw != str(canonical):
                    unresolved.append(f'noncanonical-or-missing:{row["source_row_number"]}:{field}')
                else:
                    keys.append(namespace+':'+raw)
            raw = row['raw_fields']['ephys_session_id']
            if raw != match[3] or type(row['canonical_ids']['ephys_session_id']) is not int:
                unresolved.append(f'raw-session-disagreement:{row["source_row_number"]}')
        records.append({'asset_id': asset['asset_id'], 'file_path': path, 'group_keys': sorted(set(keys)),
            'July_candidate_source_rows': [r['source_row_number'] for r in rows], 'unresolved': unresolved})
    return records


def main():
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: digest(REG), Path(__file__): digest(__file__)}
    for key in ('file_references', 'independent_references', 'July'):
        pins[Path(reg[key])] = reg[key+'_sha256']
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('registered metadata changed')
    data = {key: yaml.load(Path(reg[key]).read_text(), Loader=yaml.CSafeLoader)
            for key in ('file_references', 'independent_references', 'July')}
    prior = data['independent_references']
    if prior['status'] != 'all-external-file-reference-comparisons-independently-reconstructed':
        raise ValueError('independent file-reference prerequisite missing')
    if prior['source_hashes'].get(reg['file_references']) != reg['file_references_sha256']:
        raise ValueError('independent reference covers a different source')
    pins.update({Path(p): value for p, value in prior['source_hashes'].items()})
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('independently checked source changed')
    records = project(data['file_references'], data['July'])
    if len(records) != 4435 or len({r['asset_id'] for r in records}) != 4435:
        raise ValueError('full source catalog scope differs')
    records, groups = components(records, reg['partition']['domain'])
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('metadata changed during partition')
    output = Path(reg['output'])
    output.parent.mkdir()
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'conservative-source-group-partition-frozen-independent-verification-pending',
            'source_hashes': {str(p): value for p, value in pins.items()}, 'records': records, 'groups': groups,
            'summary': {'records': len(records), 'groups': len(groups),
                'fitting_groups': sum(g['partition'] == 'fitting' for g in groups),
                'held_out_groups': sum(g['partition'] == 'held-out' for g in groups),
                'unresolved_records': sum(bool(r['unresolved']) for r in records)},
            'finished_at_utc': datetime.now(UTC).isoformat(), 'network_requests': 0,
            'physical_donor_equivalence_established': False, 'mapping_confidence_assigned': False,
            'physiological_arrays_read': False, 'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('Conservative source groups frozen; no physiology cohort qualified')


if __name__ == '__main__':
    main()
