"""Independently reconstruct candidates using relational source-row joins."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import yaml
from verify_patchseq_whole_file_integrity import digest, same

REG = Path('docs/validation-results/post2008-patchseq-source-candidate-registration-1303.yaml')
RESULT = Path('results/patchseq-source-candidates-1304/manifest.yaml')
OUTPUT = Path('results/patchseq-source-candidates-independent-1306/manifest.yaml')


def reasons_for(reg, specimen, annotation, frozen):
    candidates = annotation['methods_candidates']
    raw = candidates[0]['raw_fields'] if len(candidates) == 1 else None
    policy = reg['source_label_predicates']
    checks = [
        ('source-group-not-target', specimen['source_core_inhibitory_group'] in reg['target_groups']),
        ('anatomy-label-not-target', specimen['raw_fields']['structure'] in reg['target_anatomy_labels']),
        ('source-reference-leaf-not-agreeing', specimen['one_agreeing_source_reference_leaf'] is True),
        ('methods-cell-association-not-unique', raw is not None and annotation['unique_cell_ID_annotation_association'] is True),
        ('literal-anatomy-agreement-not-established', annotation['literal_structure_agrees'] is True),
        ('raw-source-label-not-required:class', raw is not None and raw['class'] == policy['raw_methods_class']),
        ('raw-source-label-not-required:Gouwens_etal_2020', raw is not None and raw['Gouwens_etal_2020'] == policy['raw_methods_Gouwens_etal_2020']),
        ('raw-source-label-not-required:mapping_quality', raw is not None and raw['mapping_quality'] == policy['raw_methods_mapping_quality']),
        ('frozen-membership-not-unique-and-resolved', frozen is not None and not frozen['unresolved']),
    ]
    return [name for name, passed in checks if not passed], raw


def reconstruct(reg, ledger, coherence, split, association):
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE specimens (pos INTEGER, number INTEGER PRIMARY KEY, payload TEXT)')
        db.execute('CREATE TABLE annotations (number INTEGER PRIMARY KEY, payload TEXT)')
        db.execute('CREATE TABLE memberships (number INTEGER, payload TEXT)')
        db.execute('CREATE INDEX membership_number ON memberships(number)')
        for pos, row in enumerate(ledger['specimens']):
            db.execute('INSERT INTO specimens VALUES (?,?,?)', (pos, row['July_source_row_number'], json.dumps(row)))
        for row in coherence['records']:
            db.execute('INSERT INTO annotations VALUES (?,?)', (row['July_source_row_number'], json.dumps(row)))
        for row in split['records']:
            for number in row['July_candidate_source_rows']:
                db.execute('INSERT INTO memberships VALUES (?,?)', (number, json.dumps(row)))
        if list(db.execute('SELECT number FROM specimens EXCEPT SELECT number FROM annotations')) or list(db.execute('SELECT number FROM annotations EXCEPT SELECT number FROM specimens')):
            raise ValueError('complete source-row joins missing')
        outputs = []
        for number, left, right in db.execute('SELECT s.number,s.payload,a.payload FROM specimens s JOIN annotations a USING(number) ORDER BY s.pos'):
            specimen, annotation = json.loads(left), json.loads(right)
            cell = specimen['canonical_ids']['cell_specimen_id']
            if type(cell) is not int or cell <= 0:
                raise ValueError('noncanonical source cell')
            same(cell, annotation['canonical_cell_id'])
            same(specimen['raw_fields'], annotation['July_raw_fields'])
            matches = [json.loads(r[0]) for r in db.execute('SELECT payload FROM memberships WHERE number=?', (number,))]
            frozen = matches[0] if len(matches) == 1 else None
            reasons, raw = reasons_for(reg, specimen, annotation, frozen)
            if frozen is not None and 'author_cell:'+str(cell) not in frozen['group_keys']:
                raise ValueError('frozen cell key differs')
            provenance = None
            if str(cell) == association['source_cell_specimen_id']:
                if number != association['July_source_row_number'] or frozen is None or frozen['file_path'] != association['file_path']:
                    raise ValueError('pilot provenance differs')
                provenance = True
            outputs.append({'July_source_row_number': number, 'source_cell_specimen_id': str(cell),
                'source_core_group': specimen['source_core_inhibitory_group'],
                'raw_anatomy_label': specimen['raw_fields']['structure'],
                'target_layer_label': reg['target_anatomy_labels'].get(specimen['raw_fields']['structure']),
                'raw_author_labels': None if raw is None else {k: raw[k] for k in ('class', 'Gouwens_etal_2020', 'mapping_quality')},
                'source_annotation_candidate': not reasons, 'candidate_failures_or_unknowns': reasons,
                'frozen_group_id': None if frozen is None else frozen['group_id'],
                'frozen_partition': None if frozen is None else frozen['partition'],
                'file_path': None if frozen is None else frozen['file_path'],
                'recording_content_session_author_cell_provenance_verified': provenance,
                'prior_strict_cell_sample_correspondence': annotation['prior_strict_cell_sample_correspondence'],
                'prior_specimen_name_bridge': annotation['prior_specimen_name_bridge'],
                'RNA_alias_equivalence_established': False, 'mapping_confidence_assigned': False,
                'isolated_physiology_validated': False, 'specimen_eligibility_established': False})
        db.execute('CREATE TABLE targets (source_group TEXT, layer TEXT, partition TEXT, group_id TEXT)')
        db.execute('CREATE TABLE reasons (reason TEXT)')
        for row in outputs:
            if row['source_annotation_candidate']:
                db.execute('INSERT INTO targets VALUES (?,?,?,?)', (row['source_core_group'], row['target_layer_label'], row['frozen_partition'], row['frozen_group_id']))
            db.executemany('INSERT INTO reasons VALUES (?)', [(r,) for r in row['candidate_failures_or_unknowns']])
        strata = []
        for group in reg['target_groups']:
            for layer in sorted(set(reg['target_anatomy_labels'].values())):
                for partition in ('fitting', 'held-out'):
                    count, groups = db.execute('SELECT count(*),count(DISTINCT group_id) FROM targets WHERE source_group=? AND layer=? AND partition=?', (group, layer, partition)).fetchone()
                    strata.append({'source_group': group, 'anatomy_layer_label': layer, 'partition': partition,
                        'candidate_records': count, 'conservative_source_groups': groups})
        summary = {'records': len(outputs), 'source_annotation_candidates': db.execute('SELECT count(*) FROM targets').fetchone()[0],
            'failure_reason_counts_nonexclusive': dict(db.execute('SELECT reason,count(*) FROM reasons GROUP BY reason ORDER BY reason')),
            'qualified_specimens': 0}
    return {'records': outputs, 'strata': strata, 'summary': summary}


def main():
    pins = {REG: '6054b152fa94dcfcf5470334cfbc9c3273ec0fbf2120117b7fe74cb5aaf40816',
        RESULT: '60b967d96923bec2bdf86c6ee841f74ea5913627fee34cccc32f6602610bb837',
        Path('scripts/build_patchseq_source_candidates.py'): 'f873aedae190cd4f79185526485fc3fefbbae7bad6324e2905b36aafdefd9893'}
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('sealed candidate evidence changed')
    reg = yaml.safe_load(REG.read_text())
    report = yaml.load(RESULT.read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(p): value for p, value in report['source_hashes'].items()})
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('candidate input changed')
    if (report['status'] != 'source-annotated-candidates-built-independent-verification-pending'
        or report['physiology_accessed'] is not False or report['specimen_eligibility_established'] is not False
        or report['network_requests'] != 0):
        raise ValueError('candidate claim scope differs')
    data = {k: yaml.load(Path(v['path']).read_text(), Loader=yaml.CSafeLoader) for k, v in reg['sources'].items()}
    expected = reconstruct(reg, data['source_ledger'], data['coherence'], data['split'], data['pilot_association']['association'])
    for key in ('records', 'strata', 'summary'):
        same(expected[key], report[key])
    if len(expected['records']) != 4435 or any(digest(p) != value for p, value in pins.items()):
        raise ValueError('scope or sources changed')
    OUTPUT.parent.mkdir()
    with OUTPUT.open('x') as handle:
        yaml.safe_dump({'status': 'all-source-candidate-records-and-strata-independently-reconstructed',
            'verifier_sha256': digest(__file__), 'source_hashes': {str(p): value for p, value in pins.items()},
            'summary': expected['summary'], 'strata': expected['strata'],
            'verified_at_utc': datetime.now(UTC).isoformat(), 'candidate_collector_imported': False,
            'network_requests': 0, 'physiology_accessed': False, 'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('Every source candidate, failure reason, frozen partition and stratum independently reconstructed')


if __name__ == '__main__':
    main()
