"""Independent relational subset ranking and literal author-anchor preflight."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import yaml
from verify_patchseq_author_cell_association import positive_lexeme
from verify_patchseq_whole_file_integrity import digest, same

REG = Path('docs/validation-results/post2008-patchseq-intrinsic-subset-registration-1308.yaml')
RESULT = Path('results/patchseq-intrinsic-subset-1309/manifest.yaml')
OUTPUT = Path('results/patchseq-intrinsic-subset-independent-1311/manifest.yaml')


def ranked_subset(reg, candidates):
    slots, strata = [], []
    selection = reg['selection']
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE targets (pos INTEGER, class TEXT, layer TEXT, partition TEXT, gid TEXT, cell INTEGER, path TEXT, payload TEXT)')
        for pos, row in enumerate(candidates['records']):
            if row['source_annotation_candidate'] is not True:
                continue
            cell = row['source_cell_specimen_id']
            if type(cell) is not str or not cell.isascii() or not cell.isdecimal() or cell.startswith('0'):
                raise ValueError('invalid source-cell rank key')
            db.execute('INSERT INTO targets VALUES (?,?,?,?,?,?,?,?)', (pos, row['source_core_group'], row['target_layer_label'],
                row['frozen_partition'], row['frozen_group_id'], int(cell), row['file_path'], json.dumps(row)))
        for group in selection['source_groups']:
            for layer in selection['source_layer_labels']:
                for partition, count in [('fitting', selection['fitting_groups_per_stratum']), ('held-out', selection['held_out_groups_per_stratum'])]:
                    keys = [r[0] for r in db.execute('SELECT DISTINCT gid FROM targets WHERE class=? AND layer=? AND partition=? ORDER BY gid COLLATE BINARY', (group, layer, partition))]
                    if len(keys) < count:
                        raise ValueError('independent stratum support insufficient')
                    chosen = keys[:count]
                    strata.append({'source_group': group, 'layer_label': layer, 'partition': partition,
                        'available_groups': len(keys), 'required_groups': count, 'selected_group_ids': chosen})
                    for rank, key in enumerate(chosen, 1):
                        matches = list(db.execute('SELECT payload FROM targets WHERE class=? AND layer=? AND partition=? AND gid=? ORDER BY cell,path COLLATE BINARY,pos', (group, layer, partition, key)))
                        row = json.loads(matches[0][0])
                        slots.append({'slot': len(slots)+1, 'source_group': group, 'layer_label': layer,
                            'partition': partition, 'group_rank_in_stratum': rank, 'frozen_group_id': key,
                            'source_cell_specimen_id': row['source_cell_specimen_id'], 'July_source_row_number': row['July_source_row_number'],
                            'file_path': row['file_path'], 'preflight_failures': [], 'specimen_eligibility_established': False})
    return slots, strata


def reconcile(reg, candidates, references, manifest):
    slots, strata = ranked_subset(reg, candidates)
    total, partitions = 0, {}
    for slot in slots:
        previous = partitions.setdefault(slot['frozen_group_id'], slot['partition'])
        if previous != slot['partition']:
            raise ValueError('selected group crosses partitions')
        refs = [r for r in references['assets'] if r['path'] == slot['file_path']]
        if len(refs) != 1 or refs[0]['complete_external_file_reference_agreement'] is not True or len(refs[0]['manifest_candidates']) != 1:
            raise ValueError('author reference ambiguous or incomplete')
        reference = refs[0]
        number = reference['manifest_candidates'][0]['manifest_source_row_number']
        rows = [r for r in manifest['rows'] if r['source_row_number'] == number]
        if len(rows) != 1:
            raise ValueError('typed author source row not unique')
        row, cells = rows[0], rows[0]['cells']
        cell, size_text = positive_lexeme(cells['cell_specimen_id']), positive_lexeme(cells['file_size'])
        size = int(size_text)
        if cell != slot['source_cell_specimen_id']:
            raise ValueError('author/source-cell identity disagrees')
        same(int(cell), row['canonical_integers']['cell_specimen_id'])
        same(size, row['canonical_integers']['file_size'])
        if any(cells[k] is None or cells[k]['formula_present'] for k in ('file_name', 'sha256_checksum')):
            raise ValueError('filename/hash absent or formula')
        checksum = cells['sha256_checksum']['literal_text']
        if (cells['file_name']['literal_text'] != slot['file_path'].split('/')[-1]
            or type(checksum) is not str or len(checksum) != 64 or any(c not in '0123456789abcdefABCDEF' for c in checksum)
            or size > reg['preflight']['maximum_single_recording_bytes']):
            raise ValueError('literal author anchor or size limit differs')
        same(reference['July_session_source_row_numbers'], [slot['July_source_row_number']])
        slot.update(asset_reference_candidates=[reference['asset_id']], asset_id=reference['asset_id'],
            author_manifest_source_row_candidates=[number], expected_recording_bytes=size,
            expected_SHA256=checksum, author_manifest_source_row_number=number)
        total += size
    paths = {s['file_path'] for s in slots}
    if (len(slots) != reg['selection']['minimum_required_selection_slots'] or len(paths) != len(slots)
        or len(paths) > reg['selection']['maximum_unique_recordings']
        or total > reg['preflight']['maximum_unique_selected_recording_bytes']):
        raise ValueError('selection count or aggregate budget differs')
    return {'slots': slots, 'strata': strata, 'problems': [], 'summary': {'selection_slots': len(slots),
        'unique_recordings': len(paths), 'unique_source_groups': len(partitions),
        'known_selected_recording_bytes': total, 'slot_preflight_failures': 0, 'qualified_specimens': 0}}


def main():
    pins = {REG: '8e26374dbd7a3e0876b18d85c2ff55854668ac7c43fb61f4c05e14777f772de7',
        RESULT: 'dcf2bdf39e44c82190b63aa62de46df84314ffec47f76e01e85726cdbd9088d3',
        Path('scripts/select_patchseq_intrinsic_subset.py'): '44f8bfe95110bcee15098ab4f32fbbc8471c9ade4fb2b98560a5acf58ec8adb8',
        Path('scripts/verify_patchseq_author_cell_association.py'): 'f80c618ae7ed6639c97467f57aa7b8648820b65f31fcb8835853933ff7f379a4'}
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('sealed subset or independent numeric parser changed')
    reg, report = yaml.safe_load(REG.read_text()), yaml.safe_load(RESULT.read_text())
    pins.update({Path(p): value for p, value in report['source_hashes'].items()})
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('subset input changed')
    if (report['status'] != 'finite-subset-and-size-preflight-complete-independent-verification-pending'
        or report['network_requests'] != 0 or report['physiology_accessed'] is not False
        or report['recording_downloaded'] is not False or report['specimen_eligibility_established'] is not False):
        raise ValueError('subset claim scope differs')
    data = {k: yaml.load(Path(reg[k]).read_text(), Loader=yaml.CSafeLoader) for k in ('candidates', 'file_references', 'typed_manifest')}
    expected = reconcile(reg, data['candidates'], data['file_references'], data['typed_manifest'])
    for key in ('slots', 'strata', 'problems', 'summary'):
        same(expected[key], report[key])
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('source changed during selection verification')
    OUTPUT.parent.mkdir()
    with OUTPUT.open('x') as handle:
        yaml.safe_dump({'status': 'finite-intrinsic-subset-and-author-preflight-independently-reconstructed',
            'verifier_sha256': digest(__file__), 'source_hashes': {str(p): value for p, value in pins.items()},
            'summary': expected['summary'], 'verified_at_utc': datetime.now(UTC).isoformat(),
            'every_slot_rank_author_hash_and_size_matches': True, 'selector_imported': False,
            'network_requests': 0, 'physiology_accessed': False, 'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('All 48 selected slots, source ranks and author size/hash anchors independently verified')


if __name__ == '__main__':
    main()
