"""Explicit exact-name alignment within the independently sourced reference."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-reference-crosswalk-registration-1217.yaml')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def text(row, field):
    cell = row['cells'][field]
    if cell is None or cell['formula_present'] or cell['storage_type'] not in ('s', 'inlineStr'):
        return None
    return cell['literal_text'] if type(cell['literal_text']) is str else None


def align(records, july, taxonomy, description, reference):
    sheets = records['reference_sheets']
    provenance = [r for r in sheets['taxonomy ids']['rows'] if text(r, 'Taxonomy id') == taxonomy]
    if (len(provenance) != 1 or text(provenance[0], 'Description') != description
        or text(provenance[0], 'Reference') != reference):
        raise ValueError('unique exact source-provenance row required')
    selected = [r for r in sheets['cell set nomenclature']['rows'] if text(r, 'taxonomy_id') == taxonomy]
    counts = Counter(text(r, 'cell_set_ accession') for r in selected if text(r, 'cell_set_ accession'))
    pairs = sorted({(r['raw_fields']['T-type Accession ID'], r['raw_fields']['T-type Label']) for r in july})
    labels = defaultdict(set)
    for old_id, label in pairs:
        if old_id:
            labels[old_id].add(label)
    outputs, targets = [], defaultdict(set)
    for old_id, label in pairs:
        candidates = [r for r in selected if label and text(r, 'cell_set_preferred_alias') == label]
        target = text(candidates[0], 'cell_set_ accession') if len(candidates) == 1 else None
        allowed = bool(old_id and label and len(labels[old_id]) == 1 and target and counts[target] == 1)
        if allowed:
            targets[target].add(old_id)
        outputs.append({'legacy_accession': old_id, 'legacy_label': label,
                        'reference_candidates': [{'source_row_number': r['source_row_number'],
                            'accession': text(r, 'cell_set_ accession'), 'preferred_alias': text(r, 'cell_set_preferred_alias'),
                            'species': text(r, 'species'), 'modality': text(r, 'modality')} for r in candidates],
                        'reference_taxonomy': taxonomy, 'exact_name_alignment_supported': allowed,
                        'reference_accession': target if allowed else None,
                        'direct_accession_equality': bool(allowed and old_id == target),
                        'historical_identifier_equivalence_confirmed': False, 'class_membership_established': False,
                        'biological_eligibility_established': False})
    return {'source_provenance_row_number': provenance[0]['source_row_number'], 'alignments': outputs,
            'legacy_accessions_with_conflicting_labels': {k: sorted(v) for k, v in sorted(labels.items()) if len(v) > 1},
            'duplicate_reference_accessions': sorted(k for k, v in counts.items() if v > 1),
            'reference_targets_with_multiple_legacy_accessions': {k: sorted(v) for k, v in sorted(targets.items()) if len(v) > 1},
            'summary': {'legacy_pairs_including_empty': len(pairs), 'selected_reference_rows': len(selected),
                        'supported_exact_name_alignments': sum(r['exact_name_alignment_supported'] for r in outputs),
                        'nonempty_pairs_without_candidates': sum(bool(r['legacy_label']) and not r['reference_candidates'] for r in outputs),
                        'ambiguous_pairs': sum(len(r['reference_candidates']) > 1 for r in outputs),
                        'direct_accession_equalities': sum(r['direct_accession_equality'] for r in outputs)}}


def main():
    if digest(REG) != 'ab5a54357b89a670a08d0642781e9e3870519c66fcde3e6b5eec1a9ac8e4b08f':
        raise ValueError('sealed name-alignment registration changed')
    reg = yaml.safe_load(REG.read_text())
    pins = {Path(reg[k]): reg[k + '_sha256'] for k in ('reference_records', 'July_comparison')}
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('pinned reference evidence changed')
    root = Path(reg['output_directory'])
    root.mkdir()
    report = {'status': 'reference-crosswalk-failure-retained', 'registration_sha256': digest(REG),
              'collector_sha256': digest(__file__), 'source_hashes': {str(p): h for p, h in pins.items()},
              'started_at_utc': datetime.now(UTC).isoformat(), 'network_requests': 0,
              'physiological_arrays_read': False, 'biological_eligibility_established': False}
    try:
        records = yaml.load(Path(reg['reference_records']).read_text(), Loader=yaml.CSafeLoader)
        july = yaml.load(Path(reg['July_comparison']).read_text(), Loader=yaml.CSafeLoader)['July_rows']
        report.update(align(records, july, reg['reference_taxonomy_id'], reg['expected_description'], reg['expected_reference']))
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError('source changed during alignment')
        report['status'] = 'provenance-constrained-exact-name-crosswalk-independent-verification-pending'
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report['error'] = str(exc)
    report['finished_at_utc'] = datetime.now(UTC).isoformat()
    with (root / 'manifest.yaml').open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report['status'])
    raise SystemExit(0 if report['status'] == 'provenance-constrained-exact-name-crosswalk-independent-verification-pending' else 1)


if __name__ == '__main__':
    main()
