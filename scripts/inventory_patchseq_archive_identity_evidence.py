"""Sealed metadata-consistency diagnostics, never internal recording validation."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-archive-identity-evidence-registration-1260.yaml')
REG_HASH = '50d8406a11f393604a513ce196861c6c542631d3d76a426e299fd34ac8bcb2e7'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def value(snapshot, kind):
    if snapshot.get('present') is True and snapshot.get('type') == kind:
        result = snapshot.get('value')
        if type(result).__name__ == kind:
            return result
    return None


def check_field(snapshot, expected, kind):
    observed = value(snapshot, kind)
    if observed is None or (kind == 'int' and observed <= 0):
        return None
    return observed == expected


def digest_comparison(snapshot, source, width):
    # Digest algorithm snapshots do not contain a field-presence flag.
    observed = snapshot.get('value') if snapshot.get('type') == 'str' else None
    pattern = f'[0-9a-fA-F]{{{width}}}'
    if type(observed) is not str or type(source) is not str or not re.fullmatch(pattern, observed) or not re.fullmatch(pattern, source):
        return None
    return observed == source


def branch_evidence(branch, asset, subject, session, source_checksums):
    fields = branch['fields']
    evidence = {'raw_projection': branch,
                'literal_asset_id_agrees': check_field(fields.get('asset_id', {}), asset['asset_id'], 'str'),
                'literal_path_agrees': check_field(fields.get('path', {}), asset['path'], 'str'),
                'size_agrees': check_field(fields.get('size', {}), asset['size'], 'int'),
                'contentSize_agrees': check_field(fields.get('contentSize', {}), asset['size'], 'int'),
                'participant_candidates': [], 'activity_candidates': [], 'reported_digest_comparisons': {}}
    for label, output in [('participants', 'participant_candidates'), ('activities', 'activity_candidates')]:
        container = branch[label]
        if container.get('type') != 'list':
            continue
        for entry in container.get('entries', []):
            if 'unexpected_entry' in entry:
                evidence[output].append({'raw_fields': entry, 'literal_path_identifier_agrees': None})
                continue
            schema = value(entry.get('schemaKey', {}), 'str')
            permitted = label == 'participants' or schema == 'Session'
            agreement = check_field(entry.get('identifier', {}), subject if label == 'participants' else session, 'str') if permitted else None
            evidence[output].append({'raw_fields': entry, 'literal_path_identifier_agrees': agreement})
    algorithms = fields.get('digest', {}).get('algorithm_values', {})
    for algorithm, snapshot in algorithms.items():
        if algorithm == 'dandi:sha2-256':
            agreement = digest_comparison(snapshot, source_checksums['sha256_checksum'], 64)
        elif algorithm == 'dandi:md5':
            agreement = digest_comparison(snapshot, source_checksums['md5_checksum'], 32)
        else:
            agreement = None
        evidence['reported_digest_comparisons'][algorithm] = {'archive_snapshot': snapshot,
            'source_literal': source_checksums.get('sha256_checksum' if algorithm == 'dandi:sha2-256' else 'md5_checksum')
                if algorithm in ('dandi:sha2-256', 'dandi:md5') else None,
            'literal_reported_digest_agrees': agreement, 'independent_content_digest_verified': False}
    return evidence


def summarize(records):
    comparisons = defaultdict(Counter)
    containers = defaultdict(Counter)
    digest_counts = Counter()
    for row in records:
        for label, branch in row['metadata_branches'].items():
            for container in ('participants', 'activities'):
                snapshot = branch['raw_projection'][container]
                state = 'missing' if snapshot.get('present') is not True else snapshot.get('type', 'unknown')
                containers[label+'.'+container][state] += 1
            for key in ('literal_asset_id_agrees', 'literal_path_agrees', 'size_agrees', 'contentSize_agrees'):
                comparisons[label+'.'+key]['true' if branch[key] is True else 'false' if branch[key] is False else 'unknown'] += 1
            for group in ('participant_candidates', 'activity_candidates'):
                for candidate in branch[group]:
                    flag = candidate['literal_path_identifier_agrees']
                    comparisons[label+'.'+group]['true' if flag is True else 'false' if flag is False else 'unknown'] += 1
            for algorithm, digest in branch['reported_digest_comparisons'].items():
                digest_counts[algorithm] += 1
                flag = digest['literal_reported_digest_agrees']
                comparisons[label+'.digest.'+algorithm]['true' if flag is True else 'false' if flag is False else 'unknown'] += 1
    return {'assets': len(records), 'assets_with_successful_metadata': sum(bool(r['metadata_branches']) for r in records),
            'acquisition_status_counts': dict(Counter(r['acquisition_status'] for r in records)),
            'branch_and_candidate_comparison_counts': {k: dict(v) for k, v in sorted(comparisons.items())},
            'actor_container_type_counts': {k: dict(v) for k, v in sorted(containers.items())},
            'reported_digest_algorithm_occurrences': dict(sorted(digest_counts.items())),
            'independently_content_verified_assets': 0, 'eligible_specimens': 0}


def main():
    if sha(REG) != REG_HASH:
        raise ValueError('sealed archive evidence registration changed')
    reg = yaml.load(REG.read_text(), Loader=yaml.CSafeLoader)
    pins = {REG: REG_HASH}
    for key in ('acquisition_registration', 'independent_verifier', 'catalog', 'typed_manifest', 'file_references'):
        pins[Path(reg[key])] = reg[key+'_sha256']
    for key in ('acquisition_manifest', 'independent_verification'):
        pins[Path(reg[key])] = sha(reg[key])
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError('pinned sources differ')
    def load(key):
        return yaml.load(Path(reg[key]).read_text(), Loader=yaml.CSafeLoader)
    terminal, verified = load('acquisition_manifest'), load('independent_verification')
    if (verified['status'] != 'all-terminal-archive-metadata-receipts-independently-verified'
        or verified['terminal_manifest_sha256'] != pins[Path(reg['acquisition_manifest'])]
        or verified['verifier_sha256'] != reg['independent_verifier_sha256']):
        raise ValueError('independent verification must cover exact terminal acquisition')
    catalog = sorted(load('catalog')['assets'], key=lambda a: (a['path'], a['asset_id']))
    if len(catalog) != reg['expected_assets'] or len(verified['records']) != len(catalog) or len(terminal['records']) != len(catalog):
        raise ValueError('full verified catalog required')
    rows = {r['source_row_number']: r for r in load('typed_manifest')['rows']}
    references = {a['asset_id']: a for a in load('file_references')['assets']}
    records = []
    for asset, receipt_ref, proof in zip(catalog, terminal['records'], verified['records'], strict=True):
        if receipt_ref['asset_id'] != asset['asset_id'] or proof['asset_id'] != asset['asset_id']:
            raise ValueError('verified asset order differs')
        path = Path(receipt_ref['receipt'])
        if sha(path) != receipt_ref['receipt_sha256'] or sha(path) != proof['receipt_sha256']:
            raise ValueError('receipt changed since verification')
        pins[path] = proof['receipt_sha256']
        for raw, expected in proof['raw_evidence_hashes'].items():
            if sha(raw) != expected:
                raise ValueError('raw evidence changed since verification')
            pins[Path(raw)] = expected
        receipt = yaml.load(path.read_text(), Loader=yaml.CSafeLoader)
        if receipt['asset'] != asset or receipt['status'] != proof['status']:
            raise ValueError('verified receipt identity differs')
        source_refs = references[asset['asset_id']]['manifest_candidates']
        if len(source_refs) != 1:
            raise ValueError('one verified source file reference required')
        source = rows[source_refs[0]['manifest_source_row_number']]
        checksums = {k: source['cells'][k]['literal_text'] if source['cells'][k] else None for k in ('sha256_checksum', 'md5_checksum')}
        match = re.fullmatch(r'sub-([1-9][0-9]*)/sub-\1_ses-([1-9][0-9]*)_icephys\.nwb', asset['path'])
        if match is None:
            raise ValueError('exact catalog path required')
        branches = {}
        if proof['projection_independently_matched'] is True:
            projection = receipt['projection']
            branches['top_level'] = branch_evidence(projection['top_level'], asset, match[1], match[2], checksums)
            nested = projection['metadata_container']
            if nested.get('type') == 'dict':
                branches['metadata_container'] = branch_evidence(nested['projection'], asset, match[1], match[2], checksums)
        records.append({'asset': asset, 'acquisition_status': receipt['status'], 'metadata_branches': branches,
            'source_manifest_row_number': source['source_row_number'], 'source_reported_checksums': checksums,
            'generic_artifact_identifier_used_as_cell_ID': False, 'internal_recording_identity_confirmed': False,
            'donor_namespace_equivalence_established': False, 'specimen_eligibility_established': False})
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError('source changed during evidence inventory')
    root = Path(reg['output_directory'])
    root.mkdir()
    report = {'status': 'archive-identity-evidence-independent-reconstruction-pending', 'registration_sha256': REG_HASH,
              'collector_sha256': sha(__file__), 'source_hashes': {str(p): h for p, h in pins.items()},
              'records': records, 'summary': summarize(records), 'finished_at_utc': datetime.now(UTC).isoformat(),
              'network_requests': 0, 'physiological_arrays_read': False, 'specimen_eligibility_established': False}
    with (root/'manifest.yaml').open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(yaml.safe_dump(report['summary'], sort_keys=False))


if __name__ == '__main__':
    main()
