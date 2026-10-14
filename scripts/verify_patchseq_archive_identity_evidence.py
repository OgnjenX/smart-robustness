"""Rebuild identity evidence from original JSON, without importing its collector."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-archive-identity-evidence-registration-1260.yaml')
RESULT = Path('results/patchseq-archive-identity-evidence-1261/manifest.yaml')
OUTPUT = Path('results/patchseq-archive-identity-evidence-independent-1264/manifest.yaml')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def equal(expected, observed):
    if json.dumps(expected, sort_keys=True) != json.dumps(observed, sort_keys=True):
        raise ValueError('type-preserving evidence mismatch')


def literal(raw, name, expected, kind):
    value = raw.get(name)
    if type(value) is not kind or (kind is int and value <= 0):
        return None
    return value == expected


def hexadecimal(value, width):
    return type(value) is str and len(value) == width and all(c in '0123456789abcdefABCDEF' for c in value)


def rebuild_branch(raw, projection, asset, subject, session, source):
    branch = {'raw_projection': projection,
              'literal_asset_id_agrees': literal(raw, 'asset_id', asset['asset_id'], str),
              'literal_path_agrees': literal(raw, 'path', asset['path'], str),
              'size_agrees': literal(raw, 'size', asset['size'], int),
              'contentSize_agrees': literal(raw, 'contentSize', asset['size'], int),
              'participant_candidates': [], 'activity_candidates': [], 'reported_digest_comparisons': {}}
    for raw_key, projected_key, output_key, expected in (
        ('wasAttributedTo', 'participants', 'participant_candidates', subject),
        ('wasGeneratedBy', 'activities', 'activity_candidates', session)):
        values = raw.get(raw_key)
        if type(values) is not list:
            continue
        projected = projection[projected_key]['entries']
        if len(projected) != len(values):
            raise ValueError('actor projection cardinality mismatch')
        for entry, fields in zip(values, projected, strict=True):
            agreement = None
            if type(entry) is dict and (raw_key == 'wasAttributedTo' or entry.get('schemaKey') == 'Session'):
                agreement = literal(entry, 'identifier', expected, str)
            branch[output_key].append({'raw_fields': fields, 'literal_path_identifier_agrees': agreement})
    digests = raw.get('digest')
    if type(digests) is dict:
        for algorithm, value in digests.items():
            key, width = {'dandi:sha2-256': ('sha256_checksum', 64), 'dandi:md5': ('md5_checksum', 32)}.get(algorithm, (None, None))
            source_value = source[key] if key else None
            agreement = value == source_value if key and hexadecimal(value, width) and hexadecimal(source_value, width) else None
            branch['reported_digest_comparisons'][algorithm] = {
                'archive_snapshot': projection['fields']['digest']['algorithm_values'][algorithm],
                'source_literal': source_value, 'literal_reported_digest_agrees': agreement,
                'independent_content_digest_verified': False}
    return branch


def rebuild_summary(records):
    flattened, containers, digests = {}, {}, []
    for record in records:
        for label, branch in record['metadata_branches'].items():
            for group in ('participants', 'activities'):
                raw = branch['raw_projection'][group]
                state = raw.get('type', 'unknown') if raw.get('present') is True else 'missing'
                containers.setdefault(label+'.'+group, []).append(state)
            flags = {key: [branch[key]] for key in ('literal_asset_id_agrees', 'literal_path_agrees', 'size_agrees', 'contentSize_agrees')}
            for group in ('participant_candidates', 'activity_candidates'):
                flags[group] = [r['literal_path_identifier_agrees'] for r in branch[group]]
            for algorithm, digest in branch['reported_digest_comparisons'].items():
                digests.append(algorithm)
                flags['digest.'+algorithm] = [digest['literal_reported_digest_agrees']]
            for key, values in flags.items():
                if values:
                    flattened.setdefault(label+'.'+key, []).extend('true' if v is True else 'false' if v is False else 'unknown' for v in values)
    return {'assets': len(records), 'assets_with_successful_metadata': sum(bool(r['metadata_branches']) for r in records),
        'acquisition_status_counts': dict(Counter(r['acquisition_status'] for r in records)),
        'branch_and_candidate_comparison_counts': {k: dict(Counter(v)) for k, v in sorted(flattened.items())},
        'actor_container_type_counts': {k: dict(Counter(v)) for k, v in sorted(containers.items())},
        'reported_digest_algorithm_occurrences': dict(sorted(Counter(digests).items())),
        'independently_content_verified_assets': 0, 'eligible_specimens': 0}


def main():
    pins = {REG: '50d8406a11f393604a513ce196861c6c542631d3d76a426e299fd34ac8bcb2e7',
            RESULT: '9ea9fe950b604020c5f15098816ea8235b07ca2971e3068ce8efb669b1a99502',
            Path('scripts/inventory_patchseq_archive_identity_evidence.py'): 'b9f95a4ef8a508bd4f5a2ba0c40b21b83777e0e364fafb6587d5d0a819824dd0'}
    reg = yaml.load(REG.read_text(), Loader=yaml.CSafeLoader)
    observed = yaml.load(RESULT.read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(p): h for p, h in observed['source_hashes'].items()})
    # Known terminal lineage is pinned independently of the inventory's report.
    pins[Path(reg['acquisition_manifest'])] = '6143c5f38dee95355235728d6620313d4fa5e2cad255d7097de0d2875fcc1911'
    pins[Path(reg['independent_verification'])] = 'adcecc68047625e756421f04546212186c5fd4ffb212e5cea3030761e987e05e'
    for key in ('acquisition_registration', 'independent_verifier', 'catalog', 'typed_manifest', 'file_references'):
        pins[Path(reg[key])] = reg[key+'_sha256']
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError('pinned source changed')
    if (observed['status'] != 'archive-identity-evidence-independent-reconstruction-pending'
        or observed['registration_sha256'] != pins[REG]
        or observed['collector_sha256'] != pins[Path('scripts/inventory_patchseq_archive_identity_evidence.py')]
        or observed['network_requests'] != 0 or observed['physiological_arrays_read'] is not False
        or observed['specimen_eligibility_established'] is not False):
        raise ValueError('inventory scope or provenance mismatch')
    def load(key):
        return yaml.load(Path(reg[key]).read_text(), Loader=yaml.CSafeLoader)
    terminal, verified = load('acquisition_manifest'), load('independent_verification')
    if (verified['status'] != 'all-terminal-archive-metadata-receipts-independently-verified'
        or verified['terminal_manifest_sha256'] != pins[Path(reg['acquisition_manifest'])]
        or verified['verifier_sha256'] != reg['independent_verifier_sha256']):
        raise ValueError('terminal independent verification mismatch')
    spec = importlib.util.spec_from_file_location('independent_projection_reader', reg['independent_verifier'])
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    acquisition_reg = load('acquisition_registration')
    assets = sorted(load('catalog')['assets'], key=lambda a: (a['path'], a['asset_id']))
    if len(assets) != reg['expected_assets'] or len(observed['records']) != len(assets):
        raise ValueError('full cohort required')
    source_rows = {r['source_row_number']: r for r in load('typed_manifest')['rows']}
    refs = {a['asset_id']: a for a in load('file_references')['assets']}
    records = []
    for asset, item, proof in zip(assets, terminal['records'], verified['records'], strict=True):
        if item['asset_id'] != proof['asset_id'] or item['asset_id'] != asset['asset_id']:
            raise ValueError('asset order mismatch')
        receipt_path = Path(item['receipt'])
        if sha(receipt_path) != item['receipt_sha256'] or sha(receipt_path) != proof['receipt_sha256']:
            raise ValueError('receipt identity mismatch')
        receipt = yaml.load(receipt_path.read_text(), Loader=yaml.CSafeLoader)
        if receipt['asset'] != asset or receipt['status'] != proof['status']:
            raise ValueError('receipt scope mismatch')
        candidates = refs[asset['asset_id']]['manifest_candidates']
        if len(candidates) != 1:
            raise ValueError('unique source checksum reference required')
        source = source_rows[candidates[0]['manifest_source_row_number']]
        checksums = {k: source['cells'][k]['literal_text'] if source['cells'][k] else None for k in ('sha256_checksum', 'md5_checksum')}
        directory, filename = asset['path'].split('/')
        subject = directory.removeprefix('sub-')
        stem = filename.removesuffix('_icephys.nwb')
        file_subject, session = stem.removeprefix('sub-').split('_ses-')
        if file_subject != subject:
            raise ValueError('path subject mismatch')
        branches = {}
        if proof['projection_independently_matched'] is True:
            body = receipt_path.parent/'response.json'
            if sha(body) != receipt['response_sha256']:
                raise ValueError('response receipt mismatch')
            raw = reader.decode(body.read_bytes(), acquisition_reg['limits'])
            projection = reader.reconstruct(raw, acquisition_reg)
            equal(projection, receipt['projection'])
            branches['top_level'] = rebuild_branch(raw, projection['top_level'], asset, subject, session, checksums)
            if type(raw.get('metadata')) is dict:
                branches['metadata_container'] = rebuild_branch(raw['metadata'], projection['metadata_container']['projection'], asset, subject, session, checksums)
        records.append({'asset': asset, 'acquisition_status': receipt['status'], 'metadata_branches': branches,
            'source_manifest_row_number': source['source_row_number'], 'source_reported_checksums': checksums,
            'generic_artifact_identifier_used_as_cell_ID': False, 'internal_recording_identity_confirmed': False,
            'donor_namespace_equivalence_established': False, 'specimen_eligibility_established': False})
    equal(records, observed['records'])
    summary = rebuild_summary(records)
    equal(summary, observed['summary'])
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError('evidence changed during reconstruction')
    OUTPUT.parent.mkdir()
    with OUTPUT.open('x') as handle:
        yaml.safe_dump({'status': 'all-archive-identity-and-digest-evidence-independently-reconstructed',
            'verifier_sha256': sha(__file__), 'source_hashes': {str(p): h for p, h in pins.items()},
            'summary': summary, 'verified_at_utc': datetime.now(UTC).isoformat(),
            'inventory_collector_imported': False, 'network_requests': 0, 'physiological_arrays_read': False,
            'specimen_eligibility_established': False}, handle, sort_keys=False)
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == '__main__':
    main()
