"""Offline independent reconstruction of archive metadata acquisition receipts."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from email.parser import Parser
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-archive-identity-metadata-registration-1238.yaml')
REG_HASH = '5f94fe6a58ad551be3ea09251750bc713e77de7dec4c64ff910b486873d0ca3f'
COLLECTOR = Path('scripts/collect_patchseq_archive_identity_metadata.py')
COLLECTOR_HASH = 'e82bfcb6b581f799244d9c8e6749d6f180a1efa07b0e4344f983330aea0d3d68'
OUTPUT = Path('results/patchseq-archive-identity-independent-1241/manifest.yaml')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def decode(raw, limits):
    if len(raw) > limits['maximum_response_bytes']:
        raise ValueError('oversized response')

    def unique(items):
        names = [k for k, _ in items]
        if len(set(names)) != len(names):
            raise ValueError('duplicate JSON keys')
        return dict(items)

    def invalid(value):
        raise ValueError('nonfinite JSON constant')

    root = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)
    if type(root) is not dict:
        raise TypeError('object required')
    count = 0

    def visit(value, depth):
        nonlocal count
        count += 1
        if count > limits['maximum_json_nodes'] or depth > limits['maximum_json_depth']:
            raise ValueError('JSON structure exceeds bounds')
        if type(value) is float and not math.isfinite(value):
            raise ValueError('nonfinite number')
        if type(value) in (dict, list):
            for child in value.values() if type(value) is dict else value:
                visit(child, depth+1)

    visit(root, 0)
    return root


def reconstruct(source, reg):
    limit, allowed = reg['limits'], reg['permitted_projection']

    def atom(value):
        kind = type(value)
        if kind is str and len(value) > limit['maximum_projected_string_characters']:
            raise ValueError('string exceeds bound')
        if kind in (dict, list) and len(value) > limit['maximum_projected_list_entries']:
            raise ValueError('container exceeds bound')
        result = {'type': kind.__name__}
        if kind is dict:
            result['keys'] = sorted(value.keys())
        elif kind is list:
            result['length'] = len(value)
        else:
            result['value'] = value
        return result

    def table(obj, names):
        entries = {}
        for name in names:
            entries[name] = {'present': name in obj}
            if name not in obj:
                continue
            entries[name].update(atom(obj[name]))
            if name == 'digest' and type(obj[name]) is dict:
                entries[name]['algorithm_values'] = {algorithm: atom(value) for algorithm, value in obj[name].items()}
        return entries

    result = {'top_level': {}, 'metadata_container': {'present': 'metadata' in source}}
    branches = [(source, result['top_level'], allowed['top_level_fields'])]
    if 'metadata' in source:
        value = source['metadata']
        result['metadata_container']['type'] = type(value).__name__
        if type(value) is dict:
            result['metadata_container']['projection'] = {}
            branches.append((value, result['metadata_container']['projection'], allowed['metadata_container_fields']))
    for obj, target, names in branches:
        target['fields'] = table(obj, names)
        for label, source_key, fields in [('participants', allowed['participant_container'], allowed['participant_fields']),
                                           ('activities', allowed['activity_container'], allowed['activity_fields'])]:
            target[label] = {'present': source_key in obj}
            if source_key not in obj:
                continue
            value = obj[source_key]
            target[label].update(atom(value))
            if type(value) is list:
                target[label]['entries'] = []
                for member in value:
                    target[label]['entries'].append(table(member, fields) if type(member) is dict
                                                    else {'unexpected_entry': atom(member)})
    return result


def http_headers(raw):
    if len(raw) > 65536:
        raise ValueError('oversized headers')
    last = raw.decode('utf-8').replace('\r\n', '\n').strip().split('\n\n')[-1]
    status, _, rest = last.partition('\n')
    if status.split()[1:2] != ['200']:
        raise ValueError('HTTP status not 200')
    parsed = Parser().parsestr(rest)
    fields = {}
    for name in ('content-type', 'content-length', 'content-encoding', 'etag', 'last-modified', 'date'):
        values = parsed.get_all(name)
        if values is not None:
            fields[name] = [v.strip() for v in values]
    if fields.get('content-encoding', ['identity']) != ['identity']:
        raise ValueError('encoded response')
    if not any(v.split(';')[0].strip() == 'application/json' for v in fields.get('content-type', [])):
        raise ValueError('non-JSON content type')
    return fields


def same(expected, observed):
    if json.dumps(expected, sort_keys=True) != json.dumps(observed, sort_keys=True):
        raise ValueError('type-preserving evidence mismatch')


def verify_receipt(directory, asset, reg):
    receipt = yaml.load((directory/'receipt.yaml').read_text(), Loader=yaml.CSafeLoader)
    attempt = yaml.load((directory/'attempt.yaml').read_text(), Loader=yaml.CSafeLoader)
    same(attempt, {k: receipt[k] for k in ('asset', 'endpoint', 'registration_sha256', 'collector_sha256', 'started_at_utc')})
    same(receipt['asset'], asset)
    if (receipt['endpoint'] != reg['endpoint_template'].format(asset_id=asset['asset_id'])
        or receipt['registration_sha256'] != REG_HASH or receipt['collector_sha256'] != COLLECTOR_HASH
        or receipt['physiological_arrays_read'] is not False):
        raise ValueError('receipt identity or scope mismatch')
    paths = []
    for name, key in [('response.json', 'response_sha256'), ('headers.txt', 'headers_sha256')]:
        path = directory/name
        if key in receipt:
            if not path.exists() or sha(path) != receipt[key]:
                raise ValueError('raw evidence hash mismatch')
            paths.append((path, receipt[key]))
        elif path.exists():
            raise ValueError('unreceipted raw evidence')
    success = receipt['status'] == 'archive-metadata-projected-independent-verification-pending'
    if not success and receipt['status'] != 'acquisition-failure-retained':
        raise ValueError('unexpected acquisition status')
    client_ok = receipt.get('client_exit_code') == 0 and receipt.get('http_status') == '200'
    if success and not client_ok:
        raise ValueError('successful receipt without successful HTTP request')
    if success or client_ok:
        try:
            header = http_headers((directory/'headers.txt').read_bytes())
            body = directory/'response.json'
            if body.stat().st_size > reg['limits']['maximum_response_bytes']:
                raise ValueError('oversized raw response')
            projected = reconstruct(decode(body.read_bytes(), reg['limits']), reg)
        except (OSError, ValueError, TypeError, RecursionError):
            if success:
                raise
        else:
            if not success:
                raise ValueError('failed receipt with valid complete response needs explicit assessment')
            same(header, receipt['headers'])
            same(projected, receipt['projection'])
    elif 'projection' in receipt:
        raise ValueError('failed HTTP receipt has a projection')
    if not success and 'error' not in receipt:
        raise ValueError('failure reason absent')
    for path, expected in paths:
        if sha(path) != expected:
            raise ValueError('raw evidence changed during reconstruction')
    return {'asset_id': asset['asset_id'], 'receipt_sha256': sha(directory/'receipt.yaml'),
            'attempt_sha256': sha(directory/'attempt.yaml'), 'status': receipt['status'],
            'raw_evidence_hashes': {str(p): h for p, h in paths}, 'projection_independently_matched': success}


def main():
    if sha(REG) != REG_HASH or sha(COLLECTOR) != COLLECTOR_HASH:
        raise ValueError('sealed source changed')
    reg = yaml.load(REG.read_text(), Loader=yaml.CSafeLoader)
    catalog_path = Path(reg['catalog'])
    if sha(catalog_path) != reg['catalog_sha256'] or sha(reg['prerequisite']['result']) != reg['prerequisite']['result_sha256']:
        raise ValueError('pinned prerequisite changed')
    root = Path(reg['output_directory'])
    terminal_path = root/'manifest.yaml'
    terminal_hash = sha(terminal_path)
    terminal = yaml.load(terminal_path.read_text(), Loader=yaml.CSafeLoader)
    if (terminal['status'] != 'full-archive-metadata-acquisition-terminal-independent-verification-pending'
        or terminal['registration_sha256'] != REG_HASH or terminal['collector_sha256'] != COLLECTOR_HASH
        or terminal['physiological_arrays_read'] is not False or terminal['specimen_qualification_performed'] is not False):
        raise ValueError('terminal scope mismatch')
    assets = sorted(yaml.load(catalog_path.read_text(), Loader=yaml.CSafeLoader)['assets'], key=lambda a: (a['path'], a['asset_id']))
    if len(assets) != reg['selection']['expected_assets'] or len(terminal['records']) != len(assets):
        raise ValueError('full terminal cohort required')
    records = []
    for asset, item in zip(assets, terminal['records'], strict=True):
        directory = root/asset['asset_id']
        if (item['asset_id'] != asset['asset_id'] or item['receipt'] != str(directory/'receipt.yaml')
            or item['receipt_sha256'] != sha(directory/'receipt.yaml')):
            raise ValueError('terminal receipt mismatch')
        verified = verify_receipt(directory, asset, reg)
        if verified['status'] != item['status'] or verified['receipt_sha256'] != item['receipt_sha256']:
            raise ValueError('terminal status mismatch')
        records.append(verified)
    counts = dict(Counter(r['status'] for r in records))
    same(counts, terminal['status_counts'])
    same(len(assets), terminal['network_requests_maximum'])
    if (sha(REG) != REG_HASH or sha(COLLECTOR) != COLLECTOR_HASH
        or sha(catalog_path) != reg['catalog_sha256'] or sha(terminal_path) != terminal_hash):
        raise ValueError('source changed during verification')
    for record in records:
        directory = root/record['asset_id']
        if sha(directory/'receipt.yaml') != record['receipt_sha256'] or sha(directory/'attempt.yaml') != record['attempt_sha256']:
            raise ValueError('receipt changed during verification')
        for path, expected in record['raw_evidence_hashes'].items():
            if sha(path) != expected:
                raise ValueError('response changed during verification')
    OUTPUT.parent.mkdir()
    with OUTPUT.open('x') as handle:
        yaml.safe_dump({'status': 'all-terminal-archive-metadata-receipts-independently-verified',
            'verifier_sha256': sha(__file__), 'registration_sha256': REG_HASH,
            'terminal_manifest_sha256': terminal_hash, 'records': records, 'status_counts': counts,
            'verified_at_utc': datetime.now(UTC).isoformat(), 'collector_imported': False,
            'network_requests': 0, 'physiological_arrays_read': False, 'specimen_qualification_performed': False},
            handle, sort_keys=False)
    print(f'Independently verified {len(records)} receipts: {counts}')


if __name__ == '__main__':
    main()
