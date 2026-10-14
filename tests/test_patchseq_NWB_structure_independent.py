"""Independent structure replay and receipt tamper rejection."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('independent_structure', scripts/'verify_patchseq_NWB_structure_ranges.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def test_independent_traversal_does_not_read_values(tmp_path, monkeypatch):
    h5py = pytest.importorskip('h5py')
    source = tmp_path/'fixture.nwb'
    with h5py.File(source, 'w') as file:
        file['general/session_id'] = 'do-not-read'
        file.create_group('general/subject').attrs['private'] = 'do-not-read'
        file['general/external'] = h5py.ExternalLink('/never-open.h5', '/')
        file['acquisition/trace'] = [1, 2, 3]
    def forbidden(*args, **kwargs):
        raise AssertionError('value access forbidden')
    monkeypatch.setattr(h5py.Dataset, '__getitem__', forbidden)
    monkeypatch.setattr(h5py.AttributeManager, '__getitem__', forbidden)
    policy = yaml.safe_load(Path('docs/validation-results/post2008-patchseq-NWB-structure-registration-1271.yaml').read_text())['structure_policy']
    records = []
    with h5py.File(source, 'r') as file:
        module.reconstruct(file, policy, h5py, records)
    assert next(r for r in records if r['path'] == '/general/session_id')['dataset_shape'] == []
    assert next(r for r in records if r['path'] == '/general/external')['dereferenced'] is False
    assert not any(r['path'].startswith('/acquisition/') for r in records)
    assert 'do-not-read' not in str(records)


def test_type_preserving_comparison_rejects_bool_integer_substitution():
    with pytest.raises(ValueError):
        module.equal({'shape': [1]}, {'shape': [True]})


@pytest.mark.parametrize('field', ['new_range_requests', 'charged_new_range_bytes', 'combined_cache_bytes'])
def test_receipt_accounting_mismatch_rejected(field, tmp_path):
    def receipt(kind, fields, data, ordinal, start=None, end=None):
        headers = tmp_path/f'{ordinal}-headers.txt'
        body = tmp_path/f'{ordinal}-body.bin'
        status = 200 if kind == 'HEAD' else 206
        headers.write_text(f'HTTP/1.1 {status} OK\n'+''.join(f'{k}: {v[0]}\n' for k, v in fields.items())+'\n')
        body.write_bytes(data)
        return {'kind': kind, 'URL_host': 'dandiarchive.s3.amazonaws.com', 'URL_path': '/blobs/synthetic',
            'start': start, 'end': end, 'http_status': status, 'header_fields': fields,
            'client_exit_code': 0, 'status': 'response-retained',
            'headers_path': str(headers), 'headers_sha256': module.sha(headers), 'headers_bytes': headers.stat().st_size,
            'body_path': str(body), 'body_sha256': module.sha(body), 'body_bytes': len(data)}
    pilot = {'asset': {'size': 16}, 'HTTP_receipts': [receipt('RANGE', {'etag': ['"stable"']}, b'abcd', 0, 0, 3)]}
    report = {'HTTP_receipts': [receipt('HEAD', {'etag': ['"stable"'], 'content-length': ['16']}, b'', 1),
        receipt('RANGE', {'etag': ['"stable"'], 'content-range': ['bytes 4-7/16']}, b'efgh', 2, 4, 7)],
        'new_range_requests': 1, 'charged_new_range_bytes': 4, 'combined_cache_bytes': 8,
        'started_at_utc': '2026-10-14T00:00:00+00:00', 'finished_at_utc': '2026-10-14T00:00:01+00:00'}
    reg = {'HTTP_range_limits': {'cache_block_bytes': 4, 'maximum_range_requests': 2,
        'maximum_unique_downloaded_bytes': 8, 'maximum_run_seconds': 900},
        'URL_resolution': {'maximum_HEAD_requests': 2, 'maximum_HEAD_redirects': 1},
        'seed_cache': {'maximum_combined_cache_bytes': 12}}
    locator = 'https://dandiarchive.s3.amazonaws.com/blobs/synthetic'
    module.check_receipts(report, pilot, reg, locator)
    altered = copy.deepcopy(report)
    altered[field] += 1
    with pytest.raises(ValueError):
        module.check_receipts(altered, pilot, reg, locator)
