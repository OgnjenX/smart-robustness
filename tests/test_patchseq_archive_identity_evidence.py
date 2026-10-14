"""Archive metadata comparisons never turn incomplete identity into eligibility."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/inventory_patchseq_archive_identity_evidence.py'
spec = importlib.util.spec_from_file_location('identity_evidence', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def snapshot(value):
    return {'present': True, 'type': type(value).__name__, 'value': value}


@pytest.mark.parametrize('observed', [True, 1.0, '1', None, 0, -1])
def test_nonpositive_or_unexpected_size_unknown(observed):
    assert module.check_field(snapshot(observed), 1, 'int') is None


def test_exact_integer_and_literal_string_comparisons():
    assert module.check_field(snapshot(123), 123, 'int') is True
    assert module.check_field(snapshot(124), 123, 'int') is False
    assert module.check_field(snapshot('sub-123'), 'sub-123', 'str') is True
    assert module.check_field(snapshot('sub-123 '), 'sub-123', 'str') is False


@pytest.mark.parametrize('raw', ['', None, 'not-hex', 'a'*63, 'a'*65])
def test_invalid_digest_shape_unknown(raw):
    assert module.digest_comparison({'type': 'str', 'value': raw}, 'a'*64, 64) is None


def test_digest_literal_case_not_normalized():
    assert module.digest_comparison({'type': 'str', 'value': 'a'*64}, 'a'*64, 64) is True
    assert module.digest_comparison({'type': 'str', 'value': 'A'*64}, 'a'*64, 64) is False


def test_all_actor_candidates_and_etag_unknown_retained():
    asset = {'asset_id': 'uuid', 'path': 'path', 'size': 123}
    branch = {'fields': {'asset_id': snapshot('uuid'), 'path': snapshot('path'), 'size': snapshot(123),
        'identifier': snapshot('generic-artifact-id'), 'digest': {'present': True, 'type': 'dict',
        'algorithm_values': {'dandi:dandi-etag': {'type': 'str', 'value': 'a'*32},
                             'dandi:sha2-256': {'type': 'str', 'value': 'b'*64}}}},
        'participants': {'present': True, 'type': 'list', 'length': 2, 'entries': [
            {'identifier': snapshot('123'), 'schemaKey': snapshot('Participant')},
            {'identifier': snapshot('999'), 'schemaKey': snapshot('Participant')}]},
        'activities': {'present': True, 'type': 'list', 'length': 3, 'entries': [
            {'identifier': snapshot('456'), 'schemaKey': snapshot('Session')},
            {'identifier': snapshot('456'), 'schemaKey': snapshot('ProcessingActivity')},
            {'identifier': snapshot('456'), 'schemaKey': {'present': False}}]}}
    result = module.branch_evidence(branch, asset, '123', '456', {'sha256_checksum': 'b'*64, 'md5_checksum': 'a'*32})
    assert [r['literal_path_identifier_agrees'] for r in result['participant_candidates']] == [True, False]
    assert [r['literal_path_identifier_agrees'] for r in result['activity_candidates']] == [True, None, None]
    assert result['reported_digest_comparisons']['dandi:dandi-etag']['literal_reported_digest_agrees'] is None
    sha_report = result['reported_digest_comparisons']['dandi:sha2-256']
    assert sha_report['literal_reported_digest_agrees'] is True
    assert not sha_report['independent_content_digest_verified']
    summary = module.summarize([{'metadata_branches': {'top_level': result}, 'acquisition_status': 'success'}])
    assert summary['eligible_specimens'] == summary['independently_content_verified_assets'] == 0
    assert summary['branch_and_candidate_comparison_counts']['top_level.participant_candidates'] == {'true': 1, 'false': 1}


def test_failed_acquisition_kept_not_a_biological_failure():
    summary = module.summarize([{'metadata_branches': {}, 'acquisition_status': 'acquisition-failure-retained'}])
    assert summary['assets'] == 1
    assert summary['assets_with_successful_metadata'] == 0
    assert summary['eligible_specimens'] == 0
