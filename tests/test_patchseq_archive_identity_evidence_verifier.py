"""Independent raw-JSON comparisons preserve candidates, types and unknowns."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/verify_patchseq_archive_identity_evidence.py'
spec = importlib.util.spec_from_file_location('raw_identity_verifier', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('size', [True, 1.0, '1', None, 0, -1])
def test_noncanonical_size_is_unknown(size):
    assert module.literal({'size': size}, 'size', 1, int) is None


@pytest.mark.parametrize('value', ['g'*64, 'a'*63, 'a'*65, 123, None])
def test_invalid_hexadecimal_not_compared(value):
    assert not module.hexadecimal(value, 64)


def test_raw_candidates_and_digest_algorithms_remain_separate():
    raw = {'path': 'path', 'contentSize': 123, 'digest': {'dandi:sha2-256': 'a'*64, 'dandi:dandi-etag': 'b'*32},
           'wasAttributedTo': [{'identifier': '123'}, {'identifier': '999'}],
           'wasGeneratedBy': [{'identifier': '456', 'schemaKey': 'Session'}, {'identifier': '456', 'schemaKey': 'ProcessingActivity'}]}
    field = lambda value: {'present': True, 'type': type(value).__name__, 'value': value}
    projection = {'fields': {'digest': {'algorithm_values': {k: {'type': 'str', 'value': v} for k, v in raw['digest'].items()}}},
                  'participants': {'present': True, 'type': 'list', 'entries': [{'identifier': field('123')}, {'identifier': field('999')}]},
                  'activities': {'present': True, 'type': 'list', 'entries': [{'identifier': field('456'), 'schemaKey': field('Session')},
                                                                                   {'identifier': field('456'), 'schemaKey': field('ProcessingActivity')}]}}
    branch = module.rebuild_branch(raw, projection, {'asset_id': 'uuid', 'path': 'path', 'size': 123}, '123', '456',
                                  {'sha256_checksum': 'a'*64, 'md5_checksum': 'b'*32})
    assert branch['literal_asset_id_agrees'] is None
    assert branch['literal_path_agrees'] is True
    assert branch['contentSize_agrees'] is True
    assert [c['literal_path_identifier_agrees'] for c in branch['participant_candidates']] == [True, False]
    assert [c['literal_path_identifier_agrees'] for c in branch['activity_candidates']] == [True, None]
    assert branch['reported_digest_comparisons']['dandi:dandi-etag']['literal_reported_digest_agrees'] is None
    assert branch['reported_digest_comparisons']['dandi:sha2-256']['literal_reported_digest_agrees'] is True
    assert not branch['reported_digest_comparisons']['dandi:sha2-256']['independent_content_digest_verified']
    summary = module.rebuild_summary([{'metadata_branches': {'top_level': branch}, 'acquisition_status': 'success'}])
    assert summary['eligible_specimens'] == summary['independently_content_verified_assets'] == 0
    assert summary['branch_and_candidate_comparison_counts']['top_level.activity_candidates'] == {'true': 1, 'unknown': 1}


def test_type_preserving_comparison_rejects_bool_integer_change():
    with pytest.raises(ValueError):
        module.equal({'flag': True}, {'flag': 1})


def test_failure_record_not_dropped():
    summary = module.rebuild_summary([{'metadata_branches': {}, 'acquisition_status': 'failed'}])
    assert summary['assets'] == 1 and summary['assets_with_successful_metadata'] == 0
    assert summary['acquisition_status_counts'] == {'failed': 1}
