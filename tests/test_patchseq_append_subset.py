"""Append selection retains original failures and next-rank boundaries."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
from test_patchseq_intrinsic_subset import inputs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    from select_patchseq_append_subset import append_selection
    from select_patchseq_intrinsic_subset import select
    from verify_patchseq_append_subset import reconstruct_append
finally:
    sys.path.pop(0)


def expanded():
    base, candidates, references, manifest = inputs()
    # Add a second held-out group after the original c group.
    record = copy.deepcopy(candidates['records'][-1])
    record.update(frozen_group_id='d', source_cell_specimen_id='51', July_source_row_number=9,
                  file_path='sub-1/sub-1_ses-51_icephys.nwb')
    candidates['records'].append(record)
    reference = copy.deepcopy(references['assets'][-1])
    reference.update(path=record['file_path'], asset_id='asset-51',
                     manifest_candidates=[{'manifest_source_row_number': 9}],
                     July_session_source_row_numbers=[9])
    references['assets'].append(reference)
    row = copy.deepcopy(manifest['rows'][-1])
    row.update(source_row_number=9, canonical_integers={'cell_specimen_id': 51, 'file_size': 60})
    for key, value in [('cell_specimen_id', '51'), ('file_name', 'sub-1_ses-51_icephys.nwb')]:
        row['cells'][key].update(raw_value=value, literal_text=value)
    manifest['rows'].append(row)
    original = select(base, candidates, references, manifest)
    return base, candidates, references, manifest, original


def test_next_ranks_and_independent_doubled_prefix_agree():
    values = expanded()
    result = append_selection(*values)
    expected = reconstruct_append(*values)
    for key in expected:
        assert result[key] == expected[key]
    assert [s['frozen_group_id'] for s in result['slots']] == ['b', 'd']
    assert [s['source_cell_specimen_id'] for s in result['slots']] == ['41', '51']


def test_original_evidence_tampering_is_rejected():
    values = expanded()
    values[-1]['slots'][0]['expected_SHA256'] = 'b'*64
    with pytest.raises(ValueError):
        append_selection(*values)
    with pytest.raises(ValueError):
        reconstruct_append(*values)


def test_failed_next_rank_is_retained_not_replaced():
    values = expanded()
    # Original data unchanged; make next fitting representative oversized.
    row = values[3]['rows'][2]
    row['cells']['file_size'].update(raw_value='65', literal_text='65')
    row['canonical_integers']['file_size'] = 65
    result = append_selection(*values)
    assert result['problems']
    assert result['slots'][0]['frozen_group_id'] == 'b'
    assert 'single-recording-size-limit-exceeded' in result['slots'][0]['preflight_failures']
    with pytest.raises(ValueError):
        reconstruct_append(*values)


def test_missing_next_group_does_not_borrow_original_or_other_partition():
    values = list(expanded())
    values[1]['records'] = [r for r in values[1]['records'] if r['frozen_group_id'] != 'd']
    values[-1] = select(*values[:4])
    result = append_selection(*values)
    assert result['problems']
    assert [s['frozen_group_id'] for s in result['slots']] == ['b']
