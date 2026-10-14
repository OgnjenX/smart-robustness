"""Relational verifier tests preserve ambiguities and reject altered evidence."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/verify_patchseq_2021_file_references.py'
spec = importlib.util.spec_from_file_location('independent_files', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sources():
    rows = [{'source_row_number': 2, 'cells': {
        'file_name': {'literal_text': 'sub-123_ses-456_icephys.nwb'},
        'technique': {'literal_text': 'intracellular_electrophysiology'}},
        'canonical_integers': {'cell_specimen_id': 789, 'file_size': 1024}}]
    assets = [{'asset_id': 'asset', 'path': 'sub-123/sub-123_ses-456_icephys.nwb', 'size': 1024}]
    july = [{'source_row_number': 2, 'canonical_ids': {'cell_specimen_id': 789, 'ephys_session_id': 456}}]
    return rows, assets, july


def test_positive_reference_is_not_internal_identity():
    result = module.rebuild(*sources())
    assert result['summary']['complete_external_file_reference_agreements'] == 1
    assert not result['assets'][0]['internal_recording_identity_confirmed']
    assert result['summary']['candidate_flag_counts']['cross_source_checksum_agreement']['unknown'] == 1


@pytest.mark.parametrize('name', ['0456.nwb', 'sub-0123_ses-456_icephys.nwb', '../456.nwb', '456.nwb.extra', None])
def test_noncanonical_filenames_not_repaired(name):
    assert module.name_tokens(name) is None


def test_legacy_session_reference_incomplete():
    rows, assets, july = sources()
    rows[0]['cells']['file_name']['literal_text'] = '456.nwb'
    result = module.rebuild(rows, assets, july)
    assert result['summary']['complete_external_file_reference_agreements'] == 0
    assert result['assets'][0]['manifest_candidates'][0]['filename_subject_agrees'] is None


def test_duplicate_candidates_and_other_modalities_retained():
    rows, assets, july = sources()
    extra = copy.deepcopy(rows[0])
    extra['source_row_number'] = 3
    rows.append(extra)
    ignored = copy.deepcopy(rows[0])
    ignored['source_row_number'] = 4
    ignored['cells']['technique']['literal_text'] = 'morphology'
    rows.append(ignored)
    result = module.rebuild(rows, assets, july)
    assert result['duplicate_manifest_sessions'] == [456]
    assert len(result['assets'][0]['manifest_candidates']) == 2
    assert result['unreferenced_manifest_source_row_numbers'] == [4]
    assert result['summary']['complete_external_file_reference_agreements'] == 0


@pytest.mark.parametrize('field', ['size', 'cell', 'subject', 'annotation_duplicate'])
def test_conflict_or_ambiguity_not_accepted(field):
    rows, assets, july = sources()
    if field == 'size':
        rows[0]['canonical_integers']['file_size'] = None
    elif field == 'cell':
        rows[0]['canonical_integers']['cell_specimen_id'] = 111
    elif field == 'subject':
        rows[0]['cells']['file_name']['literal_text'] = 'sub-111_ses-456_icephys.nwb'
    else:
        duplicate = copy.deepcopy(july[0])
        duplicate['source_row_number'] = 3
        july.append(duplicate)
    assert module.rebuild(rows, assets, july)['summary']['complete_external_file_reference_agreements'] == 0


def test_modified_result_rejected():
    expected = module.rebuild(*sources())
    observed = copy.deepcopy(expected)
    observed['assets'][0]['manifest_candidates'][0]['file_size_agrees'] = 1
    with pytest.raises(ValueError):
        module.compare(expected, observed)
