"""External file references cannot silently resolve conflicting candidates."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/cross_reference_patchseq_2021_files.py'
spec = importlib.util.spec_from_file_location('file_references', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixtures():
    rows = [{'source_row_number': 2,
             'cells': {'file_name': {'literal_text': 'sub-123_ses-456_icephys.nwb'},
                       'technique': {'literal_text': 'intracellular_electrophysiology'}},
             'canonical_integers': {'cell_specimen_id': 789, 'file_size': 1024, 'file_id': 99}}]
    assets = [{'asset_id': 'asset', 'path': 'sub-123/sub-123_ses-456_icephys.nwb', 'size': 1024}]
    july = [{'source_row_number': 2, 'canonical_ids': {'cell_specimen_id': 789, 'ephys_session_id': 456}}]
    return rows, assets, july


def test_exact_external_reference_does_not_qualify_internal_identity():
    result = module.compare(*fixtures())
    asset = result['assets'][0]
    assert asset['complete_external_file_reference_agreement']
    assert asset['manifest_candidates'][0]['cross_source_checksum_agreement'] is None
    assert not asset['internal_recording_identity_confirmed']
    assert not asset['specimen_eligibility_established']


@pytest.mark.parametrize('change', ['subject', 'size', 'cell', 'legacy', 'no_cell', 'no_size', 'formula', 'modality', 'no_annotation'])
def test_conflicts_unknowns_and_partial_references_not_promoted(change):
    rows, assets, july = fixtures()
    if change == 'subject':
        rows[0]['cells']['file_name']['literal_text'] = 'sub-999_ses-456_icephys.nwb'
    elif change == 'size':
        rows[0]['canonical_integers']['file_size'] = 2048
    elif change == 'cell':
        rows[0]['canonical_integers']['cell_specimen_id'] = 999
    elif change == 'legacy':
        rows[0]['cells']['file_name']['literal_text'] = '456.nwb'
    elif change == 'no_cell':
        rows[0]['canonical_integers']['cell_specimen_id'] = None
    elif change == 'no_size':
        rows[0]['canonical_integers']['file_size'] = None
    elif change == 'formula':
        rows[0]['cells']['file_name']['literal_text'] = None
    elif change == 'modality':
        rows[0]['cells']['technique']['literal_text'] = 'morphology'
    else:
        july = []
    result = module.compare(rows, assets, july)
    assert not result['assets'][0]['complete_external_file_reference_agreement']
    if change == 'subject':
        assert result['assets'][0]['manifest_candidates'][0]['filename_subject_agrees'] is False


@pytest.mark.parametrize('source', ['rows', 'july_cell', 'july_session'])
def test_all_ambiguous_candidates_retained(source):
    rows, assets, july = fixtures()
    if source == 'rows':
        duplicate = copy.deepcopy(rows[0])
        duplicate['source_row_number'] = 3
        rows.append(duplicate)
    else:
        duplicate = copy.deepcopy(july[0])
        duplicate['source_row_number'] = 3
        if source == 'july_cell':
            duplicate['canonical_ids']['ephys_session_id'] = 999
        else:
            duplicate['canonical_ids']['cell_specimen_id'] = 999
        july.append(duplicate)
    result = module.compare(rows, assets, july)
    asset = result['assets'][0]
    assert not asset['complete_external_file_reference_agreement']
    if source == 'rows':
        assert len(asset['manifest_candidates']) == 2
    elif source == 'july_cell':
        assert len(asset['manifest_candidates'][0]['July_cell_source_row_numbers']) == 2
    else:
        assert len(asset['July_session_source_row_numbers']) == 2


@pytest.mark.parametrize('path', ['sub-123/sub-999_ses-456_icephys.nwb', '../456.nwb', 'sub-0123/sub-0123_ses-456_icephys.nwb'])
def test_bad_catalog_path_rejected(path):
    with pytest.raises(ValueError):
        module.path_tokens(path)


def test_unreferenced_other_modalities_preserved():
    rows, assets, july = fixtures()
    extra = copy.deepcopy(rows[0])
    extra['source_row_number'] = 3
    extra['cells']['technique']['literal_text'] = 'transcriptomics'
    rows.append(extra)
    assert module.compare(rows, assets, july)['unreferenced_manifest_source_row_numbers'] == [3]
