"""Independent verifier catches missing candidates and false promotions."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


def load(directory, name):
    path = Path(__file__).resolve().parents[1] / directory / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verifier = load('scripts', 'verify_patchseq_file_cross_reference')
collector = load('scripts', 'cross_reference_patchseq_files')
fixtures = load('tests', 'test_patchseq_file_cross_reference')


@pytest.mark.parametrize('fault', ['none', 'duplicate', 'other-modality', 'unknown', 'different-cell'])
def test_relational_reconstruction_agrees(fault):
    rows, assets = fixtures.fixture()
    if fault == 'duplicate':
        row = copy.deepcopy(rows[0])
        row['source_row_number'] = 3
        rows.append(row)
    elif fault == 'other-modality':
        rows[0]['raw_fields']['technique'] = 'morphology'
    elif fault == 'unknown':
        rows[0]['recognized_file_session'] = None
    elif fault == 'different-cell':
        rows[0]['canonical_cell_specimen_id'] = 457
    verifier.compare(verifier.reconstruct(rows, assets), collector.cross_reference(rows, assets))


@pytest.mark.parametrize('key', ['source_rows', 'assets', 'summary', 'unreferenced_source_row_numbers',
                               'duplicate_source_file_sessions', 'raw_value_counts', 'missing_field_counts'])
def test_altered_evidence_rejected(key):
    expected = verifier.reconstruct(*fixtures.fixture())
    altered = copy.deepcopy(expected)
    altered[key] = None
    with pytest.raises(ValueError, match=key):
        verifier.compare(expected, altered)


def test_unsupported_promotion_rejected():
    expected = verifier.reconstruct(*fixtures.fixture())
    altered = copy.deepcopy(expected)
    altered['assets'][0]['biological_eligibility_established'] = True
    with pytest.raises(ValueError, match='assets'):
        verifier.compare(expected, altered)


@pytest.mark.parametrize('name', ['123.nwb', 'sub-9_ses-123_icephys.nwb', '001.nwb', 'folder/123.nwb', 'abc.nwb'])
def test_independent_filename_decomposition(name):
    assert verifier.session(name) == collector.file_session(name)
