"""Independent relational comparisons detect corrupted revision evidence."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


def load(folder, name):
    path = Path(__file__).resolve().parents[1] / folder / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verifier = load('scripts', 'verify_patchseq_annotation_revisions')
collector = load('scripts', 'compare_patchseq_annotation_revisions')
fixtures = load('tests', 'test_patchseq_annotation_revisions')


@pytest.mark.parametrize('fault', ['none', 'different-label', 'duplicate-June', 'unknown-cell', 'unknown-session', 'duplicate-July'])
def test_reconstructs_raw_revision_changes(fault):
    july, june = fixtures.fixture()
    if fault == 'different-label':
        july[0]['raw_fields']['T-type Label'] = 'different'
    elif fault == 'duplicate-June':
        row = copy.deepcopy(june[0])
        row['source_row_number'] = 4
        june.append(row)
    elif fault in ('unknown-cell', 'unknown-session'):
        key = 'cell_specimen_id' if fault == 'unknown-cell' else 'ephys_session_id'
        july[0]['canonical_ids'][key] = None
        july[0]['raw_fields'][key] = ''
    elif fault == 'duplicate-July':
        row = copy.deepcopy(july[0])
        row['source_row_number'] = 5
        july.append(row)
    verifier.compare(verifier.reconstruct(july, june), collector.compare_rows(july, june))


@pytest.mark.parametrize('key', ['July_rows', 'comparisons', 'summary', 'raw_MET_label_counts',
                               'raw_field_change_counts', 'July_duplicate_IDs', 'July_noncanonical_ID_counts',
                               'unreferenced_June_rows_by_cell'])
def test_altered_projection_rejected(key):
    expected = verifier.reconstruct(*fixtures.fixture())
    altered = copy.deepcopy(expected)
    altered[key] = None
    with pytest.raises(ValueError, match=key):
        verifier.compare(expected, altered)


def test_false_biological_promotion_rejected():
    expected = verifier.reconstruct(*fixtures.fixture())
    altered = copy.deepcopy(expected)
    altered['comparisons'][0]['biological_eligibility_established'] = True
    with pytest.raises(ValueError, match='comparisons'):
        verifier.compare(expected, altered)
