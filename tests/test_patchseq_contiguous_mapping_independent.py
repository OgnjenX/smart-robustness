"""Independent forward runs and source-index corruption checks."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from test_patchseq_contiguous_mapping import module as collector
from test_patchseq_setting_mapping import fixture, run

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('independent_block', scripts/'verify_patchseq_contiguous_mapping.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)

PAIR = {'sweep_number': 7, 'response': '/acquisition/data_00099_AD0', 'stimulus': '/stimulus/presentation/data_00100_DA0'}


def test_forward_blocks_with_gaps_other_sweeps_and_missing_modes():
    assert module.last_block(np.array([7, 7, 8, 7, 7, 7]), np.array([0, 0, 0, np.nan, 0, 0]), 7) == [4, 5]
    assert module.last_block(np.array([7, 7]), np.array([1, np.nan]), 7) == []


@pytest.mark.parametrize('mutation', [None, 'earlier-block', 'source-row', 'prior-outcome', 'export-value'])
def test_source_index_and_original_outcome_corruption_rejects(mutation):
    table, data, metadata = fixture()
    data = {key: np.repeat(values[:1], 3, axis=0) for key, values in data.items()}
    data['column-0001'][:, 0] = [0, 1, 0]
    prior = run(table, data, metadata)
    view, selected = collector.lookup_view(table, data, 7)
    saved = {'original_strict_outcome_verbatim': prior, 'historical_range': selected,
             'continuation_outcome': run(table, view, metadata)}
    saved = copy.deepcopy(saved)
    if mutation == 'earlier-block':
        saved['historical_range']['selected_acquisition_rows'] = [0]
    elif mutation == 'source-row':
        saved['continuation_outcome']['channel_setting_rows']['ADC'] = 0
    elif mutation == 'prior-outcome':
        saved['original_strict_outcome_verbatim']['mapping_status'] = 'invented'
    elif mutation == 'export-value':
        saved['continuation_outcome']['fields']['bias_current']['expected_float32_value'] += 1e-8
    if mutation is None:
        assert module.verify_one(table, data, PAIR, metadata, saved, prior) == 1
    else:
        with pytest.raises(ValueError):
            module.verify_one(table, data, PAIR, metadata, saved, prior)
