"""Current settings use last source blocks, exact units and no favorable fallback."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
from test_patchseq_setting_mapping import fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    from map_patchseq_current_context import context
    from verify_patchseq_current_context import verify_context
finally:
    sys.path.pop(0)

PAIR = {'sweep_number': 7, 'response': '/acquisition/data_00099_AD0',
        'stimulus': '/stimulus/presentation/data_00100_DA0'}


def inputs():
    table, data, metadata = fixture()
    for label, unit, val in [('Slow current injection', 'On/Off', 0),
                              ('Slow current injection level', 'V', -.065),
                              ('Slow current injection settling time', 's', 1.)]:
        k = len(data)
        table['column_candidates'][label] = [k]
        table['column_headers'][k] = [{}, {'decoded_text': unit}, {}]
        array = np.full((2, 9), np.nan)
        array[:, 0] = val
        data[f'column-{k:04d}'] = array
    return table, data, metadata


def evaluate(table, data, metadata):
    result = context(table, data, 7, PAIR['response'], PAIR['stimulus'], metadata)
    verify_context(table, data, PAIR, metadata, result)
    return result


def test_native_pA_holding_and_V_slow_target_not_added_to_command():
    table, data, metadata = inputs()
    before = {key: value.tobytes() for key, value in data.items()}
    result = evaluate(table, data, metadata)
    assert result['holding_state'] == 'enabled' and result['slow_state'] == 'disabled'
    assert result['slow_injection']['Slow current injection level']['native_value']['value'] == -.065
    assert result['slow_injection']['Slow current injection level']['observed_unit'] == 'V'
    assert all(value.tobytes() == before[key] for key, value in data.items())


@pytest.mark.parametrize('case', ['missing-mode', 'inactive', 'duplicate-headstage', 'duplicate-column',
                                  'missing-slow', 'nonfinite-slow', 'wrong-enable-unit', 'enabled-slow',
                                  'bad-export', 'disabled-holding', 'missing-identity', 'all-NaN-last-block'])
def test_independent_unknown_and_state_branches(case):
    table, data, metadata = inputs()
    slow = table['column_candidates']['Slow current injection'][0]
    if case == 'missing-mode':
        data['column-0001'][:] = np.nan
    elif case == 'inactive':
        data['column-0005'][:, 0] = 0
    elif case == 'duplicate-headstage':
        for k in (2, 3, 4):
            data[f'column-{k:04d}'][:, 1] = data[f'column-{k:04d}'][:, 0]
    elif case == 'duplicate-column':
        table['column_candidates']['ADC'] = [2, 3]
    elif case == 'missing-slow':
        table['column_candidates']['Slow current injection'] = []
    elif case == 'nonfinite-slow':
        data[f'column-{slow:04d}'][:, 0] = np.inf
    elif case == 'wrong-enable-unit':
        table['column_headers'][slow][1]['decoded_text'] = 'pA'
    elif case == 'enabled-slow':
        data[f'column-{slow:04d}'][:, 0] = 1
    elif case == 'bad-export':
        metadata['bias_current']['value'] = 0.
    elif case == 'disabled-holding':
        k = table['column_candidates']['I-Clamp Holding Enable'][0]
        data[f'column-{k:04d}'][:, 0] = 0
        metadata['bias_current'] = {'present': False, 'undefined': 'absent'}
    elif case == 'missing-identity':
        table['column_candidates']['SweepNum'] = []
    else:
        data = {k: np.repeat(v[:1], 3, axis=0) for k, v in data.items()}
        data['column-0001'][:, 0] = [0, 1, 0]
        data[f'column-{slow:04d}'][2] = np.nan
    result = evaluate(table, data, metadata)
    if case in ('missing-slow', 'nonfinite-slow', 'wrong-enable-unit', 'all-NaN-last-block'):
        assert result['slow_state'] == 'unknown'
    elif case == 'enabled-slow':
        assert result['slow_state'] == 'enabled'
    elif case == 'bad-export':
        assert result['holding_state'] == 'unknown'
    elif case == 'disabled-holding':
        assert result['holding_state'] == 'disabled'
    else:
        assert result['mapping']['mapping_status'] == 'unknown'


@pytest.mark.parametrize('field', ['state', 'native-value', 'source-block', 'export-value'])
def test_corrupt_native_mapping_not_accepted(field):
    table, data, metadata = inputs()
    result = evaluate(table, data, metadata)
    altered = copy.deepcopy(result)
    if field == 'state':
        altered['slow_state'] = 'enabled'
    elif field == 'native-value':
        altered['slow_injection']['Slow current injection level']['native_value']['value'] *= 1e12
    elif field == 'source-block':
        altered['historical_range']['selected_acquisition_rows'] = [0]
    else:
        altered['mapping']['fields']['bias_current']['expected_float32_value'] = 0.
    with pytest.raises(ValueError):
        verify_context(table, data, PAIR, metadata, altered)
