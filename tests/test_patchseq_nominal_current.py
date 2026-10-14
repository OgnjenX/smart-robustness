"""Unknown current is not zero; source-derived nominal offsets keep raw traces."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
from test_patchseq_current_context import evaluate, inputs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    from qualify_patchseq_nominal_current import qualify
    from verify_patchseq_nominal_current import reconstruct
    from verify_patchseq_whole_file_integrity import same
finally:
    sys.path.pop(0)


def fixture():
    table, arrays, metadata = inputs()
    col = table['column_candidates']['I-Clamp Holding Enable'][0]
    table['column_headers'][col] = [{}, {'decoded_text': 'On/Off'}, {}]
    return table, arrays, evaluate(table, arrays, metadata)


def result(table, arrays, mapping):
    actual = qualify(table, arrays, mapping)
    same(actual, reconstruct(table, arrays, mapping))
    return actual


def test_native_holding_value_not_rounded_scaled_or_raw_source_modified():
    table, arrays, mapping = fixture()
    before = {k: v.tobytes() for k, v in arrays.items()}
    out = result(table, arrays, mapping)
    assert out['nominal_context_supported'] is True
    assert out['nominal_holding_offset_pA'] == 1.123456789
    assert all(v.tobytes() == before[k] for k, v in arrays.items())


def test_explicit_disabled_holding_has_zero_nominal_term_not_filled_native_value():
    table, arrays, mapping = fixture()
    col = table['column_candidates']['I-Clamp Holding Enable'][0]
    arrays[f'column-{col:04d}'][:, 0] = 0
    mapping['holding_state'] = 'disabled'
    mapping['mapping']['fields']['bias_current']['enabled_state']['value'] = 0.
    mapping['mapping']['fields']['bias_current']['native_value'] = {'undefined': 'nan'}
    assert result(table, arrays, mapping)['nominal_holding_offset_pA'] == 0.
    assert mapping['mapping']['fields']['bias_current']['native_value'] == {'undefined': 'nan'}


@pytest.mark.parametrize('case', ['unknown-holding', 'unknown-slow', 'active-slow', 'bad-export',
                                  'nonfinite-level', 'wrong-unit', 'changing-enable',
                                  'changing-level', 'nonfinite-source', 'absent-source'])
def test_unsupported_current_never_defaults_to_zero(case):
    table, arrays, mapping = fixture()
    holding = mapping['mapping']['fields']['bias_current']
    col = table['column_candidates']['I-Clamp Holding Level'][0]
    if case == 'unknown-holding':
        mapping['holding_state'] = 'unknown'
    elif case == 'unknown-slow':
        mapping['slow_state'] = 'unknown'
    elif case == 'active-slow':
        mapping['slow_state'] = 'enabled'
    elif case == 'bad-export':
        holding['concordance'] = 'discordant'
    elif case == 'nonfinite-level':
        holding['native_value']['value'] = np.inf
    elif case == 'wrong-unit':
        holding['native_unit'] = 'amperes'
    elif case == 'changing-enable':
        col = table['column_candidates']['I-Clamp Holding Enable'][0]
        arrays[f'column-{col:04d}'][0, 0] = 0.
    elif case == 'changing-level':
        arrays[f'column-{col:04d}'][0, 0] = 0.
    elif case == 'nonfinite-source':
        arrays[f'column-{col:04d}'][0, 0] = np.inf
    else:
        arrays[f'column-{col:04d}'][:, 0] = np.nan
    out = result(table, arrays, mapping)
    assert out['nominal_context_supported'] is False
    assert out['nominal_holding_offset_pA'] is None


def test_missing_annotation_rows_are_not_zero_observations_or_earlier_fallback():
    table, arrays, mapping = fixture()
    col = table['column_candidates']['I-Clamp Holding Level'][0]
    arrays[f'column-{col:04d}'][0, 0] = np.nan
    out = result(table, arrays, mapping)
    assert out['nominal_context_supported'] is True
    assert out['source_consistency']['I-Clamp Holding Level']['observed_source_rows'] == [1]
    mapping['historical_range']['selected_acquisition_rows'] = [0]
    assert result(table, arrays, mapping)['nominal_context_supported'] is False


def test_independent_verifier_rejects_filled_unknown_or_rescaled_offset():
    table, arrays, mapping = fixture()
    expected = result(table, arrays, mapping)
    altered = copy.deepcopy(expected)
    altered['nominal_holding_offset_pA'] *= 1e12
    with pytest.raises(ValueError):
        same(altered, reconstruct(table, arrays, mapping))
