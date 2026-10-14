"""Independent setting/source checks reject mapping and serialization corruption."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from test_patchseq_setting_mapping import fixture, run

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('independent_settings', scripts/'verify_patchseq_setting_mapping.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)

PAIR = {'sweep_number': 7, 'response': '/acquisition/data_00099_AD0', 'stimulus': '/stimulus/presentation/data_00100_DA0'}


def test_strict_supported_and_mixed_mode_unknowns():
    table, data, meta = fixture()
    assert module.check(table, data, PAIR, meta, run(table, data, meta))['mapped']
    data['column-0001'][1, 0] = np.nan
    proof = module.check(table, data, PAIR, meta, run(table, data, meta))
    assert not proof['mapped'] and proof['explicit_acquisition_rows_exist']


@pytest.mark.parametrize('kind', ['row', 'headstage', 'value', 'unknown'])
def test_corrupt_outcomes_rejected(kind):
    table, data, meta = fixture()
    saved = copy.deepcopy(run(table, data, meta))
    if kind == 'row':
        saved['channel_setting_rows']['ADC'] = 0
    elif kind == 'headstage':
        saved['headstage_page'] = 2
    elif kind == 'value':
        saved['fields']['bias_current']['expected_float32_value'] += 1e-9
    else:
        saved['mapping_status'] = 'unknown'
    with pytest.raises(ValueError):
        module.check(table, data, PAIR, meta, saved)
