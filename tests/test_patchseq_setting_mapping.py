"""Synthetic channel ambiguity and exact exporter serialization controls."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('setting_mapping', scripts/'map_patchseq_notebook_settings.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def fixture():
    names = ['SweepNum', 'EntrySourceType', 'ADC', 'DAC', 'Clamp Mode', 'Headstage Active']
    for label, enabled, _ in module.FIELDS.values():
        names.extend([label, enabled])
    table = {'column_candidates': {name: [k] for k, name in enumerate(names)}, 'column_headers': {}}
    arrays = {f'column-{k:04d}': np.full((2, 9), np.nan) for k in range(len(names))}
    for name, value in {'SweepNum': 7, 'EntrySourceType': 0, 'ADC': 0, 'DAC': 0,
                        'Clamp Mode': 1, 'Headstage Active': 1}.items():
        arrays[f'column-{names.index(name):04d}'][:, 0] = value
    meta = {}
    for prop, (label, enabled, unit) in module.FIELDS.items():
        arrays[f'column-{names.index(label):04d}'][:, 0] = 1.123456789
        arrays[f'column-{names.index(enabled):04d}'][:, 0] = 1
        table['column_headers'][names.index(label)] = [{}, {'decoded_text': unit}, {}]
        meta[prop] = {'present': True, 'dtype': 'float32', 'value': float(np.float32(1.123456789))}
    return table, arrays, meta


def run(table, arrays, meta):
    return module.map_pair(table, arrays, 7, '/acquisition/data_00099_AD0', '/stimulus/presentation/data_00100_DA0', meta)


def test_group_index_not_sweep_and_float32_serialization_exact():
    out = run(*fixture())
    assert out['mapping_status'] == 'mapped'
    assert all(x['concordance'] == 'agrees' for x in out['fields'].values())


@pytest.mark.parametrize('kind', ['mode', 'inactive', 'duplicate-column', 'duplicate-headstage'])
def test_ambiguities_never_silently_selected(kind):
    table, arrays, meta = fixture()
    if kind == 'mode':
        arrays['column-0001'][1, 0] = np.nan
    elif kind == 'inactive':
        arrays['column-0005'][:, 0] = 0
    elif kind == 'duplicate-column':
        table['column_candidates']['ADC'] = [2, 3]
    else:
        for k in (2, 3, 4):
            arrays[f'column-{k:04d}'][:, 1] = arrays[f'column-{k:04d}'][:, 0]
    assert run(table, arrays, meta)['mapping_status'] == 'unknown'


def test_disabled_is_absence_not_zero_and_disagreement_retained():
    table, arrays, meta = fixture()
    column = table['column_candidates']['I-Clamp Holding Enable'][0]
    arrays[f'column-{column:04d}'][:, 0] = 0
    assert run(table, arrays, meta)['fields']['bias_current']['concordance'] == 'discordant'
    meta['bias_current'] = {'present': False, 'undefined': 'absent'}
    assert run(table, arrays, meta)['fields']['bias_current']['concordance'] == 'agrees'


def test_wrong_export_value_not_tolerance_matched():
    table, arrays, meta = fixture()
    meta['bridge_balance']['value'] = np.nextafter(meta['bridge_balance']['value'], 2.)
    assert run(table, arrays, meta)['fields']['bridge_balance']['concordance'] == 'discordant'
