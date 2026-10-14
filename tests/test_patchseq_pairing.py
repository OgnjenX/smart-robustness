"""Synthetic pairing metadata only; no biological arrays or outcomes."""

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
spec = importlib.util.spec_from_file_location('pairing', Path(__file__).resolve().parents[1] / 'scripts/check_patchseq_pairing.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def attr(value):
    return {'present': True, 'scalar': {'value': value}}


def fixture():
    result = []
    for root, declared, unit in [('/acquisition/r', 'CurrentClampSeries', 'volts'),
                                  ('/stimulus/presentation/s', 'CurrentClampStimulusSeries', 'amperes')]:
        result.extend([
            {'path': root, 'object_type': 'group', 'attributes': {
                'neurodata_type': attr(declared), 'namespace': attr('core'),
                'sweep_number': attr(3), 'stimulus_description': attr('long-square')}},
            {'path': root+'/data', 'object_type': 'dataset', 'link_type': 'HardLink',
             'shape': [100], 'dtype': 'float32', 'attributes': {
                 'unit': attr(unit), 'conversion': attr(0.0010000000474974513), 'offset': {'present': False}}},
            {'path': root+'/starting_time', 'object_type': 'dataset', 'link_type': 'HardLink',
             'scalar_metadata': {'value': 2.5}, 'attributes': {'rate': attr(50000.0), 'unit': attr('Seconds')}}])
    return result


def test_exact_scaling_and_visible_spelling_deviation():
    outcome = module.evaluate(fixture())
    assert outcome['metadata_pair_passes'] == 1
    assert all(x['conversion'] == 0.0010000000474974513 for x in outcome['series'])
    assert all(x['exporter_time_spelling_deviation'] for x in outcome['series'])
    assert outcome['physiology_eligibility_established'] is False


@pytest.mark.parametrize('value', [None, True, -1, 4294967296, 3.0, '3'])
def test_invalid_sweep_never_inferred_from_path(value):
    rows = fixture()
    rows[0]['attributes']['sweep_number'] = attr(value)
    out = module.evaluate(rows)
    assert out['metadata_pair_passes'] == 0
    assert out['unkeyed_series_paths'] == ['/acquisition/r']


def test_duplicate_partner_not_selected():
    rows = fixture()
    duplicate = copy.deepcopy(rows[:3])
    for row in duplicate:
        row['path'] = row['path'].replace('/acquisition/r', '/acquisition/r2')
    assert module.evaluate(rows+duplicate)['metadata_pair_nonpasses'] == 1


def test_conflicting_ID_locations_not_ignored():
    rows = fixture()
    rows.append({'path': '/acquisition/r/sweep_number', 'scalar_metadata': {'value': 4}})
    assert module.evaluate(rows)['metadata_pair_passes'] == 0


@pytest.mark.parametrize('target,field,value', [(1, 'conversion', float('nan')), (1, 'conversion', 0),
    (1, 'unit', 'mV'), (1, 'offset', 0), (2, 'rate', 0), (2, 'unit', 'SECONDS'), (0, 'namespace', 'custom')])
def test_invalid_metadata_blocks_pair(target, field, value):
    rows = fixture()
    rows[target]['attributes'][field] = attr(value)
    assert module.evaluate(rows)['metadata_pair_passes'] == 0


@pytest.mark.parametrize('field,value', [('starting_time', 2.5000000001), ('rate', 49999.9),
                                       ('description', 'Long-square'), ('data_shape', [99])])
def test_no_clock_or_description_tolerance(field, value):
    rows = fixture()
    if field == 'starting_time':
        rows[5]['scalar_metadata']['value'] = value
    elif field == 'rate':
        rows[5]['attributes']['rate'] = attr(value)
    elif field == 'description':
        rows[3]['attributes']['stimulus_description'] = attr(value)
    else:
        rows[4]['shape'] = value
    assert module.evaluate(rows)['metadata_pair_passes'] == 0


def test_timestamps_and_duplicate_paths_require_new_validation():
    rows = fixture()
    rows.append({'path': '/acquisition/r/timestamps'})
    assert module.evaluate(rows)['metadata_pair_passes'] == 0
    with pytest.raises(ValueError, match='duplicate'):
        module.evaluate(rows+[rows[0]])


def test_non_current_clamp_retained_as_nontarget():
    rows = fixture()
    rows[0]['attributes']['neurodata_type'] = attr('VoltageClampSeries')
    out = module.evaluate(rows)
    assert out['metadata_pair_passes'] == 0
    assert out['nontarget_series'][0]['declared_type'] == 'VoltageClampSeries'
