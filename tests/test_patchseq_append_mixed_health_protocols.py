"""Mixed health metadata retains unsupported trials rather than repairing them."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
from test_patchseq_append_health_protocols import CODES, attr, health_fixture, module

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    from verify_patchseq_append_mixed_health_protocols import reconstruct
    from verify_patchseq_whole_file_integrity import same
finally:
    sys.path.pop(0)


@pytest.mark.parametrize('case', ['valid', 'CC-bath', 'missing-partner', 'rate', 'unit',
                                  'dtype', 'shape', 'conversion', 'offset', 'timestamps',
                                  'unkeyed', 'duplicate-response', 'nontarget', 'blowout'])
def test_independent_mixed_reconstruction_matches_retained_failures(case):
    rows = health_fixture()
    if case == 'CC-bath':
        rows[0]['attributes']['neurodata_type'] = attr('CurrentClampSeries')
        rows[3]['attributes']['neurodata_type'] = attr('CurrentClampStimulusSeries')
        rows[1]['attributes']['unit'], rows[4]['attributes']['unit'] = attr('volts'), attr('amperes')
    elif case == 'missing-partner':
        rows = rows[:3]
    elif case == 'rate':
        rows[5]['attributes']['rate'] = attr(49999.)
    elif case == 'unit':
        rows[1]['attributes']['unit'] = attr('mV')
    elif case == 'dtype':
        rows[1]['dtype'] = 'complex64'
    elif case == 'shape':
        rows[1]['shape'] = [2, 2]
    elif case == 'conversion':
        rows[1]['attributes']['conversion'] = attr(float('nan'))
    elif case == 'offset':
        rows[1]['attributes']['offset'] = attr(0.)
    elif case == 'timestamps':
        extra = copy.deepcopy(rows[1])
        extra['path'] = rows[0]['path']+'/timestamps'
        rows.append(extra)
    elif case == 'unkeyed':
        rows[0]['attributes']['sweep_number'] = attr(True)
    elif case == 'duplicate-response':
        for item in copy.deepcopy(rows[:3]):
            item['path'] = item['path'].replace('/acquisition/r', '/acquisition/duplicate')
            rows.append(item)
    elif case == 'nontarget':
        for index in (0, 3):
            rows[index]['attributes']['stimulus_description'] = attr('Test')
    elif case == 'blowout':
        rows[0]['attributes']['neurodata_type'] = attr('CurrentClampSeries')
        rows[3]['attributes']['neurodata_type'] = attr('CurrentClampStimulusSeries')
        rows[1]['attributes']['unit'], rows[4]['attributes']['unit'] = attr('volts'), attr('amperes')
        for index in (0, 3):
            rows[index]['attributes']['stimulus_description'] = attr('EXTPBLWOUT141203_DA_0')
    same(reconstruct(rows, CODES), module.evaluate(rows, CODES))
    if case == 'CC-bath':
        out = reconstruct(rows, CODES)
        assert out['health_pairs'][0]['metadata_pair_pass'] is False
        assert 'bath' in out['missing_protocols']
        assert out['health_series'][0]['clamp'] == 'CC'


def test_favorable_promotion_of_unsupported_trial_is_rejected():
    rows = health_fixture()
    rows[0]['attributes']['neurodata_type'] = attr('CurrentClampSeries')
    expected = reconstruct(rows, CODES)
    promoted = copy.deepcopy(expected)
    promoted['health_pairs'][0].update(metadata_pair_pass=True, failures=[])
    with pytest.raises(ValueError):
        same(expected, promoted)
