"""Code tags, not generic Test names; exact VC units and metadata pairing."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from test_patchseq_pairing import attr, fixture

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('health_protocols', scripts/'audit_patchseq_append_health_protocols.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)

CODES = {'bath': ['EXTPINBATH141203'], 'seal': ['EXTPCllATT141203'], 'breakin': ['EXTPBREAKN141203'], 'blowout': ['EXTPBLWOUT141203']}


def health_fixture():
    rows = fixture()
    rows[0]['attributes']['neurodata_type'] = attr('VoltageClampSeries')
    rows[3]['attributes']['neurodata_type'] = attr('VoltageClampStimulusSeries')
    rows[1]['attributes']['unit'], rows[4]['attributes']['unit'] = attr('amperes'), attr('volts')
    for index in (0, 3):
        rows[index]['attributes']['stimulus_description'] = attr('EXTPINBATH141203_DA_0')
    return rows


def test_exact_code_role_unit_pairing():
    out = module.evaluate(health_fixture(), CODES)
    assert out['health_pairs'][0]['metadata_pair_pass'] is True
    assert out['missing_protocols'] == ['seal', 'breakin', 'blowout']


@pytest.mark.parametrize('description', ['Test', 'EXTPINBATH141203_new_DA_0', 'extpinbath141203_DA_0'])
def test_generic_test_prefix_or_case_guess_not_allowed(description):
    rows = health_fixture()
    for index in (0, 3):
        rows[index]['attributes']['stimulus_description'] = attr(description)
    out = module.evaluate(rows, CODES)
    assert not out['health_pairs'] and len(out['nonhealth_series']) == 2


def test_clock_mismatch_wrong_role_and_missing_partner_retained():
    rows = health_fixture()
    rows[5]['attributes']['rate'] = attr(49999.)
    assert module.evaluate(rows, CODES)['health_pairs'][0]['metadata_pair_pass'] is False
    rows = health_fixture()
    rows[0]['attributes']['neurodata_type'] = attr('CurrentClampSeries')
    assert module.evaluate(rows, CODES)['health_pairs'][0]['metadata_pair_pass'] is False
    assert module.evaluate(health_fixture()[:3], CODES)['health_pairs'][0]['metadata_pair_pass'] is False


def test_official_loader_bracket_and_suffix_order():
    assert module.code('EXTPINBATH141203[2]_DA_0') == 'EXTPINBATH141203'
    assert module.code('EXTPINBATH141203_DA_0[2]') == 'EXTPINBATH141203_DA_0'
