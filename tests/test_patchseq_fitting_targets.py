"""Held-out responses remain sealed and source failures cannot become targets."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    from build_patchseq_fitting_targets import characterize_case, plan
    from patchseq_response_features import characterize, events
finally:
    sys.path.pop(0)


def case(partition='fitting'):
    slot = {'slot': 1, 'partition': partition, 'source_cell_specimen_id': '42'}
    pair = {'slot': 1, 'source_cell_specimen_id': '42', 'sweeps': [{'sweep_number': 7}]}
    wave = {'slot': 1, 'source_cell_specimen_id': '42', 'sweeps': [{'sweep_number': 7, 'sweep_QC_pass': True}]}
    health = {'slot': 1, 'mandatory_gate': 'pass'}
    nominal = {'slot': 1, 'source_cell_specimen_id': '42', 'sweeps': [{'sweep_number': 7, 'nominal_context_supported': True}]}
    return slot, pair, wave, health, nominal


def test_held_out_reader_never_called_even_with_passing_source_metadata():
    def forbidden(*args):
        pytest.fail('held-out response file or array accessed')
    out = characterize_case(*case('held-out'), forbidden)
    assert out == [{'sweep_number': 7, 'state': 'held-out-not-characterized'}]
    assert 'features' not in out[0]


@pytest.mark.parametrize('reason', ['quality', 'waveform', 'context'])
def test_failed_source_gates_retained_without_response_access(reason):
    values = case()
    if reason == 'quality':
        values[3]['mandatory_gate'] = 'unknown'
    elif reason == 'waveform':
        values[2]['sweeps'][0]['sweep_QC_pass'] = False
    else:
        values[4]['sweeps'][0]['nominal_context_supported'] = False
    out = characterize_case(*values, lambda *a: pytest.fail('excluded response read'))
    assert len(out) == 1 and 'features' not in out[0]
    assert out[0]['state'] != 'eligible-fitting-response'


def test_training_reader_exception_retained_not_replaced_or_zero_filled():
    def failed(*args):
        raise ValueError('unsupported-rate')
    out = characterize_case(*case(), failed)
    assert out[0]['state'] == 'fitting-response-unresolved'
    assert 'features' not in out[0]


def test_case_identity_and_duplicate_sweeps_rejected_before_reader():
    values = case()
    values[4]['source_cell_specimen_id'] = '43'
    with pytest.raises(ValueError):
        plan(*values)
    values = case()
    values[1]['sweeps'].append(copy.deepcopy(values[1]['sweeps'][0]))
    with pytest.raises(ValueError):
        plan(*values)


def pulse(spikes=()):
    rate, start, stop = 50000., 50000, 100000
    v = np.full(150000, -65.)
    i = np.full_like(v, 10.)
    i[start:stop] = 110.
    for delay in spikes:
        index = start+int(delay*rate)
        v[index:index+20] = np.linspace(-65., 35., 20)
        v[index+20:index+100] = np.linspace(35., -65., 80)
    epoch = {'step_start_index': start, 'step_stop_index': stop, 'pre_window': [24999, 49999]}
    return v, i, rate, epoch


def test_clean_no_event_response_and_nonzero_holding_do_not_modify_voltage_or_command():
    v, i, rate, epoch = pulse()
    v[50000:100000] = -55.
    before = v.tobytes(), i.tobytes()
    out = characterize(v, i, rate, epoch, -20.)
    assert out['detected_event_count'] == 0 and out['first_event_latency_s'] is None
    assert out['last_to_first_interval_ratio'] is None
    assert out['source_command_delta_pA'] == 100.
    assert out['nominal_input_baseline_pA'] == -10. and out['nominal_input_step_pA'] == 90.
    assert out['apparent_steady_resistance_MOhm'] == pytest.approx(100.)
    assert before == (v.tobytes(), i.tobytes())


def test_three_synthetic_events_have_indices_rates_and_interval_ratio():
    v, i, rate, epoch = pulse((.1, .3, .6))
    out = characterize(v, i, rate, epoch, 0.)
    assert out['detected_event_count'] == 3 and out['detected_event_rate_hz'] == 3.
    assert out['first_event_latency_s'] == pytest.approx(.1, abs=.001)
    assert out['last_to_first_interval_ratio'] == pytest.approx(1.5, abs=.02)
    assert out['apparent_steady_resistance_MOhm'] is None
    assert out['event_peak_indices'] == [55019, 65019, 80019]


def test_subthreshold_fast_transient_is_not_promoted_to_event():
    v, i, rate, epoch = pulse((.1,))
    v = -65.+(v+65.)*.1
    out = characterize(v, i, rate, epoch, 0.)
    assert out['detected_event_count'] == 0


def test_fixed_filter_rate_limit_nonfinite_and_nonstep_inputs_fail_closed():
    v, i, rate, epoch = pulse()
    with pytest.raises(ValueError):
        characterize(v, i, 20000., epoch, 0.)
    v[1] = np.nan
    with pytest.raises(ValueError):
        characterize(v, i, rate, epoch, 0.)
    v[1] = -65.
    i[60000] += 1.
    with pytest.raises(ValueError):
        characterize(v, i, rate, epoch, 0.)


def test_detector_only_uses_declared_step_window():
    v, _i, rate, _epoch = pulse((.1, .6))
    assert len(events(v, rate, 70000, 100000)[0]) == 1
