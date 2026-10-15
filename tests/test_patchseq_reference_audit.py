"""Synthetic audit gates; no ongoing study outcomes or source bodies read."""

import fcntl
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import verify_patchseq_reference_convergence as audit


def test_original_arithmetic_tolerance_and_exact_discrete_structure():
    assert audit.agreement({'v':1.}, {'v':1.+1e-7})
    assert not audit.agreement({'n':1}, {'n':2})
    assert not audit.agreement({'n':1}, {'n':True})
    assert not audit.agreement({'a':None}, {'a':0.})


def test_incomplete_or_disagreeing_reference_never_qualifies():
    names=('DOP853-medium/DOP853-tight','Radau-medium/Radau-tight','DOP853-tight/Radau-tight')
    cases={n:{'passed':True} for n in names}
    assert audit.reference_qualified(True,cases)
    assert not audit.reference_qualified(False,cases)
    cases.pop(names[-1]);assert not audit.reference_qualified(True,cases)


def test_missing_replay_does_not_pass_comparison():
    assert audit.compare(None,None)=={'passed':False,'replays_complete':False}


def test_waveform_feature_and_loss_all_required():
    row={'voltage':np.zeros(5),'features':{'count':0},'loss':0.}
    other={'voltage':np.zeros(5),'features':{'count':1},'loss':0.}
    assert not audit.compare(row,other)['passed']
    other={'voltage':np.zeros(5),'features':{'count':0},'loss':1.}
    assert not audit.compare(row,other)['passed']


def receipt_fixture():
    study={'accuracy_levels':[{'name':'tight','rtol':1e-12,'voltage_atol':1e-12,'gate_atol':1e-14}],
           'max_step_ms':.1,'maximum_evaluations_per_replay':100,
           'input_contract':{'warmup_s':.02}}
    receipt={'method':'Radau','rtol':1e-12,'voltage_atol_mV':1e-12,'gate_atol':1e-14,
             'max_step_ms':.1,'function_evaluations':22,
             'segments':[{'first_index':0,'last_index':1002,'constant_current_pA':5.,'nfev':10},
                         {'first_index':1002,'last_index':1004,'constant_current_pA':25.,'nfev':10}]}
    return study,receipt,np.array([0.,0.,20.,20.,0.])


@pytest.mark.parametrize('mutation',['none','holding','edge','limit','accuracy'])
def test_adaptive_receipt_requires_exact_input_edges_and_holding(mutation):
    study,receipt,command=receipt_fixture()
    if mutation=='holding':receipt['segments'][0]['constant_current_pA']+=5.
    if mutation=='edge':receipt['segments'][0]['last_index']-=1
    if mutation=='limit':receipt['function_evaluations']=101
    if mutation=='accuracy':receipt['rtol']=1e-8
    assert audit.receipt_matches(receipt,'Radau-tight',command,50000.,5.,study)==(mutation=='none')


def test_live_terminal_snapshot_is_not_a_terminal_process(tmp_path):
    lock_path=tmp_path/'runner.lock';lock_path.touch()
    (tmp_path/'checkpoint.json').write_text(json.dumps({'status':audit.TERMINAL}))
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError),audit.terminal(tmp_path):
            pass


def test_incomplete_snapshot_rejected_after_lock_release(tmp_path):
    (tmp_path/'runner.lock').touch()
    (tmp_path/'checkpoint.json').write_text(json.dumps({'status':'computing'}))
    with pytest.raises(ValueError,match='not terminal'),audit.terminal(tmp_path):
        pass
