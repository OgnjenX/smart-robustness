"""Synthetic two-input diagnostic gates and worker path; no real data access."""

import fcntl
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import run_patchseq_precision_diagnostic as runner
import verify_patchseq_precision_diagnostic as audit
from patchseq_precision_driver import FIELDS, SOURCE, PrecisionDriver, digest


def test_fixed_comparison_scope_and_missing_failures():
    assert len(runner.pairs()) == len(set(runner.pairs())) == 15
    data = {f'{b}-{f}': {} for b in (53, 160, 320) for f in (4, 8, 16)}
    comparisons = {a+'/'+b: {'passed': True} for a, b in runner.pairs()}
    assert runner.qualified(data, comparisons) and audit.reference_qualified(data, comparisons)
    for key in data:
        failed = {**data, key: None}
        assert not runner.qualified(failed, comparisons)
        assert not audit.reference_qualified(failed, comparisons)
    comparisons['160-8/160-16']['passed'] = False
    assert not runner.qualified(data, comparisons)
    assert not audit.reference_qualified(data, comparisons)


def test_full_warmup_waveform_not_only_step_counts():
    left = {'voltage': np.zeros(10), 'features': {'events': 0}, 'loss': 0.}
    right = {**left, 'voltage': np.r_[.1, np.zeros(9)]}
    assert not runner.comparison(left, right)['passed']
    assert not audit.compare(left, right)['passed']
    assert not runner.comparison(None, left)['passed']


def test_source_subset_exact_not_expanded():
    records = [{'source_batch': 'synthetic', 'slot': 0,
                'sweeps': [{'sweep_number': i} for i in (1, 2, 3)]}]
    keys = {('synthetic', 0, 1), ('synthetic', 0, 3)}
    subset = runner.restricted_records(records, keys)
    assert [s['sweep_number'] for s in subset[0]['sweeps']] == [1, 3]
    assert len(records[0]['sweeps']) == 3
    with pytest.raises(ValueError, match='support'):
        runner.restricted_records(records, {('synthetic', 0, 4)})


@pytest.mark.parametrize('defect', ['none', 'shape', 'initial', 'nan', 'gate'])
def test_saved_state_validation(defect):
    initial = np.array([-65., .1, .2, .3, .4])
    current = np.zeros(3)
    values = np.tile(initial, (3, 1))
    if defect == 'shape':
        values = values[:2]
    elif defect == 'initial':
        values[0, 0] = -64.
    elif defect == 'nan':
        values[1, 0] = np.nan
    elif defect == 'gate':
        values[1, 1] = 1.1
    assert audit.valid_states(values, current, initial) == (defect == 'none')


def test_terminal_snapshot_rejects_live_or_partial_process(tmp_path):
    path = tmp_path/'runner.lock'
    path.touch()
    (tmp_path/'checkpoint.json').write_text(json.dumps({'status': audit.TERMINAL}))
    with path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError), audit.terminal(tmp_path):
            pass
    (tmp_path/'checkpoint.json').write_text(json.dumps({'status': 'partial'}))
    with pytest.raises(ValueError, match='not terminal'), audit.terminal(tmp_path):
        pass


def test_worker_synthetic_arrays_and_orphan_protection(tmp_path):
    from test_patchseq_precision_prototype import oracle

    try:
        driver = PrecisionDriver(tmp_path/'build')
    except RuntimeError as exc:
        pytest.skip(str(exc))
    initial = np.array([-65., .05, .6, .2, .05])
    parameters = [100., 10., -65., 2000., 500., 10., -60., 1000.]
    current = np.r_[np.full(10, -50.), np.full(10, 125.), np.zeros(11)]
    inp, output, spec_path = tmp_path/'input.npz', tmp_path/'out.npz', tmp_path/'spec.json'
    np.savez_compressed(inp, current=current, initial=initial)
    spec = {'source_hashes': {str(SOURCE): digest(SOURCE)}, 'input': str(inp),
            'input_sha256': digest(inp), 'build_directory': str(tmp_path/'build'),
            'build': driver.provenance, 'parameters': dict(zip(FIELDS, parameters, strict=True)),
            'rate_hz': 50000., 'bits': 160, 'factor': 4, 'output': str(output)}
    spec_path.write_text(json.dumps(spec))
    runner.worker(spec_path)
    with np.load(output, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['states'], oracle(parameters, current, initial, 4))
    with pytest.raises(FileExistsError):
        runner.worker(spec_path)
    spec['input_sha256'] = 'changed'
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='command/initial'):
        runner.worker(spec_path)
