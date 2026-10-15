"""Fixed synthetic timing only; no recording, parameter table or response access."""

from __future__ import annotations

import argparse
import fcntl
import json
import time
from pathlib import Path

import numpy as np
from patchseq_precision_driver import FIELDS, PrecisionDriver, digest
from run_patchseq_isolated_fit import checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    with (output/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        driver = PrecisionDriver(output/'build')
        pins = {str(Path(__file__)): digest(__file__)}
        path = output/'checkpoint.json'
        parameters = dict(zip(FIELDS, [100., 10., -65., 2000., 500., 10., -60., 1000.], strict=True))
        initial = [-65., .05, .6, .2, .05]
        current = np.r_[np.zeros(167), np.full(167, 125.), np.full(167, -50.)]
        state = {'source_hashes': pins, 'build': driver.provenance, 'precision_probes': driver.probes,
                 'parameters': parameters, 'initial': initial, 'samples': 501, 'rate_hz': 50000.,
                 'current_sha256': __import__('hashlib').sha256(current.tobytes()).hexdigest(),
                 'cases': {}, 'biological_data_access': False}
        if path.exists():
            saved = json.loads(path.read_text())
            if any(saved[k] != json.loads(json.dumps(v)) for k, v in state.items() if k != 'cases'):
                raise ValueError('synthetic benchmark provenance differs')
            state = saved
        for bits in (53, 160, 320):
            for factor in (4, 8, 16):
                key = f'{bits}/{factor}'
                if key in state['cases']:
                    continue
                start = time.perf_counter()
                values = driver.integrate(parameters, current, 50000., initial, bits, factor)
                elapsed = time.perf_counter()-start
                state['cases'][key] = {'elapsed_s': elapsed, 'native_steps': 500,
                    'RK4_steps': 500*factor, 'maximum_abs_exported_voltage_mV': float(abs(values[:, 0]).max())}
                checkpoint(path, state)
        state['status'] = 'synthetic-timing-complete-not-scientific-validation'
        checkpoint(path, state)
        print(json.dumps(state['cases'], indent=2), flush=True)


if __name__ == '__main__':
    main()
