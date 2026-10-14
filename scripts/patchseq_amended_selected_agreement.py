"""Three-engine selected checks; compiled primary cannot certify itself."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from patchseq_compiled_fit_engine import adapter, numerical_controls
from patchseq_hh_compiled_driver import compiled_driver
from patchseq_hh_independent_replay import replay as independent_replay
from patchseq_isolated_hh import Parameters
from patchseq_response_features import characterize
from patchseq_selected_fit_agreement import feature_agreement
from patchseq_selected_fit_agreement import selected_agreement as brian_numpy_agreement
from verify_patchseq_fitting_targets import reconstruct

BUILD = Path('results/patchseq-amended-selected-fit-1474/build')


def selected_agreement(parameter, command, rate, holding, epoch, saved_control):
    result = brian_numpy_agreement(parameter, command, rate, holding, epoch, saved_control)
    if not result['all_reconstruction_checks_pass']:
        return result
    try:
        with compiled_driver(BUILD) as (compiled, build):
            v = compiled([parameter], command, rate, holding, 1.)[0]
            independent = independent_replay([parameter], command, rate, holding, 1.)[0]
            features = characterize(v, command, rate, epoch, holding)
            controls = numerical_controls(adapter(compiled), Parameters(**parameter), command,
                                          rate, holding, epoch, 1.)
            error = float(np.max(np.abs(v-independent)))
            checks = {'compiled_waveform_vs_independent': error <= 1e-5,
                      'compiled_features_vs_independent': feature_agreement(
                          features, reconstruct(independent, command, rate, epoch, holding)),
                      'compiled_controls_vs_saved': feature_agreement(controls, saved_control),
                      'compiled_controls_vs_independent': feature_agreement(
                          controls, result['independent_numerical_controls'])}
            result['agreement_checks'].update(checks)
            result.update(all_reconstruction_checks_pass=all(result['agreement_checks'].values()),
                          compiled_build=build, maximum_compiled_independent_voltage_error_mV=error)
            return result
    except (ValueError, FloatingPointError, OverflowError) as exc:
        return {'state': 'selected-sweep-reconstruction-unresolved',
                'all_reconstruction_checks_pass': False, 'error': str(exc), 'network_promotion': False}
