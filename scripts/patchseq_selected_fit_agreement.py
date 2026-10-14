"""Selected-map cross-engine checks; no parameter selection or source-file reads."""

from __future__ import annotations

import math

import numpy as np
from patchseq_fit_search import replay as primary_replay
from patchseq_hh_independent_replay import replay as independent_replay
from patchseq_isolated_hh import Parameters
from patchseq_response_features import characterize
from verify_patchseq_fitting_targets import reconstruct


def feature_agreement(primary, independent):
    """Exact structure/discrete features, bounded arithmetic-only float differences."""
    if type(primary) is not type(independent):
        return False
    if isinstance(primary, dict):
        return primary.keys() == independent.keys() and all(
            feature_agreement(primary[k], independent[k]) for k in primary)
    if isinstance(primary, list):
        return len(primary) == len(independent) and all(
            feature_agreement(a, b) for a, b in zip(primary, independent, strict=True))
    if type(primary) is float:
        return (math.isfinite(primary) and math.isfinite(independent)
                and abs(primary-independent) <= 1e-5+1e-9*abs(primary))
    return primary == independent


def selected_agreement(parameter, command, rate, holding, epoch, saved_control):
    """Reconstruct one selected training sweep under the sealed numerical controls."""
    try:
        reference = primary_replay([Parameters(**parameter)], command, rate, holding, 1.)[0]
        original = independent_replay([parameter], command, rate, holding, 1.)[0]
        repeated = independent_replay([parameter], command, rate, holding, 1.)[0]
        refined = independent_replay([parameter], command, rate, holding, 1., refinement=2)[0]
        longer = independent_replay([parameter], command, rate, holding, 2.)[0]
        primary_features = characterize(reference, command, rate, epoch, holding)
        fs = [reconstruct(v, command, rate, epoch, holding) for v in (original, refined, longer)]
        voltage_error = float(np.max(np.abs(reference-original)))
        refinement_error = float(np.max(np.abs(original-refined)))
        warmup_error = abs(fs[0]['baseline_mean_mV']-fs[2]['baseline_mean_mV'])
        gates = {'exact_repeat': bool(np.array_equal(original, repeated)),
                 'refinement_voltage': refinement_error <= .1,
                 'refinement_event_count': fs[0]['detected_event_count'] == fs[1]['detected_event_count'],
                 'warmup_baseline': warmup_error <= .1,
                 'warmup_event_count': fs[0]['detected_event_count'] == fs[2]['detected_event_count']}
        control = {'gates': gates, 'maximum_refinement_voltage_error_mV': refinement_error,
                   'warmup_baseline_difference_mV': warmup_error, 'passed': all(gates.values())}
        flags = {'waveform_agreement': voltage_error <= 1e-5,
                 'feature_agreement': feature_agreement(primary_features, fs[0]),
                 'saved_control_agreement': feature_agreement(saved_control, control)}
        return {'state': 'selected-sweep-reconstructed', 'agreement_checks': flags,
                'all_reconstruction_checks_pass': all(flags.values()),
                'independent_numerical_controls': control,
                'maximum_cross_engine_voltage_error_mV': voltage_error,
                'primary_features': primary_features, 'independent_features': fs[0],
                'network_promotion': False}
    except (ValueError, FloatingPointError, OverflowError) as exc:
        # A recorded primary failure is not proof of independently matching failure.
        return {'state': 'selected-sweep-reconstruction-unresolved',
                'all_reconstruction_checks_pass': False, 'error': str(exc),
                'recorded_primary_control_failed': saved_control.get('passed') is False,
                'network_promotion': False}
