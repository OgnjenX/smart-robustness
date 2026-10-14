"""Pure training-stage classification, never authority to read held-out responses."""

from __future__ import annotations


def assess_training(original, independent, selected, protocol):
    sha = independent.get('input_checkpoint_sha256')
    if (independent.get('status') != 'all-candidate-forward-feature-loss-reconstruction-retained'
        or not sha or set(independent['strata']) != set(protocol['strata'])):
        raise ValueError('independent training evidence incomplete')
    if (selected.get('status') != 'selected-training-waveform-feature-and-numerical-checks-retained'
        or selected.get('input_checkpoint_sha256') != sha
        or set(selected['strata']) != set(protocol['strata'])):
        raise ValueError('selected training evidence incomplete or different source snapshot')
    if set(original['strata']) != set(protocol['strata']):
        raise ValueError('original training scope differs')
    for evidence in (independent, selected):
        if evidence.get('held_out_access_authorized') is not False or evidence.get('network_promotion') is not False:
            raise ValueError('training evidence exceeds access or promotion boundary')
    count = protocol['search']['candidates_per_stratum']
    keys = {str(i) for i in range(count)}
    outcomes = []
    for stratum in protocol['strata']:
        reference = original['strata'][stratum]
        replayed = independent['strata'][stratum]['candidates']
        checked = selected['strata'][stratum]
        if set(reference['candidates']) != keys or set(replayed) != keys:
            raise ValueError('candidate table coverage differs')
        if any(type(row['passed']) is not bool for row in replayed.values()):
            raise ValueError('replay agreement must be explicit boolean')
        chosen = reference['selected_candidate']
        if type(chosen) is not int or str(chosen) not in keys or checked['summary']['selected_candidate'] != chosen:
            raise ValueError('selected candidate differs')
        sweeps = checked['sweeps']
        if set(sweeps) != set(reference['numerical_controls']) or not sweeps:
            raise ValueError('selected sweep coverage differs')
        if any(type(r['all_reconstruction_checks_pass']) is not bool for r in sweeps.values()):
            raise ValueError('selected reconstruction claim must be boolean')
        agreement = all(row['all_reconstruction_checks_pass'] for row in sweeps.values())
        if agreement and any(type(row['independent_numerical_controls']['passed']) is not bool
                             for row in sweeps.values()):
            raise ValueError('numerical control claim must be explicit boolean')
        numerical = agreement and all(row['independent_numerical_controls']['passed'] is True
                                     for row in sweeps.values())
        training = reference['candidates'][str(chosen)]['all_training_observable_gates_pass']
        if type(training) is not bool:
            raise ValueError('original training gate must be boolean')
        derived = {'all_selected_reconstructions_agree': agreement,
                   'all_selected_numerical_controls_pass': numerical,
                   'sealed_training_observable_gates_pass': training,
                   'training_and_numerical_eligible': agreement and numerical and training}
        if any(type(checked['summary'][name]) is not bool or checked['summary'][name] != value
               for name, value in derived.items()):
            raise ValueError('selected summary inconsistent with retained evidence')
        table_agrees = all(row['passed'] for row in replayed.values())
        if not table_agrees:
            category = 'verification-disagreement-not-biological-failure'
        elif not agreement:
            category = 'selected-reconstruction-unresolved'
        elif not numerical:
            category = 'selected-numerical-gate-failed'
        elif not training:
            category = 'bounded-selected-observable-fit-failure'
        else:
            category = 'training-eligible-for-separately-sealed-held-out-design'
        outcomes.append({'stratum': stratum, 'selected_candidate': chosen,
                         'classification': category, **derived,
                         'full_candidate_reconstruction_agrees': table_agrees,
                         'eligible_to_seal_held_out_design': table_agrees and agreement and numerical and training})
    aggregate_agreement = all(o['full_candidate_reconstruction_agrees'] for o in outcomes)
    if independent.get('all_reconstructions_agree') is not aggregate_agreement:
        raise ValueError('aggregate replay claim inconsistent')
    return {'status': 'training-stage-classification-only', 'strata': outcomes,
            'complete_map_training_eligible': all(o['eligible_to_seal_held_out_design'] for o in outcomes),
            'held_out_access_authorized': False, 'network_promotion': False,
            'biological_validation_complete': False, 'scientific_goal_complete': False,
            'source_integrity_verified_by_this_callable': False}
