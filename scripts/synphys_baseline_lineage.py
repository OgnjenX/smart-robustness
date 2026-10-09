"""Pure original-pulse metadata lineage checks; no database entry point."""

from __future__ import annotations

import math

COLUMNS = {
    "cell": ["id", "experiment_id", "electrode_id"],
    "electrode": ["id", "experiment_id"],
    "sync_rec": ["id", "experiment_id"],
    "recording": ["id", "sync_rec_id", "electrode_id"],
    "pulse_response": ["id", "recording_id", "stim_pulse_id", "pair_id", "ex_qc_pass", "in_qc_pass"],
    "patch_clamp_recording": ["id", "recording_id", "clamp_mode", "baseline_potential", "qc_pass"],
    "stim_pulse": ["id", "recording_id", "cell_id", "previous_pulse_dt", "qc_pass", "n_spikes", "first_spike_time"],
}


def positive(value):
    return type(value) is int and value > 0


def same_identity(left, right):
    """Do not let Python's bool/int or float/int equality certify a link."""
    return positive(left) and positive(right) and left == right


def join_original_pulse_metadata(pulse_ids, context, tables):
    """Keep every original contributor and all ambiguous parents.

    Matching metadata is not biological eligibility, complete waveform QC or
    permission to calculate a baseline from an unrelated cell/experiment.
    """
    pulse_ids = list(pulse_ids)
    if (not pulse_ids or len(pulse_ids) > 100000 or not all(positive(v) for v in pulse_ids)
        or len(set(pulse_ids)) != len(pulse_ids)):
        raise ValueError("bounded positive unique original pulse IDs required")
    keys = {"pair_id", "experiment_id", "pre_cell_id", "post_cell_id"}
    if set(context) != keys or not all(positive(context[key]) for key in keys):
        raise ValueError("sealed pair context differs")
    if set(tables) != set(COLUMNS):
        raise ValueError("metadata-only projection scope differs")
    indexes = {}
    for table, columns in COLUMNS.items():
        if len(tables[table]) > 100000:
            raise ValueError("metadata row bound exceeded")
        indexed = {}
        key = "recording_id" if table == "patch_clamp_recording" else "id"
        for row in tables[table]:
            if set(row) != set(columns) or not positive(row["id"]):
                raise ValueError("metadata row projection or primary ID differs")
            indexed.setdefault(row[key], []).append(row)
        indexes[table] = indexed

    def rows(table, identity):
        return indexes[table].get(identity, []) if positive(identity) else []

    def only(records):
        return records[0] if len(records) == 1 else None

    def field(row, name):
        return None if row is None else row[name]

    pre_cells = rows("cell", context["pre_cell_id"])
    post_cells = rows("cell", context["post_cell_id"])
    pre, post = only(pre_cells), only(post_cells)
    records = []
    for pulse_id in pulse_ids:
        responses = rows("pulse_response", pulse_id)
        response = only(responses)
        post_recordings = rows("recording", field(response, "recording_id"))
        post_recording = only(post_recordings)
        patches = rows("patch_clamp_recording", field(post_recording, "id"))
        patch = only(patches)
        stimuli = rows("stim_pulse", field(response, "stim_pulse_id"))
        stimulus = only(stimuli)
        pre_recordings = rows("recording", field(stimulus, "recording_id"))
        pre_recording = only(pre_recordings)
        post_syncs = rows("sync_rec", field(post_recording, "sync_rec_id"))
        pre_syncs = rows("sync_rec", field(pre_recording, "sync_rec_id"))
        post_sync, pre_sync = only(post_syncs), only(pre_syncs)
        post_electrodes = rows("electrode", field(post_recording, "electrode_id"))
        pre_electrodes = rows("electrode", field(pre_recording, "electrode_id"))
        post_electrode, pre_electrode = only(post_electrodes), only(pre_electrodes)
        parents = {"pulse_response": responses, "post_recording": post_recordings,
                   "patch_clamp_recording": patches, "stim_pulse": stimuli,
                   "pre_recording": pre_recordings, "post_sync": post_syncs, "pre_sync": pre_syncs,
                   "post_electrode": post_electrodes, "pre_electrode": pre_electrodes,
                   "pre_cell": pre_cells, "post_cell": post_cells}
        flags = {"single_" + name: len(values) == 1 for name, values in parents.items()}
        flags["selected_pair"] = response is not None and same_identity(response["pair_id"], context["pair_id"])
        for role, cell, recording, sync, electrode in (
            ("pre", pre, pre_recording, pre_sync, pre_electrode),
            ("post", post, post_recording, post_sync, post_electrode),
        ):
            flags[role + "_cell_experiment"] = cell is not None and same_identity(cell["experiment_id"], context["experiment_id"])
            flags[role + "_electrode_matches_cell"] = (cell is not None and recording is not None
                                                       and same_identity(recording["electrode_id"], cell["electrode_id"]))
            flags[role + "_electrode_experiment"] = electrode is not None and same_identity(electrode["experiment_id"], context["experiment_id"])
            flags[role + "_sync_experiment"] = sync is not None and same_identity(sync["experiment_id"], context["experiment_id"])
        flags["selected_presynaptic_cell"] = stimulus is not None and same_identity(stimulus["cell_id"], context["pre_cell_id"])
        flags["same_sync_record"] = (pre_recording is not None and post_recording is not None
                                     and same_identity(pre_recording["sync_rec_id"], post_recording["sync_rec_id"]))
        interval = field(stimulus, "previous_pulse_dt")
        state_flags = {"IC_mode": patch is not None and patch["clamp_mode"] == "ic",
                       "previous_pulse_interval_gt_8s": (type(interval) in (int, float)
                                                         and not math.isnan(interval) and interval > 8)}
        records.append({"original_pulse_response_id": pulse_id, "parents": parents,
                        "identity_flags": flags, "identity_complete": all(flags.values()),
                        "source_state_metadata_flags": state_flags,
                        "baseline_value": field(patch, "baseline_potential"),
                        "response_duration_confirmed": False,
                        "biological_eligibility_established": False})
    return {"records": records, "original_contributor_count": len(pulse_ids),
            "all_contributor_identities_complete": all(row["identity_complete"] for row in records),
            "waveform_values_read": False, "biological_eligibility_established": False}
