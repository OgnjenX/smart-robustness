# Relay calcium-state diagnostics

This readout investigates the unresolved Figure 7 spatial-selection failure.
It is not a new neuron model, parameter fit, or certification of classic SMART.
Registration 392 replays the comparator-free calibrated configuration used in
Artifact 250, adding continuous state recording only.

## Run the registered replay

From the repository root, with the project environment installed:

```sh
.venv/bin/python scripts/run_figure7_aligned_on_center_mismatch.py \
  --profile configs/calibration/figure7_aligned_on_center_mismatch_v1.yaml \
  --output docs/validation-results/figure7-relay-continuous-trace-393.yaml \
  --diagnostic-registration docs/validation-results/relay-continuous-trace-registration-392.yaml \
  --pre-event-offsets-ms 2 1 0.5 0.2 \
  --relay-trace-output tmp/figure7-relay-continuous-trace-393.npz
```

The trace destination's parent must exist. An existing trace is never
overwritten. For an additional repeat, choose new result and trace filenames
and identify it as a repeat, not a newly registered experiment. The compressed
trace is local, under ignored `tmp/`; its SHA-256 is included in the result.

## Verify and summarize

Only after the replay finishes:

```sh
.venv/bin/python scripts/audit_figure7_relay_events.py \
  --input docs/validation-results/figure7-relay-continuous-trace-393.yaml \
  --reference docs/validation-results/figure7-mismatch-pre-event-trace-audit-250.yaml \
  --trace tmp/figure7-relay-continuous-trace-393.npz \
  --output docs/validation-results/figure7-relay-continuous-summary-394.yaml
```

Analysis requires the trace checksum, runtime fingerprint, and complete trial
event trains for relay, TRN, nonspecific, and category populations to match
the saved reference. It validates array shapes, required calcium variables,
finite values, and unit labels. A failed check stops interpretation rather than
silently accepting a different simulation.

The archive loads with `numpy.load(path, allow_pickle=False)`. Each state array
has axes `[recorded cell, time]`; `cell_indices` and `time_ms` label those axes.
Negative times precede sensory onset. The `monitor_when` and `monitor_order`
fields record Brian2 sampling phase. Voltage is in mV, current in pA, and
calcium activation/inactivation gates are dimensionless. Both dendrites are
recorded, together with the existing relay current and voltage diagnostics.

Continuous histories can reveal calcium availability and temporal relations.
They do not alone prove calcium's causal necessity, establish a physiological
burst label, or turn spatially incorrect output into a reproduction. An
emitted falling-phase event must not be confused with spike initiation.

## Registered causal diagnostic

Registration 396 switches off both relay dendritic T-type conductances at
sensory onset, after the unchanged cue. It affects all first-order relay cells;
recurrent network consequences are part of this intervention. It is neither a
source-faithful model variant nor a baseline candidate. Scoring explicitly
rejects reproduction eligibility for an ablated condition.

To execute that diagnostic, use the replay command above with the following
changes (choose new filenames if repeating an already completed run):

- `--diagnostic-registration docs/validation-results/relay-calcium-ablation-registration-396.yaml`
- `--output docs/validation-results/figure7-relay-calcium-ablation-397.yaml`
- `--relay-trace-output tmp/figure7-relay-calcium-ablation-397.npz`
- add `--ablate-relay-calcium-at-stimulus`.

After completion, verify the control and describe its outcome:

```sh
.venv/bin/python scripts/audit_relay_calcium_ablation.py \
  --control docs/validation-results/figure7-relay-continuous-trace-393.yaml \
  --intervention docs/validation-results/figure7-relay-calcium-ablation-397.yaml \
  --output docs/validation-results/relay-calcium-ablation-comparison-398.yaml
```

Unlike the repeatability audit, this comparison permits changed trial events.
It requires identical cue events and recorded pre-stimulus states, matching
configuration metadata and trace checksums, and exactly zero sampled relay
calcium current throughout the sensory epoch. The result is descriptive of
this calibrated network intervention, not proof about original SMART or a
cell-autonomous rebound mechanism.
