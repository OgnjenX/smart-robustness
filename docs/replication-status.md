# Replication status

## What the 2008 model contains

The Methods section describes two hierarchically organized thalamocortical
loops. Each contains a six-layer cortical module, core and matrix thalamic
cells, local inhibitory cells, and TRN; the primary loop additionally contains
a nonspecific thalamic nucleus. Most sheets are 9×9. Cells use the minimum
unbranched compartments and Hodgkin–Huxley-type currents needed for their role.
Synapses use normalized dual exponentials, activity-dependent transmitter
depletion, and gated learning.

## Current milestone: M1 source specification

Implemented:

- reproducible YAML configuration and fingerprints;
- a complete machine-readable transcription of the 12 Table 3 cell classes,
  their 30 compartments, geometry, passive properties, and channel densities;
- unit-checked conversion from Table 3 membrane densities to total soma
  capacitances and conductances;
- a Brian2 classic-HH reference population with Traub–Miles-style Na/K gates;
- conductance-based double-exponential AMPA/GABA-like synapses;
- presynaptic resource depletion and recovery;
- a minimal thalamus–cortex–TRN/reset benchmark;
- rate and spectral analysis with predeclared beta/gamma bands;
- unit tests plus an optional Brian2 smoke test;
- all 55 nonblank recovered Supplementary Table 3 connection records as a
  typed, validated catalog: 49 chemical projections, four gap junctions, and
  two external inputs;
- raw and parsed supplementary values, stable record IDs, deterministic
  serialization, and per-record verification status;
- an executable validation-target registry and a source-strength classification
  for the published figure claims;
- an accepted vectorized-multicompartment Brian2 architecture decision.

Explicitly unresolved source anomalies:

- four supplementary records retain ambiguity flags rather than guessed
  corrections: a literal `N` receptor label, a delay printed as `01`, an NMDA
  record printed with -70 mV reversal, and a plasticity tuple printed in the
  Gaussian-spread row;
- the original KInNeSS/NeuroML archive has not been recovered, so initial states,
  exact connection realizations, seeds, and raw output traces remain unavailable.

Not yet implemented or validated:

- multicompartment dynamics that consume all Table 3 compartments and axial
  resistances (the current Brian2 population consumes paper-specific soma
  values only when `cell_class` is supplied);
- Brian2 instantiation of the supplementary projection catalog;
- all paper currents (T-type Ca, AHP, cholinergic modulation, and cell-specific variants);
- the full two-loop 9×9 network, topographic kernels, STDP, ACh vigilance,
  CSD/LFP geometry, and every published figure protocol;
- quantitative reproduction of match/gamma and mismatch/beta/reset.

Accordingly, output from M1 is a source/audit benchmark, not evidence that
the 2008 results have already been replicated.

## Validation gates

A milestone may be marked complete only when:

1. each equation and parameter has a provenance status;
2. deterministic tests pass and stochastic results reproduce across declared seeds;
3. target figure protocols and readouts are documented before tuning;
4. expected and negative-control outcomes are both reported;
5. generated data and summaries include the exact configuration fingerprint.
