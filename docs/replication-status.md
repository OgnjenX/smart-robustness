# Replication status

## What the 2008 model contains

The Methods section describes two hierarchically organized thalamocortical
loops. Each contains a six-layer cortical module, core and matrix thalamic
cells, local inhibitory cells, and TRN; the primary loop additionally contains
a nonspecific thalamic nucleus. Most sheets are 9×9. Cells use the minimum
unbranched compartments and Hodgkin–Huxley-type currents needed for their role.
Synapses use normalized dual exponentials, activity-dependent transmitter
depletion, and gated learning.

## Current milestone: M2 cell-kernel validation

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
- executable vectorized dynamics for all 12 Table 3 cell classes and all 30
  compartments, including explicitly selected axial-current conventions;
- source-backed Na/K and T-type calcium equations with printed and inferred
  kinetic alternatives represented as named conventions;
- layer-5 AHP and ACh dual-exponential dynamics with the unreported maximum
  AHP conductance required as an explicit calibration parameter;
- a predeclared Figure 8 isolated-relay protocol and qualitative tonic/burst
  scorer.

Explicitly unresolved source anomalies:

- four supplementary records retain ambiguity flags rather than guessed
  corrections: a literal `N` receptor label, a delay printed as `01`, an NMDA
  record printed with -70 mV reversal, and a plasticity tuple printed in the
  Gaussian-spread row;
- the original KInNeSS/NeuroML archive has not been recovered, so initial states,
  exact connection realizations, seeds, and raw output traces remain unavailable.
- the paper's printed sodium activation coefficient differs by a factor of ten
  from the standard Traub--Miles form, and its calcium steady-state expressions
  are greater than one over physiological voltages unless interpreted as
  reciprocals;
- the Figure 8 caption does not report the hyperpolarizing clamp voltage or
  exact epoch durations.

Not yet implemented or validated:

- Brian2 instantiation of the supplementary projection catalog;
- the full two-loop 9×9 network, topographic kernels, STDP, ACh vigilance,
  CSD/LFP geometry, and every published figure protocol;
- quantitative reproduction of match/gamma and mismatch/beta/reset.

The first Figure 8 candidate (67 mV-shifted standard Traub--Miles rates,
reciprocal T gates, Table 3 calcium density, and a -80 mV pre-pulse clamp)
produces sustained tonic trains in both conditions. It therefore fails the
published transient-burst signature and is retained as a negative validation
result, not as a reproduction.

Accordingly, output from M2 is a source/audit benchmark, not evidence that
the 2008 results have already been replicated.

## Validation gates

A milestone may be marked complete only when:

1. each equation and parameter has a provenance status;
2. deterministic tests pass and stochastic results reproduce across declared seeds;
3. target figure protocols and readouts are documented before tuning;
4. expected and negative-control outcomes are both reported;
5. generated data and summaries include the exact configuration fingerprint.
