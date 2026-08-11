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
- a source-specific Figure 19 kernel assay: both paper and ModelDB profiles
  reproduce deeper AHP after two events and fast ACh suppression. The paper
  80/100-ms profile is within 2 mV of its matched control at 500 ms; the
  archived 80/150-ms, weight-4.5 profile is not and remains a documented source
  discrepancy rather than a tuned-away result;
- a predeclared Figure 8 isolated-relay protocol and qualitative tonic/burst
  scorer.
- recovered KInNeSS 2008 equation semantics, including the `Simple tau`
  function and exact directional axial Equation 7;
- separate paper and executable AHP profiles, plus source-specific Figure 8
  cell geometry and channel densities;
- explicit specific-capacitance selection instead of a hidden default.
- all 12 first-order populations from `SMART.nml` instantiated at their source
  sizes: 812 cells and 1,950 compartments. This intrinsic-only sector uses the
  executable geometry, edge-serialized KInNeSS axial values, population-specific
  reversal potentials, and the network's 5/20-ms layer-5/layer-6II AHP profile.
- a separate integrity-pinned derived catalog of all 55 executable ModelDB
  projections (51 chemical and four electrical) plus 11 external channels;
- projection-specific receptor ports for all 51 ModelDB chemical records, including
  AMPA/GABA dual-exponential kinetics, NMDA voltage block, and the equal-tau
  alpha-function case;
- the 50 chemical projections whose source and target are both in the
  first-order sector, with exact ModelDB delays, weights, asymptotes,
  one/many/all methods, Gaussian sigmas, wrap/extend borders, and ring flags;
- three in-scope gap-junction records using KInNeSS Equation 8 geometry; one
  cross-area electrical record and two V2 chemical sources are reserved for
  the higher-order loop;
- all ten conductance-based KInNeSS input channels with four 0--255 source
  dimensions and Equation 5 driving-potential semantics;
- source-wide transmitter depletion on layer 5, layer 6II, and layer 6I using
  the serialized recovery/depletion values;
- typed ModelDB learning metadata and the exact piecewise Equation 6
  postsynaptic spike gate, including each record's 20/25-ms depotentiation
  interval; modifiable projections now initialize at their serialized
  asymptotic baseline and retain the separate maximum weight;
- KInNeSS Equations 25/28 adaptive-weight dynamics for the serialized
  presynaptically gated, postsynaptically gated, and dual-AND-gated rules;
  rule-level tests verify potentiation for pre-before-post and depression for
  post-before-pre timing;
- the remaining direct-current external channel using KInNeSS Equation 6
  current-density semantics; its four archived sensitivities are all zero and
  are therefore retained as an inert protocol-controlled input;
- CI validation split into independent Brian2 processes: 163 lightweight tests,
  five sector-construction tests, six connectivity tests, and two long AHP
  tests currently pass (176 total).

Explicitly unresolved source anomalies:

- four supplementary records retain ambiguity flags rather than guessed
  corrections: a literal `N` receptor label, a delay printed as `01`, an NMDA
  record printed with -70 mV reversal, and a plasticity tuple printed in the
  Gaussian-spread row;
- ModelDB backup 112923 has now been recovered and integrity-pinned; its raw
  files are not vendored because the archive has no explicit redistribution license.
- the paper's printed sodium activation coefficient differs by a factor of ten
  from the standard Traub--Miles form, and its calcium steady-state expressions
  are greater than one over physiological voltages unless interpreted as
  reciprocals;
- the Figure 8 caption does not report the hyperpolarizing clamp voltage or
  exact epoch durations, and the legacy Figure 8 XML omits leak and membrane
  capacitance values.

Not yet implemented or validated:

- direct Brian2 instantiation of the supplementary catalog (the executable
  baseline instead uses the distinct, integrity-pinned ModelDB catalog while
  retaining the supplement as an independent audit);
- exact legacy meaning of the KInNeSS `ring` flag (currently retained as a
  center-excluding Gaussian candidate), driven network validation of Equation
  5/6 learning and its bounds, ACh vigilance,
  CSD/LFP geometry, and every published figure protocol;
- quantitative reproduction of match/gamma and mismatch/beta/reset.

The first Figure 8 candidate (67 mV-shifted standard Traub--Miles rates,
reciprocal T gates, Table 3 calcium density, and a -80 mV pre-pulse clamp)
produces sustained tonic trains in both conditions. It therefore fails the
published transient-burst signature and is retained as a negative validation
result, not as a reproduction.

That candidate predates recovery of the dedicated `Ca_rebound.xml` model and
is now classified as a useful failed paper-only reconstruction. The recovered
file resolves the T-gate equation roles and supplies a different Figure 8 cell;
validation of that executable-source profile is the active M2 task.

The latest source-specific candidate now reproduces the transient
hyperpolarized burst signature, but not the depolarized tonic train. This
improvement followed correction of T-gate initialization into the absolute
voltage coordinate. It remains a partial result and is not promoted to a
reproduction claim.

Accordingly, output from M2 is a source/audit benchmark, not evidence that
the 2008 results have already been replicated.

## Validation gates

A milestone may be marked complete only when:

1. each equation and parameter has a provenance status;
2. deterministic tests pass and stochastic results reproduce across declared seeds;
3. target figure protocols and readouts are documented before tuning;
4. expected and negative-control outcomes are both reported;
5. generated data and summaries include the exact configuration fingerprint.
