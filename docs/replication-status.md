# Replication status

## What the 2008 model contains

The Methods section describes two hierarchically organized thalamocortical
loops. Each contains a six-layer cortical module, core and matrix thalamic
cells, local inhibitory cells, and TRN; the primary loop additionally contains
a nonspecific thalamic nucleus. Most sheets are 9×9. Cells use the minimum
unbranched compartments and Hodgkin–Huxley-type currents needed for their role.
Synapses use normalized dual exponentials, activity-dependent transmitter
depletion, and gated learning.

## Current milestone: M0 scaffold

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
- unit tests plus an optional Brian2 smoke test.

Recovered but not yet fully encoded:

- the original Elsevier supplementary connection table (called Supplementary
  Table 4 in the article and Supplementary Table 3 in its own caption).

Not yet implemented or validated:

- multicompartment dynamics that consume all Table 3 compartments and axial
  resistances (the current Brian2 population consumes paper-specific soma
  values only when `cell_class` is supplied);
- the supplementary table's complete projection/conductance matrix in code;
- all paper currents (T-type Ca, AHP, cholinergic modulation, and cell-specific variants);
- the full two-loop 9×9 network, topographic kernels, STDP, ACh vigilance,
  CSD/LFP geometry, and every published figure protocol;
- quantitative reproduction of match/gamma and mismatch/beta/reset.

Accordingly, output from M0 is a software/analysis benchmark, not evidence that
the 2008 results have already been replicated.

## Validation gates

A milestone may be marked complete only when:

1. each equation and parameter has a provenance status;
2. deterministic tests pass and stochastic results reproduce across declared seeds;
3. target figure protocols and readouts are documented before tuning;
4. expected and negative-control outcomes are both reported;
5. generated data and summaries include the exact configuration fingerprint.
