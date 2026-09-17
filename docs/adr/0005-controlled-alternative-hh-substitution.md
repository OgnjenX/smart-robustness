# ADR 0005: morphology-preserving Pospischil-type somatic HH substitution

## Status

Accepted for preregistered implementation. No alternative-HH network outcome
has been observed. The immutable `classic-smart-calibrated-v1.0.0` control is
unchanged.

## Context

The completed AdEx and GIF screening changed the somatic spike generator while
retaining SMART morphology, ports, dendritic currents, topology, learning, and
analysis. None of the registered cortical substitutions preserved all Figure 6
gates. This does not distinguish dependence on explicit voltage-gated Na/K
conductances from dependence on the exact SMART somatic kinetics.

Pospischil et al. (2008, DOI `10.1007/s00422-008-0263-8`) describe minimal
conductance-based cortical models using modified Traub--Miles Na/K currents and
an optional slow M-type potassium current. Their RS and FS models provide a
source-backed alternative parameterization and fitting family. The published
models are single-compartment cells; replacing the whole SMART cell with them
would simultaneously remove dendritic integration, compartment-targeted
receptors, axial propagation, and active layer-5 dendrites.

## Decision

Implement a **morphology-preserving Pospischil-type somatic HH substitution**.
Only the somatic spike-generating Na/K kinetics, maximal conductances, threshold
shift, and optional M-current change. Retain:

- SMART Table 3 morphology, capacitance, leak, and axial coupling;
- every compartment-targeted AMPA, NMDA, GABA-A, and GABA-B port;
- active layer-5 distal Na/K channels;
- thalamic T-type calcium and SMART layer-5 AHP/ACh;
- depletion, plasticity, projections, weights, delays, stimuli, and analyses.

Parameters are selected separately for each cortical cell class from a frozen
Sobol set using only isolated classic-cell current-step targets. Training levels
select the candidate; interleaved holdout levels remain sealed until selection.
Network behavior is never a fitting objective.

The first network study contains classic, cortical-all, excitatory-only, and
inhibitory-only arms. Thalamic substitutions are excluded because the earlier
registered relay rebound target was invalid for the classic control.

## Consequences

A pass would show that exact classic somatic kinetics are unnecessary when
explicit conductance-based spike generation is retained. A failure would show
only that this source-backed family and independently selected parameter map do
not preserve the frozen gates. It would not prove that all HH variants fail or
that ART requires the exact Grossberg--Versace equations.

This intervention is not described as a pure Pospischil neuron, a point-HH
SMART network, or a replacement of SMART's dendritic mechanisms.

