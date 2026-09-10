# ADR 0003: preserve SMART morphology while substituting somatic AdEx dynamics

## Status

Accepted for the first controlled neuron-model comparison. The immutable
`classic-smart-calibrated-v1.0.0` control remains unchanged.

## Context

Replacing each SMART cell with a point AdEx neuron would alter several factors
at once: dendritic integration, receptor placement, axial propagation,
thalamic rebound currents, electrical coupling, and the currents used for the
higher-order field calculation. A difference in ART behavior would then not be
attributable to the spike generator.

The scientific question is instead whether the frozen SMART phenomena survive
a controlled change in intrinsic cellular dynamics, and under which
independently selected neuron parameters.

## Decision

Phase 1 replaces the somatic fast Na/K spike generator with adaptive
exponential integrate-and-fire dynamics. It retains the source-backed SMART
compartments, passive membrane, axial coupling, compartment-targeted synapses,
NMDA block, thalamic T-current, SMART AHP/ACh, depletion, learning, topology,
delays, protocols, and analyses. The active layer-5 distal Na/K mechanism is
also retained so somatic spike substitution is not confounded with removal of
active dendritic propagation.

The experiment has three arms: frozen classic control, one literature-derived
AdEx parameter set, and cell-class-specific AdEx parameters fitted only to
isolated classic-cell responses. The isolated-cell fit must be sealed before
any alternative-network result is inspected. Network outputs are holdouts and
cannot contribute to the fitting objective.

This intervention is named a **morphology-preserving somatic AdEx
substitution**. It is not described as a pure point-AdEx SMART implementation.

## Consequences

If ART behavior changes, Phase 1 supports attribution to somatic spike and
adaptation dynamics under otherwise preserved SMART machinery. Retained
T-current and AHP/ACh can interact with AdEx adaptation; those interactions are
part of the declared hybrid intervention and will be reported. Later phases
may remove active dendritic or calcium mechanisms, use GIF/HH variants, add
noise ensembles, or update anatomy, but each requires a separate registration.
