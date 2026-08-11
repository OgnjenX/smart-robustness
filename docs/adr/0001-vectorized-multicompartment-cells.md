# ADR 0001: vectorized multicompartment SMART cells

- Status: accepted for implementation; axial convention pending validation
- Date: 2026-08-11
- Scope: classic SMART Brian2 backend

## Decision

Represent each homogeneous SMART population as one vectorized Brian2
`NeuronGroup`. One group index is one biological neuron; each published
compartment is an explicit state within that neuron, such as `v_soma`,
`v_proximal`, and `v_distal`.

The production network will not instantiate one `SpatialNeuron` per biological
cell. `SpatialNeuron` may be used as an independent passive isolated-cell
cross-check, but not as the canonical 9×9 implementation.

## Context

SMART is multicompartmental but not morphologically detailed. Table 3 defines
two- or three-compartment cells with electrically uniform soma, proximal, and
distal sections. A general cable-tree abstraction would add simulator-specific
discretization behavior that is absent from the published equations and would
require hundreds of independent Brian2 objects.

The vectorized representation keeps all 81 cells in a homogeneous sheet in one
compiled object, supports explicit reproducible `i, j` connection arrays, and
allows synapses to target named compartment/receptor ports.

## Equation boundary

For a three-compartment cell, the equation builder will emit separate balance
equations of the form

```text
C_s dV_s/dt = I_ion,s + G_sp (V_p - V_s) + I_syn,s + I_drive,s
C_p dV_p/dt = I_ion,p + G_sp (V_s - V_p) + G_pd (V_d - V_p) + I_syn,p
C_d dV_d/dt = I_ion,d + G_pd (V_p - V_d) + I_syn,d
```

Only currents present in the source specification are compiled into a cell
class. Circuit builders interact with typed compartment and receptor handles;
they do not manipulate generated Brian2 variable names directly.

Population construction is necessarily two-pass:

1. validate the complete supplementary projection catalog;
2. determine every incoming target-compartment/receptor/kinetics port;
3. compile and instantiate population equations;
4. create projection objects and explicit connectivity.

Projection-specific ports are preferred where kinetics, reversal, plasticity,
or depletion differ. This prevents multiple unrelated synapse objects from
silently sharing an incompatible conductance state.

## Axial-current ambiguity

The paper and Table 3 use conductance/resistance terminology that admits more
than one dimensional interpretation. The code will initially expose two named
conventions:

- `paper_literal`: the closest dimensional implementation of Equation 2;
- `symmetric_cable`: reciprocal edge conductance from half-compartment
  resistances in series, conserving equal-and-opposite total axial current.

The frozen classic profile will select a convention only after source recovery
or predeclared isolated-cell comparisons. Selection and any calibration must be
recorded in the provenance ledger; a generic cable convention must not be
silently substituted.

## Spike event

The classic backend will reproduce the paper's event convention rather than a
standard threshold/reset shortcut: the soma first arms after exceeding 30 mV,
then emits one synaptic spike event when it crosses 0 mV downward. Scheduling
tests must verify one event per action potential and that axonal delay starts at
that event.

## Consequences

Benefits:

- direct control of the published compartment equations and event ordering;
- efficient 9×9 sheets and compatibility with Brian2 code generation;
- explicit soma/proximal/distal synaptic targeting;
- a clean backend boundary for later AdEx, GIF, point-HH, and alternative
  multicompartment robustness experiments.

Costs and controls:

- generated equation strings are more complex, so a typed equation builder and
  unit tests are required;
- incoming ports must be known before population construction, requiring the
  two-pass build;
- passive vectorized cells must be cross-checked against an independent
  formulation, and integration convergence must be documented before tuning.

## Reproducibility requirements

Reference artifacts will preserve independent seeds for connectivity, initial
state, stimuli, electrode placement, and later perturbations. They will also
store the resolved source-specification hash, explicit connection arrays and
hash, simulator version, integration method, timestep, and configuration
fingerprint.
