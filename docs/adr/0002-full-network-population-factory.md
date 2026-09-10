# ADR 0002: inject the population factory into complete SMART builders

## Status

Accepted after the calibrated behavioral freeze. This decision does not alter
the frozen `classic-smart-calibrated-v1.0.0` control.

## Context

The reduced benchmark runner already has a named neuron-model registry, but the
source-constrained first-order and V1–pulvinar–V2 builders instantiated the
classic multicompartment population directly. That made the validated circuit
executable, but it did not provide a controlled substitution point for the
planned neuron-model robustness matrix.

The complete circuit cannot use an arbitrary Brian2 `NeuronGroup`. Synapse,
stimulus, learning, reset, recording, and compartment analyses depend on the
population adapter contract used by `CompartmentalPopulation`: a Brian2 group,
declared compartments, compiled receptor/input port metadata, and the existing
input/ACh methods where applicable.

## Decision

The complete builders accept a keyword-only `population_factory`. The builder
calls it once per source population with exactly four keyword arguments:

- `name`: stable area-qualified population name;
- `size`: source-backed population size;
- `params`: the complete resolved SMART cell and port parameter mapping;
- `brian`: the selected Brian2 module/context.

The classic `create_compartmental_hh_population` function remains the default.
The factory parameter is forwarded through intrinsic, chemical, connected, and
voltage-clamp first-order builders and is also used for all 24 populations in
the complete V1–pulvinar–V2 network.

Alternative AdEx, GIF, point-HH, or multicompartment backends must implement a
compatible population adapter. They must not silently reinterpret missing
ports or alter topology. Any deliberate port reduction belongs to a separately
preregistered experiment, not the first neuron-model comparison matrix.

## Consequences

The frozen classic behavior is unchanged because callers that omit the factory
still execute the same function with the same resolved parameters. Robustness
experiments can inject a single alternative cellular backend end-to-end while
holding the circuit and protocol fixed. Tests record all factory calls and
verify that the first-order and complete builders preserve source population
counts and topology through this boundary.
