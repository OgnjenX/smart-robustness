# ADR 0004: Controlled stochastic GIF substitution

Status: accepted and preregistered before GIF network execution (2026-09-10)

## Decision

The GIF comparison uses the Pozzorini--Mensi generalized integrate-and-fire
family: a leaky somatic membrane with two spike-triggered current components,
two spike-triggered threshold components, reset and absolute refractoriness,
and an exponential stochastic escape rate.

Only the somatic fast Na/K spike generator is replaced. SMART morphology,
passive parameters, dendritic channels, axial dynamics, T-type calcium,
AHP/ACh, conductance ports, depletion, STDP, topology, weights, delays,
stimuli, and analysis gates remain fixed.

The numerical literature arm uses the excitatory and inhibitory means reported
in Setareh et al. (2017), Table 1, which were extracted from Mensi et al.
(2012). Absolute source thresholds and resets are converted to offsets from
the source leak reversal before transfer to each SMART cell. No source-paper
heterogeneity is added. The thalamic mapping is explicitly a fixed transfer
rule, not a biological parameter claim.

Because firing is stochastic, the deterministic AdEx assessment cannot be
reused. Twenty network seeds, isolated fitting seeds, sealed seed holdouts,
aggregation rules, and progression thresholds are fixed in the manifests
before any GIF SMART-network result is observed.

## Consequences

- Brian2 remains suitable because it exposes the exact hazard, reset, history
  kernels, conductance currents, and compartment equations.
- The model tests spike-history effects and stochastic spike initiation that
  AdEx does not represent in the same way.
- A failed literature transfer is a result about that registered arm, not all
  possible GIF parameterizations.
- Network outcomes may never be used to tune GIF parameters. A new mechanistic
  hypothesis would require a separately labeled registration.

## Primary sources

- Mensi et al. (2012), doi:10.1152/jn.00408.2011.
- Pozzorini et al. (2015), doi:10.1371/journal.pcbi.1004275.
- Setareh et al. (2017), doi:10.3389/fncom.2017.00052.
