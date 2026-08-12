# Classic SMART reproduction roadmap

The work is ordered to expose source and cellular errors before they become
network-tuning errors. Each milestone ends in a reviewable commit or pull
request and a machine-readable validation artifact.

## M0 — reference scaffold (complete)

- Reproducible configuration and fingerprints.
- Point Hodgkin–Huxley reference backend.
- Dual-exponential synapse and depletion primitives.
- Reduced benchmark and spectral analysis.
- Complete Table 3 cell catalog: 12 cell classes and 30 compartments.

M0 is a software scaffold, not a reproduction of the published SMART network.

## M1 — official connection catalog

- Encode every nonblank record in the recovered Supplementary Table 3.
- Preserve source text alongside normalized units and parsed plasticity or
  depletion parameters.
- Validate receptor, reversal, conductance, weight density, target compartment,
  spatial rule, delay, and rise/fall constants.
- Mark ambiguous or unavailable values explicitly; never fill a blank by
  copying a neighboring value without a documented derivation.

Exit: complete catalog audit, stable IDs, source spot checks, and no unclassified
nonblank supplement cells.

## M2 — published multicompartment cells

- Implement soma/proximal/distal voltage states and passive axial coupling.
- Add cell-specific Na/K channels, thalamic T-type calcium, layer-5 AHP, and ACh
  modulation.
- Implement the paper's two-stage spike-event convention.
- Validate every cell class in isolation, including LGN tonic/rebound-burst and
  layer-5 AHP/ACh protocols.

Current checkpoint: the first-order sector builds all 812 source cells and
1,950 compartments, 50 in-scope ModelDB chemical projections, three in-scope
gap-junction records, and ten externally controllable voltage-driven input
channels. Projection-specific kinetics, delays, topology metadata, KInNeSS
input semantics, and source-wide transmitter depletion are executable. Exact
legacy ring-kernel semantics, current injection, plasticity, and published
stimulation protocols remain required before this milestone is a functional
network reproduction.

Protocol checkpoint: the recovered horizontal/vertical PNGs contain the
paper's centered five-cell bars at green=120. Combined with the serialized 0.4
relay sensitivity and -60 mV leak, this reconstructs the reported -12 mV
driving potential. The first 100-ms connected run fails the 40-Hz target due to
a synchronized startup TRN transient and subsequent broad relay rebound; see
`validation-results/first-order-bar-001.yaml`. Subsequent source corrections
supersede that numerical candidate; `first-order-bar-003.yaml` is the active
source-consistent failure. Typed runtime profiles now carry stable SHA-256
fingerprints, and `validation.first_order` provides reusable isolated-relay
and connected-bar runners plus a predeclared 40-Hz scorer. The first coarse
matrix is recorded in `isolated-relay-sweep-001.yaml`; no candidate passed.
The follow-up `isolated-relay-axial-sweep-002.yaml` is retained as superseded:
visual inspection of the primary equation exposed an incorrect axial geometry
expression in the implementation. After correcting expanded-manuscript
Equation 9, the baseline isolated relay remains silent, and the corresponding
full-sector result in `first-order-bar-004.yaml` also produces zero relay
spikes. The correction therefore removes a fidelity defect but does not by
itself recover the Methods 4.9 40-Hz stream.
The next discriminator makes all-zero input mappings explicit. Omitting those
channels as inactive and using the unresolved 2 uF/cm2 capacitance candidate
recovers exactly 40 Hz in the isolated relay. In the connected sector it passes
either active rate (20-ms warmup) or selectivity (100-ms warmup), but not both;
see `isolated-relay-input-semantics-003.yaml` and `first-order-bar-005.yaml`.
Using zero rather than steady-state gate initialization removes the startup TRN
event. The 1.5 uF/cm2 candidate then passes both connected relay-rate and
selectivity gates (`first-order-bar-006.yaml`). This advances the work to the
published 100-ms learning episode; it does not yet establish cortical or
match/mismatch reproduction.

The spike detector now exposes its voltage coordinate explicitly. Applying the
paper's 30-to-0 mV event rule to soma voltage relative to its leak preserves the
isolated 40-Hz result but fails the connected drive gate (20 Hz per active relay
and 4,617 TRN spikes; `first-order-bar-007.yaml`). The physical-coordinate
setting therefore remains the current passing candidate while this source
ambiguity is audited; neither setting is silently treated as canonical.

Learning checkpoint: Equation 6's biphasic post-spike gate and the serialized
20/25-ms depotentiation intervals are tested. Adaptive records initialize at
their source-serialized `weight`; the distinct `assymptoticWeight` supplies the
decorrelated baseline. KInNeSS Equations 25/28
resolve and implement the presynaptically gated, postsynaptically gated, and
dual-AND-gated variants. Driven protocol validation and bounded long-run
behavior remain required before plasticity is considered reproduced.

Runtime checkpoint: the complete connected first-order sector integrates for
0.1 ms at 0.01-ms resolution with finite soma voltages and bounded finite
weights. With no input, adaptive weights remain bit-identical to baseline.
The actual relay-to-layer-4 Brian2 projection also potentiates and
depotentiates under forced positive and negative Equation 6 lobes while
remaining bounded. Longer protocol and convergence runs remain part of the
published learning-validation gate.

Exit: equation/unit tests, numerical convergence tests, isolated-cell report,
and explicit resolution of voltage-coordinate ambiguities.

## M3 — complete first-order 9×9 sector

- Construct deterministic 9×9 population sheets and explicit serialized edge
  lists from the supplement's one-to-one, Gaussian, local, broad, and all-to-one
  rules.
- Target the published soma/proximal/distal receptor ports.
- Match the published first-order population, neuron, compartment, and equation
  counts attributable to the LGN–V1 sector.

Exit: topology audit, connection-count snapshots, deterministic rebuild across
seeds, and one short stable sector simulation.

## M4 — learning, match, mismatch, and reset

- Implement Equation 6 learning with the published postsynaptic and dual-AND
  gates, bounded weights, and supplementary annotations.
- Implement transmitter depletion and recovery on all annotated projections.
- Encode the five-cell-bar stimulus, learned horizontal expectation, orthogonal
  mismatch, nonspecific arousal, and layer-4 reset protocols.

Exit: Figure 6 qualitative learning gates and maps; Figure 7 approximate 40/70
Hz nonspecific response; Figure 10 causal reset sequence and negative controls.

## M5 — first-order oscillation validation

- Reproduce 1,000 ms match and mismatch trials.
- Apply the stated mean subtraction, 200 ms Hamming analysis, and both published
  interpretations of the inconsistent middle frequency band.
- Validate match gamma dominance, mismatch slower/beta reset dynamics, and
  local synchrony near the reported 44 Hz simulation peak.

Exit: multi-seed report with predeclared tolerances, spectra, rates, synchrony,
ablations, and configuration fingerprints.

## M6 — higher-order pulvinar–V2 loop

- Add V1–pulvinar–V2 populations and projections with the same cell, port,
  topology, plasticity, and provenance abstractions.
- Validate long-range feedback, lower-frequency inter-area synchrony, and the
  reported 1 ms inter-area delay protocol.
- Add LFP/CSD geometry only after transmembrane-current accounting is verified.

Exit: full two-loop structural audit and higher-order validation report.

## M7 — freeze and experiment

- Tag the validated implementation and archive exact environment, configs,
  explicit connectivity, seeds, and generated reference metrics.
- Keep classic parameters immutable under the frozen profile.
- Introduce AdEx, GIF, alternative HH, and expanded multicompartment models only
  through the neuron-model registry, changing one experimental factor at a time.

Exit: a versioned classic baseline and a robustness experiment matrix that can
distinguish architecture-level effects from neuron-model-dependent effects.

## Original-source recovery track (recovered)

ModelDB alternate version 112923 preserves the complete source bundle and was
recovered from ModelDB's download endpoint. Its `SMART.nml`, dedicated Figure 8
calcium-rebound XML, dedicated AHP/ACh NeuroML, and stimuli are now the primary
executable references. The archive has no explicit redistribution license, so
only hashes and extracted facts are committed. M2–M6 must audit existing
paper-derived code against this source before claiming reproduction.
