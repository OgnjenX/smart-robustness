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

## Original-source recovery track

ModelDB accession 112922 preserves metadata and an author-era link to
`Brain_Research_Paper_KINNESS_SMART_network.rar`, described as the model network
and input stimuli. The public ModelDB GitHub mirror does not contain the payload,
and no exact Internet Archive snapshot was found during the initial recovery
attempt. Recovery remains valuable because it may resolve initial states,
connectivity realizations, stimuli, and analysis scripts. Any recovered archive
must be inspected for a model-specific license before redistribution.
