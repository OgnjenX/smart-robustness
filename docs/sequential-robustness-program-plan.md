# Sequential SMART robustness program

## Purpose

Complete the neuron-model robustness question before testing post-2008 circuit
anatomy, while keeping `classic-smart-calibrated-v1.0.0` immutable. The program
must distinguish dependence on the somatic spike generator, active dendritic
and compartmental mechanisms, and network anatomy. It must not use a later
phase to rescue or redefine an earlier negative result.

The historical control remains a calibrated behavioral reconstruction, not a
bit-identical recovery of the unavailable 2008 execution. Its documented
Figure-8 source boundary and Figure-15 44-Hz numeric failure remain part of
every comparison.

## Current starting point

- The frozen classic control passes the registered Figure 6 learning gates and
  the calibrated Figure 7, 10, 14, and 16 behavioral targets.
- Morphology-preserving cortical AdEx and GIF substitutions are complete. No
  registered arm preserved the conjunction of all Figure 6 gates.
- Those results reject the tested independently fitted parameter maps; they do
  not show that ART requires Hodgkin--Huxley neurons.
- Relay/TRN replacement is blocked under the registered rebound-matching
  contract because the classic relay itself does not express the required
  source-port rebound target.
- Alternative conductance-based HH and compartment/mechanism interventions
  remain untested.

## Rules shared by every phase

1. Register hypotheses, intervention boundaries, parameter sources, seed sets,
   metrics, gates, and stopping rules before observing network outcomes.
2. Fit neuron parameters only to isolated-cell or source-port targets. Network
   behavior is a sealed holdout and is never a fitting objective.
3. Preserve the frozen control's topology, stimuli, learning rules, projection
   scales, seeds, durations, and analyses unless anatomy is the explicitly
   registered independent variable.
4. Run the validation ladder in order. A failed prerequisite is reported and
   stops downstream claims; it is not repaired by outcome-guided tuning.
5. Distinguish published, derived, calibrated, and exploratory values in
   `docs/parameter-provenance.yaml`.
6. Keep species, sensory modality, and cortical area explicit. Mouse barrel,
   mouse visual, and primate visual measurements are separate evidence strata.
7. Preserve raw outputs locally, hash-pin them in committed assessments, and
   record environment, commit, configuration fingerprint, and random seeds.

## Phase 0 — seal the completed screening

### Work

- Merge or otherwise archive `codex/neuron-model-localization` without changing
  the frozen classic release.
- Publish a concise closeout matrix for classic, AdEx, and GIF arms, including
  individual-gate survival and prohibited conclusions.
- Tag the completed screening state, provisionally
  `neuron-model-screening-v1.0.0`.

### Exit gate

The assessment, hashes, focused tests, full CI result, and interpretation
boundary are reproducible from a clean checkout.

## Phase 1 — alternative conductance-based HH control

### Scientific question

Does Figure 6 fail after changing somatic spike kinetics while retaining an
explicit conductance-based Na/K generator and all SMART morphology, ports,
dendritic mechanisms, and network parameters?

### Implementation

- Add a new somatic conductance-based HH backend rather than aliasing the
  classic SMART equations. Select one minimal cortical HH family after a source
  audit; freeze the equations before fitting.
- Fit a per-cell-class parameter map to preregistered isolated targets: resting
  potential, input resistance/time constant, rheobase, steady-state firing
  rate, adaptation where applicable, spike threshold/amplitude/width, and
  source-shaped conductance-input transfer.
- Retain the existing morphology-preserving adapter contract so the only first
  intervention is the somatic Na/K mechanism.
- Run classic control plus cortical-all, excitatory-only, and inhibitory-only
  substitutions. Deterministic arms receive exact rerun checks; any stochastic
  parameter ensemble must have a frozen seed set.

### Progression gates

1. Isolated fits pass without network consultation.
2. Figure 6 passes all six frozen gates.
3. Only a Figure-6-passing arm may proceed to Figure 7 match/mismatch and
   Figure 10 causal reset.
4. Only a behavioral-pass arm may proceed to Figure 14/15 spectral behavior
   and Figure 16 higher-order behavior.

### Interpretation

- Pass: the exact classic SMART somatic kinetics are not necessary.
- Fail: conductance-based spiking alone is insufficient under the registered
  mapping; this still does not prove family-level impossibility.

## Phase 2 — compartment and active-mechanism localization

### Scientific question

Which retained biophysical mechanisms explain any difference between the
classic, alternative-HH, AdEx, and GIF outcomes?

### Sequential interventions

Run these one at a time, each with its own registration:

1. Disable the active distal Na/K mechanism in layer-5 cells while preserving
   all compartments and receptor placement.
2. Passivize cortical dendrites while retaining somatic classic HH spiking,
   morphology, axial coupling, and compartment-targeted synapses.
3. Collapse cortical cells to a point-HH representation with total capacitance,
   leak, and receptor conductances conserved by a preregistered aggregation
   rule.
4. If justified by the preceding results, introduce an expanded layer-5 model
   separating basal, proximal apical, and distal tuft integration. Do not fit
   this expansion to network outcomes.

### Exit gate

A mechanism-dependence report identifies which simplifications preserve each
SMART gate and which conclusions remain unidentifiable. Freeze the completed
neuron-model study as `neuron-model-robustness-v1.0.0`.

## Phase 3 — register the contemporary-anatomy matrix

Create a new intervention family that always runs against the frozen classic
control. For each module, preregister two levels:

- **Parameter-invariant:** add only the evidence-backed pathway or cell-class
  distinction; do not retune classic parameters.
- **Empirically bounded recalibration:** permitted only after the invariant
  outcome is sealed, using a finite source-backed range fixed in advance.

Classify each outcome as:

- survives unchanged;
- survives only after bounded compensation;
- strengthens the target phenotype;
- breaks the target phenotype; or
- remains inconclusive because the empirical mapping is underdetermined.

Initial evidence anchors for the registrations include Constantinople and
Bruno (2013, DOI `10.1126/science.1236425`) for parallel thalamic recruitment
of deep sensory cortex; the adult mouse V1 layer-5 dendritic study associated
with DOI `10.1016/j.celrep.2024.114638` for visual-specific monosynaptic
thalamic input; Fişek et al. (2023,
DOI `10.1038/s41586-023-06007-6`) for active apical integration of visual
feedback; Blot et al. (2021, *Neuron*) for distinct visual intracortical and
transthalamic signals; Saalmann et al. (2012,
DOI `10.1126/science.1223082`) for pulvinar-mediated cortical coordination;
and Tasic et al. (2018, DOI `10.1038/s41586-018-0654-5`) for cortical cell-class
diversity. Each phase must conduct a focused source audit before converting
these qualitative anchors into parameters.

## Phase 4 — direct visual-thalamic input to layer 5

### Rationale

Modern visual evidence supports thalamic input outside the classic layer-4
entry motif, including monosynaptic input to layer-5 pyramidal dendrites. This
is the smallest and most directly testable contemporary extension.

### Design

- Use visual-system evidence for the primary arm; keep somatosensory evidence
  as convergent motivation rather than interchangeable parameters.
- Add first-order visual-thalamic input to a documented layer-5 compartment,
  initially without splitting layer-5 cell identity.
- Freeze target compartment, spatial footprint, delay, receptor composition,
  and a finite conductance range before execution.
- Include pathway-off control, unchanged classic control, and input-matched
  controls that separate extra mean drive from laminar routing.
- After sealing the first result, register an L5 subtype split only if the
  source literature supports a mapping to the modeled visual species/area.

### Outcomes

Run the full progression ladder: Figure 6 learning; Figure 7 match/mismatch;
Figure 10 reset and negative control; Figure 14/15 spectra and synchrony; and
Figure 16 inter-area behavior. Also measure whether the added path shortens
latency, changes resonance stability, or alters vigilance/reset threshold.

## Phase 5 — active apical integration of cortical feedback

- Represent higher-area feedback to V1 layer-1/apical-tuft targets separately
  from somatic or proximal excitation.
- Compare a conductance-only pathway with an independently specified active
  dendritic integration arm.
- Measure retinotopic match versus offset/surround conditions, somatic output,
  dendritic events, gamma synchronization, and false resonance.
- Keep this phase separate from direct thalamus-to-L5 so their interaction is
  not mistaken for either main effect.

## Phase 6 — updated pulvinar/transthalamic routing

- Audit the existing V1--pulvinar--V2 loop against contemporary visual
  evidence.
- Separate layer-5 driver and layer-6 modulatory corticothalamic pathways.
- Represent direct corticocortical and transthalamic routes as parallel paths
  with independently sourced delays, termination layers, and short-term
  dynamics.
- Test pathway ablations and matched-input controls before combining this
  module with Phases 4 or 5.
- Evaluate Figure 16 behavior plus information routing, phase coupling, and
  whether pulvinar effects are sensory, state-dependent, or both.

## Phase 7 — inhibitory cell-class specificity

- Begin with functional PV-, SST-, and VIP-like classes rather than attempting
  the full transcriptomic taxonomy.
- Preserve the original aggregate inhibitory strength in the first arm so the
  intervention tests routing and timing rather than simply adding inhibition.
- Register cell-class-specific targets and connections from one species/area
  evidence set.
- Test PV timing/gamma control, SST dendritic gating, and VIP disinhibition as
  separate modules before their combination.

This phase is last because it introduces the largest number of new populations
and interaction parameters.

## Phase 8 — interactions and final synthesis

- Combine only modules whose single-factor outcomes are already sealed.
- Use a small preregistered factorial design to test interactions among direct
  thalamic L5 input, apical feedback, updated pulvinar routing, and inhibitory
  subclasses.
- Cross contemporary anatomy with alternative neuron models only after the
  classic-neuron modern-anatomy control passes its own prerequisites.
- Report main effects, interactions, uncertainty, failed gates, and parameter
  regions. Do not select only successful seeds or parameter combinations.

## Implementation map

Likely new or extended components:

- `src/smart_robustness/models/compartmental_hh.py`: support the registered
  alternative-HH and mechanism flags without changing classic defaults.
- `src/smart_robustness/models/registry.py`: promote backends only when their
  contracts and tests are complete.
- `src/smart_robustness/models/selective.py`: reuse population-localization
  dispatch for HH and mechanism arms.
- `src/smart_robustness/classic_sector.py`: leave classic assembly behavior
  unchanged; expose extension hooks rather than editing the frozen path.
- `src/smart_robustness/modern_anatomy.py`: add opt-in projection/population
  modules and matched-drive controls.
- `src/smart_robustness/validation/`: add a common staged progression runner
  over the existing Figure 6, 7, 10, 14, 15, and 16 validators.
- `configs/models/`: isolated-fit protocols and frozen parameter maps.
- `configs/robustness/`: one immutable registration manifest per intervention.
- `scripts/`: sharded runners, result merger, and assessment generation.
- `docs/validation-results/` and `docs/parameter-provenance.yaml`: registrations,
  hash-pinned assessments, sources, and interpretation boundaries.
- `tests/`: adapter contracts, intervention isolation, source/seed invariants,
  progression stopping, deterministic reruns, stochastic ensembles, and full
  control-regression coverage.

## Verification strategy

For every phase:

1. Unit-test equations, units, ports, compartment targeting, and selective
   dispatch.
2. Verify the intervention changes only registered fields by comparing
   serialized network fingerprints against the control.
3. Run isolated-cell/source-port tests before opening network outcomes.
4. Re-run the frozen classic sentinel beside every new arm.
5. Execute the staged behavioral and spectral validators without skipping a
   failed prerequisite.
6. Run deterministic reproducibility checks or the complete registered seed
   ensemble.
7. Run lint, focused tests, the full repository test suite, and GitHub CI.
8. Commit an assessment containing raw-result hashes, software versions,
   configuration fingerprints, exact gates, and prohibited conclusions.

## Sequential stopping policy

A negative phase is a valid result. Stop an arm when its registered prerequisite
fails, record the surviving sub-gates, and continue to the next scientifically
distinct intervention only after the assessment is frozen. Missing source data
or an invalid classic target is reported as blocked; it is never replaced by
network-guided parameter fitting.
