# Classic SMART constrained calibration

Calibration starts from `classic-smart-source-constrained-v0.1.0` and never
modifies that tag. Its purpose is to infer a small family of plausible missing
legacy conventions—not to optimize arbitrary synaptic weights until plots look
similar.

## Scientific contract

1. Search dimensions must be unresolved, conflicting, or not identifiable in
   the surviving sources.
2. Each dimension must list provenance, admissible values or bounds, and why it
   is allowed to vary.
3. Published weights and conductances remain fixed unless the publication
   itself leaves them unidentified.
4. Candidates pass mechanistic gates in order; later output-rate agreement
   cannot compensate for an incorrect causal pathway.
5. Figures 10 and 14–16 are holdouts. Their results may reject a candidate but
   cannot be used to select or tune one.
6. Every evaluated candidate receives a deterministic fingerprint and retains
   failed as well as successful metrics.

## Stages

### A — isolated cellular viability

- relay tonic/burst direction from Figure 8;
- TRN quiescent before drive and recruitable by measured layer-6II/relay gates;
- layer-5 AHP/ACh direction and apical-to-soma propagation.

Candidates failing an isolated causal gate do not enter network calibration.

### B — first-order learning chain

- relay-to-layer-4 recruitment;
- layer-4-to-layer-2/3-to-layer-5-to-layer-6II causal sequence;
- Figure 6 bottom-up and top-down orientation;
- no autonomous population-wide startup regime.

### C — match/mismatch mechanism

- all five horizontal relay cells active during match;
- mismatch relay output confined to the horizontal/vertical overlap;
- more post-startup TRN output in match;
- nonspecific thalamus approximately 40 Hz match and 70 Hz mismatch.

### D — locked holdout evaluation

Without changing the candidate, evaluate:

- Figure 10 causal reset and disconnection control;
- Figure 14 gamma versus slower mismatch activity;
- Figure 15 local frequency target;
- Figure 16 lower-frequency long-range dominance.

## Selection rule

Selection is lexicographic, not a single unconstrained loss:

1. all isolated causal gates;
2. all Figure 6 causal/spatial gates;
3. Figure 7 spatial and TRN-order gates;
4. Figure 7 rate errors within predeclared tolerances;
5. complexity preference for fewer inferred deviations from the frozen profile.

If no candidate passes a stage, report that result and revise the uncertainty
model explicitly. Do not silently widen bounds or expose holdout metrics to the
optimizer.

The initial executable contract is
`configs/calibration/classic_uncertainty_space.yaml`.

The first Stage A matrix (`calibration-stage-a-trn-114.yaml`) evaluated 96
combinations of the initial contract and produced no quiescent-yet-recruitable
TRN candidate. Before any network calibration, the contract was therefore
revised to include the already documented official-source conflict between the
paper's printed Na/K rates and the archived ModelDB/Traub–Miles rates. This
revision follows the predeclared no-survivor rule above; it does not expose a
published conductance or consult any holdout result.

The revised 192-candidate matrix also produced no survivor. A second explicit
revision then exposed the paper's internal calcium-density conflict: Table 3
cell-specific densities versus Methods 4.6's global 250 mS/cm² statement. The
new global-density half was evaluated separately in
`calibration-stage-a-trn-116.yaml`, preserving the prior cell-specific result.
It likewise produced no survivor, so all 384 registered cellular combinations
are rejected at the TRN viability gate before network calibration.

The other isolated gates were then rerun independently under the same contract
fingerprint (`calibration-stage-a-remaining-117.yaml`). None of the 20
predeclared Figure 8 leak/capacitance candidates reproduced both tonic and
burst responses. The paper Figure 19 AHP/ACh profile passed all three kernel
gates, while the archived ModelDB profile passed frequency dependence and ACh
suppression but not 500-ms recovery. Of four axial conventions, only the
paper-literal Equation 2 profile propagated sustained distal layer-5 input into
somatic events. These partial passes cannot bypass the failed relay and TRN
gates.

The Figure 8 archive serializes calcium `g_bar=250` under an unrecovered
version-1 unit system. Its two documented endpoint interpretations—250
mS/cm² literal and 0.25 mS/cm² after a µS-to-mS conversion—separately recover
the burst and tonic halves but not both. Because Figure 8 is a registered
training target, a sparse logarithmic conductance grid between those endpoints
is predeclared in `run_figure8_legacy_calibration.py`. This grid may select a
version-1 unit mapping but may not consult network or holdout outputs.

That ten-point grid produced no survivor
(`figure8-legacy-unit-calibration-118.yaml`). The 0.25 mS/cm² endpoint restores
the tonic train but makes the hyperpolarized response sustained; the literal
250 mS/cm² endpoint preserves the transient burst but yields only one tonic
onset event. Interior points either sustain both conditions or collapse both to
onset events. A scalar calcium-unit conversion is therefore rejected as a
sufficient Figure 8 explanation.

Figure 8 scoring has now been corrected to match the published observable:
somatic membrane-voltage peaks rather than Equation-8 axonal release events
(`figure8-voltage-observable-audit-124.yaml`). Under literal calcium density,
the previously reported one/two release events conceal 137/109 voltage peaks
in the two conditions, so the earlier transient-burst pass was spurious. A
source-unit grid from 0.00025 through 0.25 mS/cm² finds a clean four-peak tonic
trace at 0.2 mS/cm², but the hyperpolarized condition has 15 prolonged peaks.
Pulse-unit and sub-nS finite-clamp grids also have no survivor. The active
Figure 8 discrepancy is therefore the legacy version-1 membrane/default
implementation, not merely its event detector, current unit, calcium unit, or
clamp strength.

The archived and printed Na-rate families also differ in two independent
sodium constants: activation scale and inactivation offset. The two possible
source hybrids were therefore exposed separately and screened across the full
remaining TRN space (`calibration-stage-a-trn-hybrid-nak-119.yaml`). All 384
control/driven pairs were finite, 314 controls were quiescent, and none was
recruitable. Archived activation preserves autonomous spiking; printed
activation preserves quiescence but not recruitment. Changing the inactivation
offset does not break that tradeoff, so neither hybrid is promoted.

The first TRN matrices were subsequently found to omit projection 010, the
relay→TRN AMPA port, despite describing the assay as relay/layer-6II drive.
They are retained but marked `superseded-incomplete-drive`. A fresh Figure 7
diagnostic measured peak gates of 20.60478 relay AMPA, 1.73256 layer-6II AMPA,
and 0.11284 layer-6II NMDA. The corrected complete 768-candidate matrix
(`calibration-stage-a-trn-complete-drive-120.yaml`) contains six survivors.
The lexicographically first is frozen in `trn_stage_a_survivor_v1.yaml`; its
control peaks at +24.27 mV with zero events, while complete drive reaches
+31.21 mV and emits four events. This repairs the TRN isolated gate without
fitting a receptor strength. The same candidate still fails isolated layer-5
apical-to-soma propagation, and the dedicated Figure 8 gate remains unresolved,
so it cannot yet advance as a complete Stage A survivor.

The selected TRN survivor has now been run through the complete official
Figure 6 training episode (`calibration-network-trn-survivor-121.yaml`). It is
finite, but layer 2/3 and layer 5 emit no events, the cortical feedforward chain
is incomplete, and neither bottom-up nor top-down learned map passes its
predeclared orientation gate. A measured-gate isolated discriminator
(`figure6-layer23-transfer-discriminator-122.yaml`) shows that excitation alone
can reach the soma under every axial convention. With the observed 2.30-ms
inhibitory-current delay, only paper-literal Equation 2 emits an event; the
KInNeSS profiles remain suppressed. Since the full Equation-2 network also
failed Figure 6 and Equation 2 is not a TRN survivor, no global profile is
promoted. The next training-only discriminator is the source-defined transient
excitation/inhibition waveform at layer 2/3.

The registered Gaussian-spread ambiguity was then tested using the KInNeSS
paper's wording that Spread X/Y set Gaussian variances. The resulting candidate
(`calibration-network-gaussian-variance-123.yaml`) recruits layer 4 earlier but
doubles layer-4 and interneuron events, increases TRN/category over-recruitment,
and still produces no layer-2/3 or layer-5 events. All Figure 6 learning gates
remain failed, so the variance interpretation is retained as a negative
source-discriminator rather than promoted.

An intact-versus-projection-free Figure 6 relay assay now identifies an earlier
network bottleneck (`figure6-relay-current-balance-125.yaml`). The source drive
is correctly reconstructed and the relay can fire repetitively in isolation,
but the intact TRN feedback leaves it with one event. Removing only the three
serialized TRN-to-relay GABA_A projections restores 19 events. This is not a
license to delete TRN inhibition: the next calibration discriminator must test
source-supported TRN recruitment, transmitter, spatial-weight, and GABA gate
semantics while retaining the official projections. The control also leaves
layer 4 with one event, so thalamocortical transfer remains a subsequent gate.

All six independently admissible TRN survivors have also been screened in the
intact network for the first 20 ms of the same Figure 6 episode
(`figure6-relay-survivor-screen-126.yaml`). All six emit exactly one relay event
at 1.89 ms. The three registered axial conventions and both Equation-8 event
rules are therefore exhausted at this gate; subsequent work should not switch
among them to manufacture relay repetition.

The isolated-TRN screen's registered 5-ms pre-drive has also been tested as a
connected-network equilibration (`figure6-relay-equilibration-127.yaml`). It
reduces center-relay output from one event to zero and is rejected. Do not add
an undocumented warmup to hide the intact thalamic suppression discrepancy.

The archived relay cell has now been separable from the global intrinsic-cell
choice as the explicit source profile
`modeldb_relay_paper_table3_others`. Its complete Figure 6 run
(`figure6-relay-source-hybrid-128.yaml`) removes the inactive full-sheet relay
volley, restricts relay and layer-4 output to the five bar pixels, and restores
the qualitative Figure 6b bottom-up orientation. It does not recruit layer 2/3
or layer 5 and fails Figure 6c, so it remains unpromoted. This moves the next
training discriminator back to the measured layer-4 excitation/feedforward
inhibition waveform under the corrected five-cell relay input.

The paper-literal Equation 2 axial profile has now been crossed with the
archived-relay source profile (`figure6-relay-axial-source-hybrid-129.yaml`).
This interaction restores the complete ordered cortical chain, a causal
teaching-before-relay pair, and strong Figure 6b orientation. Its only failed
gate is combined Figure 6c top-down contrast: 0.007370 versus the registered
0.01 minimum. Keep the acceptance threshold fixed. The next discriminator is
the wide/narrow corticothalamic learning decomposition and category-cell
spatial activity under this exact fingerprint.

That decomposition is now recorded in
`figure6-relay-axial-map-decomposition-130.yaml`: inactive vertical arms remain
unchanged, the narrow field contributes most positive contrast, and mixed
potentiation/depression in the wide horizontal arm causes the remaining gap.
Artifact `figure6-leading-source-alternatives-131.yaml` exhausts registered
event-rule, initialization, calcium-kinetics, Na/K-rate, archived-category, and
serialized-weight alternatives. Keep the leading fingerprint fixed while
auditing the exact presynaptically gated learning phase and its source timing.

That audit is complete in `figure6-top-down-learning-phase-132.yaml`. The
three integrated Equation 25/28 components reconstruct every measured weight
change with maximum error `1.52e-16`. Projection 005's nearest horizontal
targets are depressed because their `-0.068561` anti-causal term exceeds the
`+0.057777` causal term; its farther targets see only `-0.028240` depression
and potentiate. Projection 007's nearest targets potentiate normally, and
inactive vertical targets do not change. The next discriminator must therefore
address a source-supported cause of the wide-field postsynaptic phase or its
Equation 6 depression scale. Do not tune the fixed `0.01` map gate, delete TRN
feedback, or alter the already verified presynaptic category waveform.

The projection-level Equation 25 bounds interpretation has now been tested
without changing the weak Gaussian pre-map
(`figure6-projection-level-learning-bounds-133.yaml`). It fails with combined
top-down contrast `-0.01374`: inactive vertical weights grow toward the uniform
`w0=0.05`. Therefore the local spatial decorrelation baseline remains required
by Figure 6c, and projection-level `D` cannot be obtained merely by switching
the complete learning bounds. The next source discriminator must isolate the
Equation 6 postsynaptic scale from Equation 25's local decorrelation baseline,
or correct the relay spike phase through another documented mechanism.

The isolated projection-level `D` hybrid is now rejected as well
(`figure6-projection-depression-scale-134.yaml`): top-down contrast is
`-0.01531`, and narrow active weights become negative. Do not tune `D`
independently of Equation 25 bounds. The remaining mechanistic route is the
documented category-arrival/relay-spike phase under the local bounded learning
profile; first attribute depression to individual teaching volleys, then test
only source-supported event, delay, or intrinsic mechanisms that move that
phase.

That route has produced the first complete Figure 6 survivor
(`figure6-population-resolved-axial-136.yaml`). Pairing the archived relay with
archived axial edges and the paper Table 3 cells with paper Equation 2 restores
the declared 40-Hz relay rate, the cortical chain, later causal gamma-phase
pairs, Figure 6b contrast `0.23252`, and Figure 6c contrast `0.03045`. The
full-episode causal assessor now selects the closest teaching/relay pair across
all category spikes instead of incorrectly testing only the onset category
spike. Promote `figure6_population_resolved_axial_v1.yaml` to the next stage.
Freeze all Figure 6 conventions and proceed to Figure 7 match/mismatch rate and
TRN-order gates; do not retune Figure 6 while evaluating those holdouts.
