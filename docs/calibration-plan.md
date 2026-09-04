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
- over the plotted 100-ms Figure 7c window, nonspecific thalamus fires four
  times (40 Hz) during match and seven times (70 Hz) during mismatch.

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

That route produced the first Figure 6 rate/timing/shape survivor
(`figure6-population-resolved-axial-136.yaml`). Pairing the archived relay with
archived axial edges and the paper Table 3 cells with paper Equation 2 restores
the declared 40-Hz relay rate, the cortical chain, later causal gamma-phase
pairs, Figure 6b contrast `0.23252`, and Figure 6c contrast `0.03045`. The
full-episode causal assessor now selects the closest teaching/relay pair across
all category spikes instead of incorrectly testing only the onset category
spike. Artifact 140 subsequently retracts that promotion: the combined learned
map peaks at `0.11853`, below the official approximately `0.5--2.5`
after-learning range. Keep `figure6_population_resolved_axial_v1.yaml` as the
leading mechanistic candidate, but restore Figure 6c learned-map magnitude as
the active calibration gate before freezing Figure 6 conventions.

The first learned-state Figure 7 holdout is recorded in artifact 137 and fails:
match and mismatch are both 10 Hz, all 81 TRN cells share one startup event,
and neither relay-subset nor TRN-order gates pass. This is now consulted
calibration evidence, not an untouched holdout. Artifacts 138 and 139 reject
the two immediate source-coherent TRN packages because they fail the frozen
Figure 6 prerequisite through 319- or 810-event TRN over-recruitment. Treat the
holdout as diagnostic evidence that the weak learned state is insufficient;
do not calibrate Figure 7 ahead of the restored Figure 6 amplitude gate. After
that gate passes, the next Figure 7 discriminator must localize why the
full spatial TRN sheet recruits globally despite the isolated survivor, using
the recorded per-pathway currents and negative controls before exposing any
new parameter.

Artifact 141 closes the leading profile's exact learning terms to `1.22e-16`.
At representative horizontal target 39, wide and narrow weights grow by only
`0.01366` and `0.02640`, with less than `0.5 ms` of positive postsynaptic
overlap across four relay spikes. The next Figure 6 discriminator is therefore
the source interpretation of Equation 6's postsynaptic threshold/duration,
kept separate from the +30-mV/falling-zero output event detector in Equation 8.

Artifacts 142--145 exhaust the source-bounded Equation 6 threshold/coordinate
alternatives. Absolute `0` and `-20 mV` thresholds peak at only `0.15232` and
`0.16854`; leak-relative `0 mV` reaches `0.52741` but recruits 58 relay events.
The corrected gate requires a `2.0` combined peak (tolerance around the
published approximately `2.5` maximum) and confines relay activity to four
events in each of the five horizontal cells. No threshold profile passes.

Artifacts 146 and 147 resolve the next official-source contradiction. Methods
4.3 dual-AND gating reaches only `0.19319` under absolute voltage. Its
interaction with leak-relative voltage reaches `0.96876` but recruits 58 events
across 43 relay cells. Neither passes, so learning-rule choice is not a
sufficient Figure 6 explanation.

Artifact 149 closes the strongest rejected candidate's learning terms to
`5.55e-16`. It shows that leak-relative dual-AND gating potentiates vertical
nonspiking targets (`~0.23--0.25`) as well as horizontal targets (`~0.48`),
because subthreshold surround depolarization remains above the leak-relative
threshold. This route is mechanistically closed. Next audit action-potential
waveform duration against the unresolved legacy membrane/Na-K implementation,
without changing the established output-event or spatial gates.

Artifact 150 completes that waveform screen. The only two spiking Na/K
families remain above +30 mV for `0.17--0.18 ms`; the two printed-activation
families never spike. No registered Na/K family materially extends the
positive learning phase, so this discriminator is closed without promotion.

Artifact 151 then separates Equation 6's spike timestamp from Equation 8
release. The upward +30-mV timestamp preserves all spike counts but lowers the
combined map peak to `0.10750`; its earlier depression tail is less favorable
than the existing falling-phase timestamp. This timing route is closed.

Artifact 152 tests the last source-coordinate combination: leak-relative
`+30 mV`. It preserves confined 40-Hz bar recruitment but peaks at only
`0.18606`. All registered Equation 6 threshold/coordinate alternatives are now
closed; do not lower the threshold into the diffuse subthreshold regime.

Artifacts 166--168 continue the post-holdout TRN source decomposition. The
complete 2x2 Table 3/SMART.nml potassium density/reversal matrix produces no
post-bottom-up TRN event. The corresponding soma-channel/calcium-reversal
matrix finds one excitable candidate at the archived 120-mV reversal, but it
recruits all 81 TRN cells before bottom-up onset under the diagnostic cue lead.
The primary paper specifies simultaneous bottom-up/top-down excitation, so the
candidate received one paired canonical run. Match and mismatch are identical
(zero relay cells, 229 TRN events, 30-Hz nonspecific output) and the candidate
is rejected. Do not fit an intermediate reversal. Next factor only the exact
10-versus-100 mS/cm2 dendritic TRN calcium-density conflict before concluding
that the surviving public sources cannot identify the legacy propagation
convention.

Artifact 169 completes that 10-versus-100 mS/cm2 crossing. All four archived-
density channel/reversal combinations produce zero TRN events even though
proximal dendrites exceed +85 mV. The exact public-source cube is exhausted.
The no-survivor rule now permits a separately named behavior-calibration stage:
predeclare a finite density grid bounded by 10 and 100 mS/cm2, first reject
values that allow top-down-only population-wide TRN output, then test the
surviving values under simultaneous match/mismatch onset. This stage may train
on Figure 7 because that holdout has already been consulted, but it must leave
Figures 10 and 14--16 locked and may not be described as recovery of the
unreported original density.

The behavior grid does not produce a survivor. Artifact 170 rejects only the
10-mS/cm2 endpoint at the top-down-only gate; 15--100 mS/cm2 remain cue-safe.
Artifact 171 then shows that all seven cue-safe values preserve exactly the
five matched horizontal relay cells but produce zero TRN events during the
50-ms simultaneous match condition. Mismatch evaluation is correctly skipped.
Scalar dendritic calcium-density calibration is therefore closed. Return to
source-level compartment topology and legacy axial/event propagation semantics
before exposing any additional continuous cellular parameter.

Artifact 172 closes the topology branch without a simulation sweep.
`SMART.nml` declares the Reticular cell as a `linear` cable in the serialized
order Soma, Dendrite 0, Dendrite 1; only the soma has `monitorSpikes=true`.
The implementation already compiles those compartments into the two adjacent
edges soma--proximal and proximal--distal, and the archived manual independently
places chemical output at the somatic/axonal detector. A star topology and
dendritic event output are therefore source-incompatible and must not be fitted.
The next permissible behavior-calibration discriminator is a predeclared,
localized soma--proximal propagation grid with the distal edge, intrinsic
channels, event threshold, synaptic strengths, and Figures 10 and 14--16 held
fixed. It must be reported as calibration of an unidentified legacy runtime
effect, not as an original SMART parameter.

Artifacts 173 and 174 complete that localized propagation grid. All ten
soma--proximal edge scales from 1x through 16x pass the top-down-only safety
gate. Under simultaneous match onset, scales 1--3x retain exactly relay cells
38--42 but emit no TRN event; scales 4--16x activate all 81 relay cells and
still emit no TRN event. The largest sampled somatic peak over the complete
grid is approximately -23.77 mV, while proximal dendrites reach regenerative
positive voltages. No scale reaches Stage 2a, so mismatch is not run. A
localized axial-gain correction is therefore closed as a sufficient behavior
calibration. Do not search edge scales more finely or alter the fixed linear
topology.

Artifacts 175 and 176 complete the source-bounded detector-origin grid. The
numeric offset preserves the printed +30/0-mV two-stage detector while moving
both landmarks together; 0, 67, and 69 mV are the absolute, fixed-shift, and
TRN leak-relative source anchors. A corrected Stage 1 records equilibration
output explicitly and requires its final 10 ms to be quiescent. Offsets 0--40
mV retain the source-profile one-event-per-TRN startup volley and are cue-safe;
50 mV generates 405 equilibration TRN events, including 162 in the tail and a
further 162 in the cue lead, and is rejected. Offsets 60, 67, and 69 mV remove
the startup volley and remain tail/cue quiet. Under simultaneous match, 0--40
mV preserve exactly relay cells 38--42 but emit no TRN event, while 60/67/69 mV
activate all 81 relay cells and still emit no TRN event. No offset advances to
mismatch. Detector voltage-origin calibration is closed; do not interpolate
between 40 and 50 mV or tune the arm/release thresholds independently.

Artifact 177 tests the only predeclared second-order rescue suggested by those
single-factor screens: the complete registered 10--100 mS/cm2 dendritic
calcium-density grid at the registered 50-mV detector offset. Every density
fails before bottom-up onset. Equilibration TRN output rises from 162 to 405
events across the grid, and every candidate emits at least 81 additional TRN
events during the top-down-only cue lead. No density reaches the simultaneous
match gate. Calcium density therefore cannot regularize the event-coordinate
transition into a causal match detector. Do not fit intermediate density or
offset values. The next calibration must represent a qualitatively different
legacy compartment-to-event transfer hypothesis, remain separately named from
the source reconstruction, and continue to lock Figures 10 and 14--16.

Artifacts 178--180 test that transfer hypothesis as a soma--proximal detector
blend while preserving somatic chemical output, cable topology, intrinsic
channels, and the printed +30/0-mV detector. All registered 0--1 blend values
pass the original cue screen. In the 50-ms match assay, 0.5 is the sole
survivor: relay cells 38--42 emit five events and all 81 TRN cells emit once.
A fresh 50-ms mismatch pair then produces the orthogonal relay set 22, 31, 40,
49, 58, the same 81 TRN events, and zero nonspecific output in both conditions.
Thus the blend recovers an early match pathway but not condition-dependent
arousal.

Artifact 181 corrects two validation assumptions by re-reading the primary
paper. Top-down-only specific-thalamic cells are inhibited through TRN in the
one-against-one regime, so cue-evoked TRN events are permitted if relay and
nonspecific output remain silent. In addition, the complete-module mismatch
trace uses 300-ms epochs and first increases nonspecific output around 50 ms;
the 50-ms pair is therefore only an early pathway screen. Reopen only the
already registered 600/800/1000-pA top-down-current dimension at blend 0.5,
then use a 300-ms pair for the final Figure 7 arousal gate. Do not alter weights,
delays, geometry, or holdout figures.

Artifacts 182--184 execute that corrected protocol. All registered currents
are settled and cue-safe; increasing current advances the isolated category
event from 8.89 to 5.83 to 4.47 ms but produces no cue-period TRN output. All
three currents reproduce the same early 50-ms match signature (five horizontal
relay events and 81 TRN events). In the 300-ms pair, relay counts are match vs.
mismatch 389/432, 441/419, and 418/426 at 600, 800, and 1000 pA. Only 800 pA
has the correct relay-count direction. Every current yields 81/81 TRN events
and zero/zero nonspecific events, so none passes. The undocumented current
amplitude is closed at its registered bounds; do not extrapolate above 1000 pA.
Existing diagnostics instead localize the next source-runtime discrepancy to
nonspecific thalamus: proximal voltage exceeds +129 mV while soma peaks near
+13 mV, below the +30-mV event arm threshold. Audit/calibrate that population's
compartment-to-event transfer separately before changing network weights.

Artifacts 185--187 complete the nonspecific-thalamus compartment-to-event
screen. All seven registered soma--proximal detector blends are settled and
cue-safe. During the 100-ms mismatch assay, blends 0--0.2 remain silent;
0.3/0.5 emit one nonspecific event and 0.7/1.0 emit two, so those four advance
to the independently computed match comparison. Every survivor is exactly
condition-invariant: match/mismatch counts are 20/20 relay events, 81/81 TRN
events, and respectively 1/1, 1/1, 2/2, or 2/2 nonspecific events. The transfer
is sufficient to expose dendritic activity at the output detector but cannot
create mismatch selectivity from identical upstream event trains. No candidate
advances to 300 ms. Keep this population-specific transfer available as a
named calibration mechanism, but do not promote a value or tune it further.
The next discriminator must act before the shared first relay/TRN volley.

Artifacts 188 and 189 test whether simultaneous source onset was placing the
learned expectation too late. At 800 pA, the sensory relay volley begins at
4.10 ms, the selected category cell emits at 5.83 ms, and archived feedback
delays place the first relay-NMDA, TRN-NMDA, and TRN-AMPA arrivals at 7.83,
8.83, and 9.83 ms. Those three arrival times and the zero-lead control were
registered before simulation. All nonzero leads are cue-safe and advance the
relay/TRN volleys, with match modestly earlier than mismatch, but every pair
still contains five orientation-specific relay events and 81/81 TRN events.
Source-receptor arrival alignment is therefore insufficient.

Artifacts 190 and 191 test the only registered local interaction suggested by
that latency difference. At the 9.83-ms TRN-AMPA alignment, detector blends
0.40--0.46 give no TRN output and allow all 81 relay cells to become active;
0.48--0.50 preserve the five-cell horizontal match and generate 81 TRN events.
Fresh mismatches at all three match-surviving values retain the five vertical
relay cells and the same 81 TRN events. No value advances to 100 ms. This
second-order source-timing/event-transfer interaction is closed; do not
interpolate the 0.46--0.48 transition further.

Artifact 192 closes the functional inhibitory-arrival extension. The 9.83-ms
trial's first TRN event occurs 3.82 ms after bottom-up onset, but a 13.65-ms
lead proves that this was relay-driven rather than top-down-driven. Leads 13.65
and 13.75 ms, respectively aligning that apparent event and its archived
0.1-ms GABA delay, remain free of cue-period TRN and relay output. After
bottom-up onset they recruit the five horizontal cells plus vertical neighbors
31 and 49, while all 81 TRN cells again emit once. Neither reaches mismatch.
Do not extend the cue lead: top-down corticoreticular drive is spike-ineffective
in this reconstruction, so post hoc timing cannot make its inhibition causal.

Artifact 193 corrects the Figure 6 source-strength prerequisite. The former
`2.0` absolute learned-map peak gate was inferred from a raster colorbar whose
underlying numeric matrix was never published; moreover, the printed learning
law bounds the sum of the two adaptive components at `1.55`. It is therefore
not an identifiable official target. Using archived spike identities from the
same candidate/runtime, the active candidate reproduces the source-supported
Figure 6 claims: exactly four relay events in each of cells 38--42, complete
cortical recruitment, a causal learning pair, and horizontally oriented
bottom-up and top-down maps. Figure 6 is promoted as a qualitative,
source-strength reproduction with a recorded combined adaptive peak of
approximately `0.893`; exact absolute map amplitude remains explicitly
unverified. Figure 7 calibration may proceed from this learned state, but no
classic baseline may be frozen until the published match/mismatch dynamics and
the still-locked later figures pass.

Artifacts 194 and 195 invalidate the apparent 0.5-blend early-match survivor.
Exact detector instrumentation shows that the fresh-network 81-cell TRN volley
contains zero in-trial +30-mV upcrossings. Each sampled detector instead
releases one pre-stimulus latched arm and never re-arms; its post-event maximum
is below +8 mV. Carrying the current Figure 6 episode into Figure 7 in the same
network consumes that cold-start state, after which matched recognition gives
zero TRN crossings/events, zero nonspecific output, and 181 relay events across
all 81 cells. The blend is rejected as a behavioral match mechanism. Do not
count initialization-latch release as resonance, and do not tune the detector
threshold from this result. The next discriminator must audit whether Figure
6's nominal four-event relay train itself contains a startup-latch release,
then return to the missing stimulus-evoked corticoreticular/TRN regenerative
pathway.

Artifact 196 completes that Figure 6 audit. In a separate 20-ms no-input
control, all five sampled relay detectors stay between approximately -70 and
-59.2 mV, remain unarmed, and emit nothing. During the canonical training
episode, each of relay cells 38--42 has exactly four +30-mV upcrossings, four
arm transitions, four release transitions, and four emitted events, ending
unarmed. The 40-Hz relay criterion is therefore detector-cycle-valid and the
qualitative Figure 6 promotion remains in force. The active failure is again
specific to evoked TRN recruitment in recognition; do not reopen Figure 6 rate
calibration on account of the TRN startup latch.

Artifacts 296--301 add one source-derived timing diagnostic after the later
pre-event current audit localized the direct-feedback deficit to waveform
timing. The archived adaptive AMPA channels 005/007 have a 2-ms delay and
2/7-ms normalized dual-exponential kinetics, fixing their analytic peak
3.507736 ms after arrival. Adding that peak time to the measured 5.85-ms
category latency registers a sole 11.357736-ms cue lead; no timing grid or
parameter change is allowed. The match passes twice in independent fresh
networks with 15 relay, 635 genuine-cycle TRN, and four nonspecific events.
The fixed mismatch remains five-cell/15-event recruitment, produces 653 TRN
events, and stays at four nonspecific events. Close receptor-peak timing as a
match-only repair. Do not interpolate around 11.357736 ms or tune the
successful match endpoint; the next candidate must explain pre-regenerative
suppression of bottom-up-only mismatch relay cells and subsequent nonspecific
disinhibition.

Artifacts 302--307 cross receptor-peak timing with the exact five common
TRN-to-relay gains previously declared for the capacity assay. Match-first
screening retains gains 1.5, 2, and 3; the lowest survivor, gain 1.5, produces
10 relay, 584 TRN, and four nonspecific events and reproduces independently
with complete detector cycles. Its sole fixed mismatch produces exactly the
same aggregate counts, with two events in each vertical relay cell 22, 31, 40,
49, and 58. Close the full interaction under the preregistered lowest-survivor
rule; do not inspect mismatch at gains 2 or 3. Scalar inhibitory capacity can
preserve the repaired match but cannot create SMART's condition-dependent
spatial selection or nonspecific disinhibition.

Artifacts 308--309 test the sole parameter-free radial-annulus ring
interpretation at receptor-peak timing, archived corticoreticular gain, and
hard-bound learned state. The full-detector match retains the correct five
horizontal relay cells with 10 events and 607 genuine-cycle TRN events, but
emits five nonspecific events (50 Hz). It therefore fails before mismatch.
Close the ring-by-peak interaction without a radius, gain, or headroom sweep.

Artifacts 310--311 then admit one explicitly calibrated radius because exact
legacy `ring=true` semantics remain missing. Projection 012's sigma 1.5 fixes
the nearest-arm peak scale at 0.4714045207910316; the two-cell alternative is
effectively the already rejected default annulus and is not repeated. The
nearest-arm candidate produces 25 rather than 20 relay events during the fresh
Figure 6 handoff, so recognition is never constructed. Close geometry-derived
radius calibration before Figure 7 and retain the exact failure counts.

Artifacts 312--313 begin an explicitly labeled mesoscopic reconstruction after
source recovery and geometry-derived calibration were exhausted. The candidate
multiplies only active relay-image pixels by normalized learned support from
adaptive projections 005/007, blended with preregistered floors 0, 0.25, 0.5,
and 0.75; floor 1 is the previously verified uniform-input control. It does not
inspect orientation, condition, or overlap labels and leaves nonspecific/matrix
input, training, weights, cells, and synapses unchanged.

No smooth-floor candidate preserves the exact match. Floors 0/0.25 recruit only
cells 39--41 with 6 relay, 573 TRN, and 6 nonspecific events; floor 0.5 recruits
the correct horizontal set with 8/609/6 events; floor 0.75 gives 13/616/5.
Every candidate misses the 40-Hz nonspecific gate, so none advances to
verification or mismatch. Close smooth multiplicative support blending and do
not interpolate toward the known floor-1 control.

Artifacts 314--315 test a sole standard half-maximum saturated gate rather
than fitting the threshold between the failed smooth floors and uniform-input
control. The gate recruits only learned-field cells 39--41, yielding 9 relay,
607 TRN, and 5 nonspecific events. It fails both the full horizontal relay-set
and 40-Hz match gates. Close thresholded half-maximum support without moving
the threshold after observing the learned map; mismatch remains locked.

Artifacts 316--317 preregister and test fixed-cardinality learned competition.
Cardinality five comes from the five nonzero sensory pixels in the archived
training image, not from Figure 7 output; the gate ranks only selected-category
adaptive weights and supplies no orientation or condition label. The sole
match exactly reproduces relay cells 38--42 with three events each, 635 TRN
events, and four nonspecific events (40 Hz). Registration 318 fixes this
endpoint for one independently rebuilt match with full detector-cycle checks;
mismatch remains locked until that verification passes.

Artifact 319 independently rebuilds the top-five endpoint and reproduces
15/635/4 exactly. Across diagnostic cells 22, 31, 38--42, 49, and 58, emitted
TRN events equal detector upcrossings, arm transitions, and releases (9 or 10
cycles per cell). Registration 320 therefore unlocks one fixed vertical
mismatch with no retuning; all later figures remain locked.

Artifacts 320--321 execute that sole mismatch. Fixed-cardinality competition
recovers the published spatial and causal pathway: only overlap cell 40 emits
three relay events, match exceeds mismatch in active relay cells and TRN events
(635 versus 560), and nonspecific output is directionally disinhibited. All
sampled mismatch TRN events are fresh detector cycles. However, nonspecific
output rises only from four to five events (50 Hz), not the official seven
(70 Hz), so complete Figure 7 still fails. Preserve the successful comparator
and next calibrate only the already named nonspecific compartment-to-event
transfer against match, keeping mismatch as a new holdout.

Artifacts 322--323 reuse the complete predeclared nonspecific soma--proximal
detector-blend grid after upstream selectivity is restored. Every positive
blend preserves the exact 15 relay and 635 TRN match counts, but fractions
0.1--0.7 suppress all nonspecific events and the proximal endpoint emits only
one. No value preserves the four-event/40-Hz match, so none advances to
mismatch. Close a shared arm-and-release coordinate blend. The next output
candidate, if pursued, must separate dendritic arming from somatic rearming
rather than interpolate this family.

Artifacts 324--325 test that split-coordinate hypothesis with the same finite
arm grid and somatic release fixed at its baseline endpoint. Fractions 0.1--0.7
still emit no nonspecific match events; a fully proximal arm instead emits 100
events in 100 ms while relay/TRN remain exactly 15/635. There is no 40-Hz
survivor between silence and pathological 1-kHz output. Close split arming and
somatic rearming without interpolating the 0.7--1 transition; mismatch remains
locked.

Artifacts 326--327 resolve the remaining direct source conflict on projection
049, from TRN to the nonspecific thalamic distal dendrite. The archived ModelDB
record uses conductance 1.461 and 1/4-ms rise/fall constants, whereas the paper
supplement gives 1.5 and 1/7 ms. The complete supplement tuple is represented
as a named runtime convention and tested first against Figure 6, without
altering the historical default. It fails that prerequisite: relay output rises
from the required 20 events to 104, including activity outside the five trained
cells; trained cells 38--42 emit 5/6/6/6/5 events, and the cortical chain stops
after layers 6II and 4. Keep the archived tuple as the calibrated baseline,
close this source alternative, and do not construct its Figure 7 holdouts.
