# Classic SMART constrained calibration

Calibration starts from `classic-smart-source-constrained-v0.1.0` and never
modifies that tag. Its purpose is to infer a small family of plausible missing
legacy conventions—not to optimize arbitrary synaptic weights until plots look
similar.

## Scientific contract

### Research sequence

The active goal remains source-faithful SMART reproduction and an explicitly
validated, versioned baseline. After that, compare neuron models and identify
the parameter regions in which the baseline observations survive. Only after
those comparisons, explore evidence-backed updates to the 2008 anatomy and
dynamics (including cell-type-, area-, and thalamic-nucleus-specific input to
layer 5). Keep these as separately identified circuit variants, not retroactive
changes to the classic baseline. Their benefit, neutrality, or disruption is
an experimental question, not an assumption.

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
- during the initial mismatch comparison, relay output confined to the
  horizontal/vertical overlap while nonoverlap cells are inhibited;
- later mismatch relay output accepted only when preregistered continuous
  traces demonstrate the paper's hyperpolarization-to-T-type-calcium burst
  sequence; ordinary tonic or recurrent escape fails;
- match specific-thalamic output tonic and mismatch output burst-mode;
- TRN output and relay hyperpolarization consistent with the stated inhibitory
  mismatch mechanism;
- over the plotted 100-ms Figure 7c window, nonspecific thalamus fires four
  times (40 Hz) during match and seven times (70 Hz) during mismatch.

The paper does not identify a numerical boundary between initial selection and
later rebound. No observed candidate transition may be reused as a post-hoc
time cutoff; event classification must use physiology and morphology fixed
before the run. Artifact 468 remains failed under its earlier preregistration.
Artifacts 471--473 apply that rule to an identity-locked calcium trace: the
first nonoverlap escapes have no preceding hyperpolarization or T-channel
availability recovery, and the output consists of isolated peaks rather than a
transient burst. The failed endpoint is therefore not a hidden reproduction of
the paper's later rebound mode. The next admissible step is readout-only
localization of the waning inhibitory balance in the same trace, followed by an
explicit revision of the remaining source uncertainty—not calcium fitting.

Source audit 476 replaces part of that uncertainty with an executable 2004
KInNeSS benchmark: 205/205 archived axonal events align to a falling physical
-20-mV soma crossing plus the declared 2-ms delay within one saved 0.05-ms
sample. Artifacts 477--479 test the recovered rule without any effective
projection scaling. Spatial selection and the cortical chain survive, but the
relay produces only one five-cell volley and therefore cannot form a causal
post-teaching pair. That complete source-coherent candidate is closed before
Figure 7. One cross with the already established Figure 6 projection-transfer
convention is admissible as a separately labeled unit-semantics discriminator;
no new threshold or scale may be inferred from the failed run.

Artifacts 480--482 complete that predeclared cross with the prior Figure 6
0.01/0.01/0.03 transfer convention. The recovered detector now yields exactly
four events in each horizontal relay cell, a complete cortical chain, a causal
teaching pair, and positive horizontal bottom-up/top-down map contrasts. It
passes Stage B under the source-supported shape gate. The transfer factors
remain calibrated rather than recovered, so the next step is one
comparator-free, unexpanded 100-ms match before mismatch is opened.

Artifacts 483--485 show that this exact match preserves all five horizontal
relay cells with four events each and correctly terminates the selected
category current, but the paper Table-3 nonspecific cell emits 33 events rather
than four. Mismatch remains locked. Registration 486 returns to Stage B with
the discrete complete SMART.nml nonspecific-cell/serialized-axial alternative;
no channel hybrid, detector threshold, or inhibitory gain may be inferred from
the 330-Hz failure.

Artifacts 486--488 pass that complete official-cell cross through Stage B:
relay confinement/count, cortical recruitment, causal teaching, and both map
shape gates all pass. Registration 489 fixes one fresh 100-ms match with actual
learned weights, no expansion, no prime, and no comparator. Mismatch remains
locked until exact 40-Hz match output is demonstrated.

Artifacts 489--491 fail that exact match at two nonspecific events while
retaining the complete four-volley horizontal relay pattern. Do not interpolate
between the 33-event paper-cell and two-event executable-cell endpoints.
Registration 492 instead fixes the one-factor official-source axial comparison:
the complete executable nonspecific cell under paper Equation 2, starting at
the full Figure 6 prerequisite.

Artifacts 492--494 pass the paper-axial endpoint through Figure 6. The reported
22-event nonspecific learning output cannot substitute for the independent
Figure 7 match gate. Registration 495 requires one unchanged match with exact
fresh learned weights and the pre-existing four-event target.

Artifacts 495--497 fail that match at 24 nonspecific events while preserving
the exact relay match. Both axial endpoints are closed. Registration 498 fixes
the next source-only factorial: the paper supplement's longer/stronger distal
TRN GABA record, with no fitted conductance or kinetics and Figure 6 required
before recognition.

Artifacts 498--500 pass the supplement-GABA endpoint through Figure 6 with
registered learning readouts identical to the archived-GABA control.
Registration 501 fixes one recognition match that changes only the official
projection-049 tuple; mismatch remains locked until exact 40 Hz is obtained.

Artifacts 501--503 reject the supplement tuple at the same 24-event match
output as its control. Registration 504 returns to Figure 6 for the remaining
paper-versus-executable nonspecific calcium-kinetics equation swap. Calcium
density and all other factors remain fixed; no kinetic interpolation is
allowed.

Artifacts 504--506 pass the paper nonspecific calcium kinetics through Figure
6 at fixed Table-3 density. Registration 507 requires one fresh match against
the unchanged four-event target; no learning-condition rate or calcium value
may be reused to tune it.

Artifacts 507--509 fail the paper-kinetics match at 18 nonspecific events.
Registration 510 fixes the remaining executable-cell axial-by-kinetics
combination at Figure 6: serialized KInNeSS axial edges plus paper calcium
kinetics. No continuous interpolation is allowed.

Artifacts 510--512 pass that final combination through Figure 6. Registration
513 requires its independent four-event recognition match; failure closes the
two-by-two axial/kinetics source factorial without interpolation.

Artifacts 513--515 close the executable-cell factorial at one event for the
last endpoint; the complete match matrix is 2/24/18/1 events and contains no
four-event survivor. Registration 516 tests the paper-coherent nonspecific
cell/axial/calcium combination at Figure 6, while retaining the recovered
detector and clearly labeling calibrated network conventions.

Artifacts 516--518 pass the paper-coherent nonspecific cellular bundle through
Figure 6. Registration 519 requires one independent match; its four-event gate
cannot be altered by the 23-event learning-condition observation.

Artifacts 519--521 fail the paper-coherent match at 24 nonspecific events while
retaining exact relay matching. No available cellular source bundle now
survives the match gate. Before another behavioral run, recover direct evidence
for nonspecific/intralaminar event handling from a legacy benchmark, manual, or
source body; do not fit a population-specific threshold from these outcomes.

Artifact 522 identifies the exact SMART-era KInNeSS 0.3.4 RC2 and SANNDRA
1.2.0 RC2 releases, but the linked source archives are not preserved. The
official examples inventory contains no nonspecific benchmark. The surviving
manual documents one axonal soma-to-binary conversion and SMART.nml exposes no
population-specific detector parameter; the SANNDRA history names
`spikeevents.h` but does not preserve its body. Consequently, a
nonspecific-only threshold is not authorized. The next preregistered work must
audit RK4 scheduling, initialization, and axon/synapse update order against the
surviving executable relay benchmark before reopening Figure 7.

Artifacts 523--525 verify that the registered Brian2 translation already
implements a next-step postsynaptic visibility boundary. With a 0.01-ms clock
and the archived 0.1-ms projection delay, a fixed impulse leaves the summed
gate zero at 0.10 ms and exposes it at 0.11 ms. Do not add another global delay.
The same audit partitions the failed match into three nonspecific events before
any upstream circuit event and 21 later events.

Artifacts 526--528 causally localize all 21 later events to the intact
TRN-to-nonspecific GABA route: removing projections 047--049 deletes them,
whereas removing layer-6II projections 050--051 changes no event count. These
ablations are diagnostics and cannot be selected as models. Next add a
default-off nonspecific dendritic T-current ablation and verify default identity
before one registered match-only causal run. Do not tune calcium density, alter
the common detector, or open mismatch from this diagnostic.

Artifacts 529--531 add the default-off nonspecific T-current ablation and run
the one registered connected match. Nonspecific output falls from 24 events to
one, while the exact relay sequence and TRN event count are preserved. Precise
TRN and category sequences are not identical, so feedback prevents a strict
cell-autonomous conclusion. Next record and replay the intact external and
projection-047--051 inputs into an isolated nonspecific cell; require an exact
intact replay before comparing the zero-T-current arm. Do not fit a current,
conductance, delay, or detector from that replay.

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
cells; trained cells 38--42 emit 5/6/6/6/5 events. The initial runner did not
monitor all cortical-chain populations, so its separate chain gate is not used
as evidence; the relay failures alone decisively reject the prerequisite. Keep
the archived tuple as the calibrated baseline, close this source alternative,
and do not construct its Figure 7 holdouts.

Artifacts 328--330 screen one common multiplicative transfer on the three
archived TRN-to-nonspecific GABA contacts while preserving their relative
compartment weights. Artifact 329 is superseded because its runner omitted the
three cortical monitors required by its own Figure 6 chain gate; the unchanged
grid is rerun correctly in Artifact 330. All four scales pass complete Figure
6 and preserve 15 relay and 635 genuine-cycle TRN match events. Scales 0.25 and
0.5 emit only two nonspecific events; scales 0.75 and 1.0 emit the required
four. The preregistered weakest-survivor rule therefore fixes 0.75 for an
independent Figure 6-plus-match rebuild. Do not inspect mismatch until that
verification passes.

Artifacts 331--332 independently rebuild the fixed 0.75 candidate. The fresh
network again passes every Figure 6 prerequisite and reproduces the match at
15 relay, 635 TRN, and four nonspecific events. For every sampled TRN cell,
emitted events equal threshold upcrossings, detector arms, and releases. This
verification authorizes one separately registered vertical mismatch at the
unchanged 0.75 scale; it does not itself establish Figure 7 reproduction.

Artifacts 333--334 execute that sole fixed mismatch. The successful comparator
path remains intact: only relay cell 40 emits, with three events; match exceeds
mismatch in active relay cells and TRN events (635 versus 560); and sampled TRN
events are complete detector cycles. Nonspecific output, however, is four
events in both conditions, rather than 40-Hz match versus 70-Hz mismatch. Close
common TRN-to-nonspecific GABA transfer scaling: weakening the source weights
to the lowest match survivor removes the prior 50-Hz directional effect rather
than amplifying it. Do not interpolate between 0.75 and 1.0 after this holdout.

Artifacts 335--336 test the article's conflicting Methods 4.6 statement that
every existing calcium current uses 250 mS/cm2. Applied literally to the
current calibrated network, the discrete source alternative destroys Figure 6
selectivity: all 81 relay cells emit once, layer 4 emits 243 events, layers 2/3
and 5 emit 81 and 162, causal learning timing fails, and the learned map loses
horizontal contrast. Reject the global-250 convention before recognition;
retain compartment-specific Table 3/ModelDB densities and do not introduce a
nonspecific-only source hybrid after observing this failure.

Artifacts 337--338 audit the fixed top-five pair without changing any model or
protocol parameter. Match and mismatch reproduce 4 and 5 nonspecific events,
and each count exactly equals +30-mV detector upcrossings, arm transitions,
and release transitions; neither condition ends with a latched detector. The
underlying soma has 22 versus 23 positive local maxima, most of which remain
below +30 mV. No single threshold applied to the observed peak distributions
can preserve four match events and yield seven mismatch events. Close shared
nonspecific threshold/rearming calibration. The next candidate must alter a
predeclared condition-sensitive input or membrane mechanism while preserving
the exact match and spatial pathway, not reinterpret detector bookkeeping.

Artifacts 339--344 cross the fixed top-five comparator with the already
source-derived 7.85-ms receptor-arrival alignment. The sole match candidate
passes at 15/633/4 and repeats exactly in an independent full-detector rebuild.
The preregistered mismatch then retains only overlap relay cell 40 and reduces
TRN output to 560, but nonspecific output stays at four events. Close the
interaction without timing or cardinality interpolation. Relative to the old
spatially incorrect 4/7 pair, the corrected mismatch reduces the peak
layer-6II-to-nonspecific NMDA gate from approximately 1.55 to 0.546 while the
TRN GABA integral changes from about 1229 to 1200 gate-ms. This establishes
that the earlier exact 70-Hz rate relied on excitation carried by nonoverlap
paths that should have been silent. The next mechanism must make genuine
nonspecific disinhibition dominate after spatial mismatch, not restore those
incorrect relay paths.

Artifacts 345--346 isolate that disinhibitory path by removing recognition-only
layer-6II-to-nonspecific AMPA/NMDA projections 050/051. Their gates and currents
are exactly zero, but the full counts remain 15/635/4 in match and 3/560/5 in
mismatch. Because mismatch still has the lower TRN GABA integral and one extra
nonspecific event, the inhibitory path has the correct causal sign. Because
the ablation changes no aggregate rate, cortical excitation is not the source
of the missing two mismatch events. Treat this network only as a causal
diagnostic. The next calibration target is the TRN-GABA-to-nonspecific membrane
transfer, with detector and corticointralaminar hypotheses closed.

Artifacts 347--348 localize the intact path's nonlinear compartment roles with
four fixed, non-promotable ablations. Removing somatic GABA yields 4/4
match/mismatch nonspecific events, removing proximal GABA yields 0/0, removing
distal GABA yields 5/5, and removing all three yields 0/0. Upstream relay/TRN
counts remain fixed at 15/635 and 3/560. The result rejects a scalar
"less inhibition means more arousal" interpretation: proximal inhibition is
required for rebound output, distal inhibition suppresses one match event, and
somatic inhibition supplies the one-event mismatch advantage. Do not infer a
weight from an ablation endpoint. The next admissible step is one separately
registered intact-network transfer mechanism that preserves all three source
contacts, passes fresh Figure 6 and match before mismatch, and remains labeled
as calibrated reconstruction rather than recovered source.

Artifacts 349--350 apply the discrete primary-paper T-type gate equations only
to the nonspecific cell, leaving the selected relay/TRN kinetics and all
synapses fixed. The fresh network passes every Figure 6 prerequisite, but
match produces 15 relay, 639 TRN, and eight nonspecific events (80 Hz). It
therefore fails before mismatch, which remains uninspected. Close this
population-specific source alternative and retain the executable-backup
nonspecific kinetics; do not interpolate between the two equation families.

Artifacts 351--352 test one preregistered twofold sensitivity on only somatic
TRN-to-nonspecific GABA projection 047. The altered network passes complete
Figure 6 and exact match at 15 relay, 635 TRN, and four nonspecific events,
with fresh detector cycles. Do not inspect mismatch yet. Independently rebuild
the same persistent factor-2 network and repeat Figure 6 plus match; only that
verification can authorize one fixed mismatch.

Artifacts 353--354 complete that independent verification. The newly rebuilt
network repeats every Figure 6 and match gate, including exact 20-event
training and 15/635/4 recognition counts. Register one vertical mismatch with
the identical factor-2 somatic transfer. Do not change or add any scale after
the holdout result; unlock Figure 10 only if every Figure 7 pathway and 40/70
Hz gate passes.

Artifacts 355--356 run that one fixed mismatch. The pair retains exact match
and overlap-only mismatch, with relay/TRN/nonspecific totals of 15/635/4 and
3/560/5. Every gate passes except the published seven-event/70-Hz mismatch
target. Close factor-2 somatic GABA without adding another scale. The next
candidate must be an independently motivated intact-network membrane-transfer
mechanism, not a stronger value selected from this failed holdout.

Artifacts 357--358 test archived KInNeSS serialized-edge axial coupling only
inside the nonspecific three-compartment cell, avoiding the upstream TRN
confound of Artifact 138. Figure 6 remains exact, and match preserves 15 relay
and 635 TRN events, but nonspecific soma never approaches threshold and emits
zero events. Close this discrete axial source alternative before mismatch;
retain paper-literal nonspecific coupling and do not add an axial scale or
interpolation.

Artifacts 359--360 replace only the nonspecific intrinsic cell with its
complete recovered executable record. Figure 6 remains exact and upstream
match stays at 15 relay/635 TRN, but the soma peaks at 28.14 mV and emits zero
events under the paper +30-mV detector. Close this cell record as a standalone
alternative and retain the paper Table 3 intrinsic cell; do not create a
post-hoc sodium/potassium hybrid. The only source-coherent follow-up is the
discrete executable-cell plus contemporaneous KInNeSS detector combination,
registered before execution and tested match-first.

Artifacts 361--362 add the exact KInNeSS -20-mV event arm to the recovered
nonspecific cell while retaining the calibrated peak latch. Match emits 287
nonspecific events, beginning at alternating 0.02-ms intervals, because the
-20-to-0-mV arm/release regions overlap. Reject this handler combination
before mismatch and do not interpolate the threshold. The next discrete test
may use the existing hysteretic handler only in nonspecific thalamus; all
upstream detectors and parameters must remain fixed.

Artifacts 363--364 execute that isolated hysteretic-handler test. Hysteresis
eliminates alternating-step rearming but still produces 22 nonspecific match
events while Figure 6 and upstream 15/635 match counts remain exact. Reject the
handler before mismatch. Close the complete recovered-cell/KInNeSS-threshold
family without adding a refractory period, threshold interpolation, or
individual sodium/potassium tuning.

Artifacts 365--366 decompose the retained 4/5 pair without changing any model
or protocol parameter. Match and mismatch are identical through the shared
startup and then retain exact 15/635/4 and overlap-only 3/560/5 controls. The
post-divergence mismatch action potentials are inhibitory-rebound events:
proximal and distal T-type currents of roughly 6--8 nA nearly cancel GABA
currents of similar magnitude, while layer-6II current is only tens of pA.
The intervening positive soma maxima are Na/K-dominated after-spike ringing,
not two nearly suprathreshold independent rebound cycles. Next audit the
temporal cycle content of the fixed aggregate TRN-to-nonspecific drive; do not
amplify the ringing or fit an intrinsic conductance from this diagnostic.

Artifacts 367--368 analyze the already recorded TRN events with one fixed
offline rule: 1-ms bins containing at least 10% of the 81-cell population are
active, and contiguous active bins form one volley. Match contains ten volleys
but four nonspecific responses. Mismatch contains exactly seven volleys of
81/81/81/81/81/81/57 events, but the first two have no nonspecific response and
the last five each have one. The official 70-Hz temporal opportunities already
exist upstream; the two-event deficit is downstream transfer loss. The next
admissible candidate is the same population-uniform KInNeSS handler already
used for TRN, applied with the complete recovered nonspecific cell and tested
match-first. Do not alter ionic or synaptic parameters.

Artifacts 369--370 execute that one uniform-handler transfer. The fresh
network passes every Figure 6 prerequisite and keeps upstream match exact at
15 relay and 635 TRN events, but the recovered nonspecific cell still emits 22
events rather than four. Moving the falling event coordinate from 0 to -30 mV
therefore does not group the repeated voltage excursions into the official
match spikes. Reject before mismatch and close fixed-threshold KInNeSS handler
transfer without interpolating either voltage.

Artifacts 371--374 calibrate one explicit non-source arm at +20 mV using only
the recovered-cell match waveform, then independently rebuild it. Both fresh
runs pass every Figure 6 gate and reproduce exact 15/635/4 match activity with
the same runtime fingerprint. Artifacts 375--376 open one fixed mismatch
holdout: relay activity remains overlap-only at cell 40 and TRN falls to 560
events, but nonspecific output remains five rather than seven. The recovered
mismatch waveform has only six positive somatic maxima, so no lower shared
amplitude threshold can create seven independent cycles. Reject +20 mV and
close scalar event-arm calibration; do not inspect another threshold.

Artifacts 377--378 test the paper/manual initialization ambiguity without
consulting another behavioral holdout. Starting the complete recovered
nonspecific thalamic cell at its serialized -64 mV leak reversals with gates at
their corresponding steady-state values does not relax to a quiescent rest in
500 ms. The trajectory remains finite, but the last 100 ms span 51.39 mV at
the soma (about -56.98 to -5.59 mV), with smaller persistent dendritic
oscillations. Therefore a chosen pre-protocol warm-up cannot be called a
source-grounded resting-potential correction. Close equilibration-time fitting;
if initialization is revisited, it must come from an independently derived
fixed point or newly recovered simulator source, not Figure 7 output.

Artifacts 379--381 derive and independently verify the only fixed point found
from a 125-start bounded current-balance search of the recovered nonspecific
cell. Fifty-four starts converge to soma/proximal/distal voltages of about
-37.643/-29.377/-25.709 mV. Refined to a residual below 2e-11 pA, that state is
exactly stationary for 100 ms. Exact stationarity is not stability. Artifacts
382--383 therefore apply one preregistered +1e-6-mV somatic perturbation with
all gates left at the fixed-point state. It grows into 56.32 mV of somatic
motion within 100 ms. Reject this point as a robust resting initialization and
keep Figure 7 locked. Do not round the point into a behavioral candidate or
fit its precision; only newly recovered primary source could require that
non-robust initialization.

Artifacts 384--385 revisit the archived external-input `connectFromAll`
semantics as one discrete source interpretation. KInNeSS Equations 5--6 say a
zero-valued input remains an extra leak, while SMART.nml marks the 9x9 input to
nonspecific and matrix thalamus as `connectFromAll`. Counting all 81 image
locations rather than only the five nonzero bar pixels preserves every Figure
6 gate and exact upstream match activity at 15 relay/635 TRN events, but
nonspecific match output rises to six events (60 Hz). This configuration fails
before mismatch. Audit 386 subsequently establishes that source count was one
during the cue interval and reset to one on clearing. Thus this experiment
does not test persistent 81-location topology or resolve source semantics.
The next source audit must account for the entire input lifecycle.

Methodological correction (2026-09-05): repeated Figure 7 mismatch evaluations
make that condition part of calibration, despite earlier "holdout" wording.
Deterministic rebuilds establish repeatability, not independent validation.
Source interpretations must be assessed against source evidence; failure to
recover a target under a calibrated configuration cannot falsify the source
interpretation itself. Future registrations must disclose prior exploratory
observations, including those preceding the resting-state audit registration.

Artifacts 413--426 resolve the mixed `Relay_INT` input route and restore the
paper's simultaneous Figure 7 timing without a reconstructed comparator.
Treating archived projection 042 as the declared voltage-driven image input
passes the complete Figure 6 prerequisite. With the actual learned arrays,
unit headroom, simultaneous bottom-up/top-down onset, and no calcium ablation,
match activates exactly horizontal relay cells 38--42 four times each and
produces 605 genuine-cycle TRN events. The declared image also makes exactly
those five interneurons emit once at 2.96 ms. Their projection-002 GABA transfer
is intact: the direct compiled assay peaks at gate 1.861 and about -814 pA, and
the first relay events sample about -759 to -760 pA. The apparent near-zero
late range was measured only after 40 ms, after this fast gate decayed.

The simultaneous candidate still fails the official match target because the
nonspecific cell emits five events at 0.74, 50.36, 56.04, 73.81, and 94.08 ms,
not four. The 0.74-ms event precedes all recognition-path events and is a
startup-state response. Retain it in the score: do not silently discard it or
select an equilibration interval from Figure 7. Registration 424 therefore
does not authorize mismatch, and the classic behavioral baseline remains
unfrozen. The next admissible work is source-grounded recovery of the legacy
protocol/initial-state lifecycle, not another fitted conductance or threshold.

Artifacts 427--429 then run one unchanged vertical condition solely to localize
the failed simultaneous-onset match. Match and mismatch each activate all five
directly driven bar cells four times; TRN output is 605 versus 603 events and
nonspecific output is five versus five. Thus only the weak correct-sign TRN
ordering survives. Overlap-only relay selection, fewer active mismatch relay
cells, disinhibition, and the exact 40/70-Hz pair all fail without a comparator.

The event-time audit exposes a protocol distinction. The selected layer-6II
cell emits at 5.85 ms and all four archived learned-feedback projections have
2-ms delays, so their earliest relay arrival is 7.85 ms. Yet the first relay
volley occurs at 6.03--6.05 ms with exactly zero top-down excitation current.
Consequently simultaneous *command onset* does not implement the paper's
simultaneous bottom-up/top-down excitation at LGN. It lets all mismatch image
cells fire before the two-against-one circuit can compare them. The earlier
7.85-ms lead remains a derived receptor-arrival alignment, not a published
protocol number. Next test that alignment with a sustained Figure 7 expectation
as one discrete current-duration interpretation; run match before mismatch and
do not alter model parameters or scoring.

Artifacts 430--432 test that fixed receptor-arrival alignment with a sustained
800-pA category current. The horizontal active set remains exact but every
relay cell emits only twice (10 total), TRN emits 564 times, and nonspecific
thalamus emits five times. Sustaining the current therefore changes the old
one-event result from 10/604/6 to 10/564/5 but does not restore relay recurrence
or 40-Hz match output. Both current-duration endpoints fail under the declared
interneuron input, so no intermediate duration may be selected from Figure 7.
The 81-cell 5.68-ms TRN cue-lead volley and absent interneuron output are shared
with the one-event alignment and precede sensory onset. Registration 433 opens
one unchanged vertical run only to test whether the arrival-aligned sustained
circuit has the qualitative overlap-selection mechanism; it is explicitly not
a validation holdout and cannot promote the baseline.

Artifacts 433--435 complete that bounded localization. The fixed vertical
mismatch emits twice in every directly driven relay cell 22/31/40/49/58 (10
events), versus twice in every matched horizontal cell (also 10). TRN output
is 564 for match and 568 for mismatch, while both conditions produce five
nonspecific events. None of overlap-only selection, match-greater relay/TRN
recruitment, mismatch disinhibition, or the exact 40/70-Hz pair is recovered.

The timing intervention also crosses the intrinsic startup transient. All 81
TRN cells emit at 5.68 ms, 2.17 ms before arrival-aligned sensory onset, and no
Relay_INT image cell fires during the subsequent trial; simultaneous command
onset instead let the five image-aligned interneurons fire at 2.96 ms. Thus a
7.85-ms lead changes the circuit state and removes feed-forward inhibition; it
cannot be treated as a neutral correction. Reject arrival-aligned sustained
timing and close cue-duration fitting. Return to simultaneous command onset as
the source-facing protocol while auditing the unavailable legacy initial-state
lifecycle. If primary runtime evidence remains unavailable, separate any
multi-figure behavioral calibration explicitly from the source-constrained
classic reconstruction.

Artifacts 436--438 open the first explicitly behavioral mechanism screen after
the source-facing timing and initialization routes are closed. A uniform relay
image gain of 1.0 repeats the five-cell horizontal versus five-cell vertical
escape with identical 194-event TRN output during 25-ms trials. Gains 0.75,
0.5, and 0.25 silence relay output in both conditions; each leaves the five
image-aligned Relay_INT events intact and yields identical 243-event TRN
output. No gain passes, so the registered coarse input-gain family is closed
without refinement. Next test one source-bounded learned-feedback endpoint
under declared-interneuron input; do not combine it with input scaling or a
reconstructed comparator.

Artifacts 439--441 test the sole maximal learned-feedback endpoint with the
declared interneuron route. Scaling selected category row 40 by the largest
common archived-bound factor, 3.653169, leaves the 25-ms match and mismatch
condition-insensitive: both emit one relay event in each of their five driven
cells at 6.03--6.05 ms and both produce 194 TRN events. These relay events
precede learned-feedback arrival, so headroom cannot prevent them. Close
headroom under this route without interpolation. Next test a finite logarithmic
gain on the already verified projection-002 feed-forward inhibitory transfer;
this is behavioral causal calibration, not source recovery.

Artifacts 442--444 test that finite projection-002 gain screen at 1, 2, 4,
and 8 times its reconstructed Relay_INT-to-relay GABA weight. Gains 1 and 2
leave the full five-cell horizontal and vertical sensory bars active, with
equal within-pair TRN counts of 194 and 198. Gains 4 and 8 silence every relay
cell in both conditions and produce 243 TRN events in each. The image-aligned
Relay_INT events remain present at every gain, so the result is a genuine
transfer-strength test rather than loss of interneuron recruitment. No gain
passes. Close this grid without interpolation or extension. Before another
behavioral screen, audit the source-record timing of relay excitation,
Relay_INT inhibition, and learned feedback at the first relay volley.

Registration 445 resolves that timing order from the archived records and the
fixed 800-pA control: Relay_INT inhibition arrives at 3.06 ms, relay events
begin at 6.03--6.05 ms, and the selected layer-6II event at 5.85 ms cannot
deliver learned feedback through its 2-ms delay until 7.85 ms. The current
amplitude remains unreported in Methods 4.9. Because the declared Relay_INT
input is a later upstream source-semantic correction, reopen only the already
bounded 600- and 1000-pA endpoints under simultaneous command onset. Do not
rerun 800 pA, interpolate, extend the bounds, or combine this timing test with
another parameter change. A 25-ms survivor is only eligible for a fresh
100-ms pair.

Artifacts 445--447 close that one-time current endpoint recheck. At 600 pA,
the category event occurs at 8.92 ms and learned feedback arrives at 10.92 ms,
after the 6.03-ms relay volley. At 1000 pA, the category event advances to
4.49 ms but its 2-ms-delayed feedback still arrives at 6.49 ms, 0.46 ms after
relay escape. Both endpoints preserve the full driven bar in match and
mismatch and produce equal 194-event TRN output. Together with the fixed
800-pA control, the bounded amplitude family has no survivor and is closed
without interpolation or expansion. Next test receptor-level simultaneity as
a protocol-semantics diagnostic: initialize the archived learned feedback
arrival at sensory onset without integrating a top-down-only cue interval or
altering source delays and weights.

Registration 448 defines that one-factor diagnostic. At sensory onset, one
selected-category arrival is initialized at the four archived layer-6II-to-
relay on-center records and the two layer-6II-to-TRN off-surround records.
Weights, later delays, receptor kinetics, the ordinary 800-pA one-event cue,
fresh Figure 6 handoff, and declared Relay_INT input remain fixed. This
deliberately collapses the six first-event delay differences and is not legacy
runtime recovery. It asks only whether receptor-level simultaneity is causally
sufficient for the paper's two-against-one spatial comparison in a 25-ms pair.

Artifacts 448--450 reject combined receptor-level priming as a sufficient
timing repair. All six records are verifiably primed for category source 40,
but match and mismatch both lose every relay event and each produces 200 TRN
events. The ordinary category event remains at 5.85 ms and Relay_INT image
events remain present, so the intervention did not erase cue delivery. Close
unit first-arrival priming without fitting its amplitude or time. The result
points to an imbalance between direct learned on-center excitation and the
TRN-mediated off-surround. Next decompose the same fixed prime into relay-only
and TRN-only arms, compared against the existing no-prime and combined-prime
controls.

Registration 451 fixes that decomposition before execution. Both arms use the
same source-40 unit arrival at sensory onset. The relay arm primes records
003/005/006/007; the TRN arm primes records 009/012. The no-prime and combined
controls are imported rather than rerun. This experiment localizes causal
dominance only: neither arm is source recovery, and no amplitude, timing,
weight, delay, or protocol dimension may be adjusted from its outcome.

Artifacts 451--453 show an asymmetric decomposition. Relay on-center-only
priming is indistinguishable from no prime: match and mismatch retain their
complete driven bars and equal 194-event TRN trains. TRN off-surround-only
priming is indistinguishable from the combined prime: both conditions lose all
relay output and produce 200 TRN events. Thus early off-surround recruitment
dominates while the actual Figure 6 learned on-center cannot rescue matched
cells. Next test one source-bounded interaction endpoint by crossing combined
receptor simultaneity with the maximal archived-bound selected-row headroom
already fixed in Artifacts 439--441. No grid or interpolation is authorized.

Registration 454 fixes the sole endpoint cross before execution: the actual
fresh Figure 6 selected row is scaled by the previously measured maximal
common factor 3.6531686628985414, then the same unit arrival is initialized at
all six on-center/off-surround records at sensory onset. This asks whether
stronger bounded learned support can rescue only matched two-against-one cells
from the verified early off-surround. No intermediate headroom, prime change,
or additional parameter dimension may follow from this endpoint.

Artifacts 454--456 reject the maximal interaction endpoint. The fixed factor
3.6531686628985414 plus all-six-record prime rescues only outer horizontal
cells 38 and 42 at 11.91 ms; matched cells 39--41 remain silent, mismatch is
fully silent rather than overlap-only, and TRN direction is reversed at
189 versus 200 events. This partial response proves that bounded learned
support interacts with early inhibition, but not with the paper's spatial
geometry. Close the endpoint without interpolation. Next repeat the exact
candidate for 55 ms with pathway readouts only, to localize direct on-center,
TRN GABA, Relay_INT GABA, bottom-up current, and compartment-voltage balances.

Registration 457 adds only fixed-time pathway readouts to that rejected
candidate. The run is extended to 55 ms solely to satisfy the established
diagnostic window, and its first 25-ms relay, TRN, and nonspecific event trains
must exactly reproduce Artifact 455. Samples at 0, 1, 2, 3, 4, 5, 6, 8, 10,
11, 11.9, 12, and 15 ms cover all nine diagnostic relay cells, including
silent cells. No correction may be selected from the observed currents.

Artifacts 457--459 preserve that prefix exactly and expose delayed, incomplete
condition separation. Match expands from outer cells 38/42 at 11.91 ms to all
five horizontal cells near 40 ms, while mismatch remains fully silent through
55 ms. This is not Figure 7: the overlap cell is absent and TRN output is lower
in match (377 versus 443). All 117 registered current samples and all 39
voltage samples for overlap cell 40 are identical between conditions through
15 ms. At 4 ms, direct input plus top-down excitation minus TRN inhibition is
about +511 pA, but the declared image-driven Relay_INT contribution is about
-770 pA, reversing the net four-path balance to about -259 pA. Before any
further calibration, return to the archived mixed Relay_INT gate and determine
which of its direct-input and nested-projection metadata can coexist under the
KInNeSS input equations. No projection-002 gain or timing value is inferred
from this trace.

Audit 460 resolves the next source decision conservatively. The KInNeSS
framework defines `dependency=input` through Equations 5--6 as an external
four-color voltage-driven channel, matching the gate's green sensitivity and
direct `connectFromOne` method. Its nested Layer-4 metadata cannot instead be
compiled as an ordinary ligand gate without violating that definition, and no
surviving source specifies an additive hybrid. Retain `declared_external_input`
and reject a newly invented combined route. Because exact legacy gain handling
is unavailable, a subsequent finite projection-002 scaling screen is permitted
only as transparent behavior calibration, with the source value retained as
control and a mandatory persistent Figure 6 prerequisite for any survivor.

Artifacts 461--463 test the finite trace-derived gains 0.9, 0.8, 0.7, and 0.6
with the source gain 1.0 imported as the hash-pinned rejected control. Gain 0.8
is the sole short-window survivor: all five matched cells fire, mismatch is
confined to overlap cell 40, and match has more TRN events (183 versus 165).
The neighboring values fail on opposite sides, so this is a narrow effective
regime rather than evidence for a robust or recovered source constant.
Registration 464 applies exactly 0.8 persistently during Figure 6. Every
existing learning prerequisite remains fixed; only a complete pass may unlock
one independently registered 100-ms Figure 7 pair.

Artifacts 464--466 pass the persistent Figure 6 prerequisite at gain 0.8:
20 relay events remain divided four per horizontal cell, the cortical chain
and causal pair are complete, and top-down contrast is 0.563306. The changed
learned state fixes a new maximal archived-bound common factor of
3.6651418062578482. Registration 467 locks one 100-ms match/mismatch pair with
that factor, the same all-six unit receptor prime, and exact 4/7 nonspecific
event targets. The pair cannot tune any parameter; a pass only unlocks holdout
evaluation, while a failure closes this candidate.

Artifacts 523--525 close the scheduler alternative: Brian's summed receptor
gate becomes visible on the next integration step, matching KInNeSS's
synchronous data-exchange contract, so no additional global step delay is
authorized. Artifacts 526--528 then show that all 21 late nonspecific events in
the paper-coherent candidate require TRN GABA, whereas removing direct
layer-6II excitation leaves all 24 events unchanged. The connected T-current
ablation in Artifacts 529--531 reduces output to one event but changes later
TRN and category sequences through feedback, requiring an isolated replay.

Artifacts 532--534 complete that replay without fitting. The connected source
exactly repeats Artifact 520. An isolated intact cell driven by the recorded
five receptor gates and external inputs reproduces all 24 event times and all
state samples with zero error. Under the same fixed inputs, zeroing only the
two dendritic T conductances leaves one early event and no late events. This
establishes cell-autonomous T-current necessity for the excess late activity,
but it does not recover the unavailable source detail or reproduce Figure 7.
The next permitted step is a separately labeled, match-only one-dimensional
effective T-conductance calibration on the fixed replay trace, followed by one
fresh connected verification. Mismatch and all spectral/reset holdouts remain
locked.

Artifacts 535--537 execute the complete fixed 17-point match-only scale grid.
Event count rises from one at scale zero to 24 at the paper value. The unique
four-event point is scale 0.1875 (46.875 mS/cm2), selected without interpolation
by the preregistered closest-to-source rule. Because this is an 81.25-percent
reduction and the isolated train consists of three startup events plus one late
event, it remains a low-source-confidence effective candidate. Add it as an
explicit fingerprinted convention and run one fresh connected Figure 6 and
match with no other change. Do not open mismatch unless both prerequisites
pass exactly.

Artifacts 538--540 verify scale 0.1875 in the complete feedback network. The
fresh Figure 6 prerequisite passes every gate, including 20 relay events split
four per trained horizontal cell and the complete cortical learning chain. The
connected match also passes every fixed gate: cells 38--42 emit 20 relay events
and nonspecific thalamus emits four events in 100 ms. One vertical mismatch is
now authorized under the identical fingerprint. No mismatch-driven retuning is
permitted; failure closes this candidate, while a pass advances to the
preregistered spectral/reset holdouts.

Artifacts 541--543 execute the fixed independent pair and reject the candidate.
The match repeats its five-cell, 20-relay-event, four-nonspecific-event result,
but mismatch also activates all five driven cells with 20 relay events. TRN
output increases in mismatch (608 versus 549) instead of decreasing, and both
nonspecific conditions remain at 40 Hz. Close scale 0.1875 without retuning.
The next work must address the source-described two-against-one relay/TRN
comparison as a separately labeled reconstruction; intrinsic event-gain
calibration alone cannot reproduce Figure 7.

Registration 544 fixes that next reconstruction before execution. The primary
paper constrains simultaneous bottom-up and learned top-down excitation *at the
LGN cell*, whereas the public protocol reconstruction's first relay volley
precedes learned-feedback arrival. Earlier timing and receptor-priming families
are closed, so the registered cross uses the previously tested five-target
learned-field transform as an explicit mesoscopic coincidence gate. It combines
that transform once, without a grid, with independently match-selected
nonspecific T scale 0.1875. The fresh Figure 6 handoff and the complete 4/7-Hz,
relay-subset, and TRN-order contract are fixed. Failure closes the interaction;
success would be calibrated behavioral reproduction, not recovered 2008 source.

Artifacts 544--546 complete the fixed cross. Fresh Figure 6 passes, match
retains all five horizontal relay cells with 20 events, and mismatch is reduced
to three events in overlap cell 40 alone. The nonspecific output exactly
matches the rendered Figure 7c traces at 4 versus 7 events (40/70 Hz). One
predeclared mechanistic gate fails: TRN output is 549 in match versus 584 in
mismatch, despite the much larger matched relay train. The interaction is
closed without retuning. Next perform an unchanged source-decomposition repeat
that preserves full TRN event trains and separates relay-collateral drive from
direct layer-6II drive; only after locating the inversion may a new source- or
mechanism-justified candidate be registered.

Registration 547 fixes that diagnostic repeat with an unchanged profile and
hash-pinned script. It adds only complete TRN event trains, per-cell and 1-ms
event summaries, and source-separated relay-collateral, corticoreticular, and
recurrent TRN readouts. The failed candidate remains closed. The diagnostic
will determine whether the 549-versus-584 inversion originates in afferent
drive or in recurrent TRN dynamics before any further mechanism is proposed.

Artifacts 547--549 localize the inversion to recurrent spatial TRN dynamics.
The exact 20/3 relay, 549/584 TRN, and 4/7 nonspecific pair repeats with added
readouts. Across nine cells spanning the learned and mismatching fields, the
match/mismatch integral ratios are 5.76 for relay-collateral AMPA, 3.06 for
layer-6II AMPA, and 2.96 for layer-6II NMDA. Those cells emit 79 versus 72
events, in the correct direction. The remaining 72 TRN cells emit 470 versus
512; mismatch organizes repeated near-whole-sheet volleys despite weaker
afferent drive. Next use one recognition-only dual ablation of recurrent TRN
GABA records 008/011 as a causal diagnostic. Keep gap junctions and every
afferent path intact; do not promote the ablation or infer a replacement gain.

Registration 550 fixes that causal ablation before execution. Both chemical
TRN-to-TRN GABA records (008 soma and 011 proximal) are removed only during the
fresh recognition pair. Relay and layer-6II afferents, within-TRN gap junctions,
all TRN outputs, the learned coincidence transform, and effective T scale are
unchanged. The diagnostic asks only whether recurrent GABA is necessary for
the peripheral mismatch-wide volleys; it cannot become a baseline candidate or
select a replacement strength.

Artifacts 550--552 show that recurrent chemical GABA is essential for a stable
TRN regime but is not the sole source of the count inversion. Removing records
008/011 raises TRN output from 549/584 to 2008/2028 events and makes nearly all
81 cells emit 24--26 events. It also destroys the exact 4/7 nonspecific result,
yielding 22/22. The mismatch-greater-than-match order nevertheless persists by
20 events, entirely outside the nine field-centered diagnostic cells. Next
preregister a recognition-only ablation of projection 013 gap junctions while
retaining both recurrent GABA records. Do not tune an electrical-coupling gain
or promote an ablated network.

Registration 553 fixes that electrical-coupling diagnostic before execution.
Only projection 013, the distal-dendritic within-TRN gap-junction record, is
removed in the fresh recognition pair. Chemical recurrent GABA, all afferent
and output paths, the learned coincidence transform, and the effective T scale
remain unchanged. The result can identify whether electrical coupling is
necessary for the peripheral inversion, but it cannot select a conductance or
become a baseline candidate.

Artifacts 553--555 show that projection 013 counteracts rather than generates
the failed global ordering. Its removal changes TRN output from 549/584 to
551/608, increasing the mismatch excess from 35 to 57; 56 of those 57 events
are peripheral. It also adds a fourth mismatch relay event and changes the
nonspecific pair from 4/7 to 4/8. Electrical coupling is therefore not the
source of the inversion. The next fixed diagnostic must remove records 008 and
011 separately, with the other recurrent paths retained, to determine whether
their combined ablation concealed opposing somatic and proximal effects.

Registration 556 fixes a two-arm compartment decomposition before execution.
One arm removes only somatic recurrent-GABA record 008; the other removes only
proximal record 011. Each retains the other chemical path, projection 013 gap
junctions, all afferent/output paths, and the unchanged calibrated pair. The
fixed 549/584 control supplies the reference. Neither arm can be tuned,
promoted, or interpreted as recovered original SMART.

Artifacts 556--558 identify somatic recurrent-GABA projection 008 as the
dominant causal path for the inversion in this reconstruction. Removing 008
reverses the TRN order to 1055/818, whereas removing proximal record 011 leaves
a reduced 566/571 inversion. Both removals reduce the fixed 35-event mismatch
excess, so the preregistered simple-opposition criterion fails; the 2008/2028
dual-ablation outcome instead shows strong nonlinearity. Neither arm is a
candidate: somatic removal changes match relay output to 15 and nonspecific
output to 5/7, while proximal removal produces 22/22. Next audit the unrecovered
KInNeSS `ring=true`/`connectFromMany` semantics behind projection 008 before
considering any discrete source-justified alternative. Do not tune its weight
from these outcomes.

Artifact 559 completes the required source audit. SMART.nml explicitly fixes
record 008's weight 0.3, sigma 2/2, 0.1-ms delay, `ring=true`, and wrapped
border. The archived manual fixes peak-weight scaling and the 0.001 cutoff but
does not define `ring=true`; prior source recovery reached the same limit. The
current center-excluded Gaussian connects each cell to all 80 peers on the 9×9
torus, with no self-edge. No newly recovered evidence authorizes a different
stencil, a fitted radius, or a projection-specific ring meaning. A next screen
may vary only the effective record-008 transfer as explicitly calibrated
recognition behavior, with source scale 1.0 as control and no claim of original
parameter recovery.

Registration 560 opens that calibrated phase with a finite dyadic match-only
screen of projection 008 scales 0.75, 0.875, 0.9375, 0.96875, and the source
control 1.0. Fresh Figure 6 must pass first. Every endpoint is evaluated before
mismatch is consulted; only endpoints preserving the exact five-cell/20-relay
and four-event nonspecific match gates may advance. No endpoint is selected by
TRN count, and no interpolation or source-recovery claim is allowed.

Artifacts 560--562 complete the match-only screen without consulting mismatch.
Only scales 0.875 and 1.0 preserve the exact 20-relay/five-cell/four-event
nonspecific match gates, with 576 and 549 TRN events respectively. Scale 0.75
gives five nonspecific events, 0.9375 gives three, and 0.96875 gives only 17
relay and three nonspecific events. The response is non-monotonic, so no
interpolation is authorized. Register one fresh mismatch for each of exactly
the two survivors and no other scale.

Registration 563 unlocks mismatch for exactly scales 0.875 and 1.0. Their
archived match TRN totals, 576 and 549, are fixed before execution. Each fresh
mismatch must remain overlap-only, have fewer active relay cells than match,
produce fewer TRN events than its paired match, and emit exactly seven
nonspecific events. All gates are conjunctive; no partial ranking, new scale,
or interpolation is allowed.

Artifacts 563--565 close the one-dimensional projection-008 scale family.
Both survivors preserve overlap-only mismatch relay activity and exactly seven
nonspecific events. Scale 0.875 improves the TRN inversion to 576/595 from the
source control's repeated 549/584, but both fail match-greater TRN ordering.
The family is non-monotonic and no interpolation is allowed. A next bounded
calibration may combine only scale 0.75—the registered endpoint with 20 match
relay events, 621 TRN events, and one excess nonspecific event—with lower
nonspecific T scales 0.125, 0.15625, and the 0.1875 control. Match must be
screened before any mismatch trial.

Registration 566 fixes that persistent two-factor match screen. Projection 008
scale 0.75 is crossed with nonspecific T scales 0.125, 0.15625, and 0.1875.
Both dimensions apply during each candidate's fresh Figure 6 learning episode
and subsequent recognition trial. Any Figure 6 failure blocks match. Every
complete match survivor—not a TRN-ranked subset—may advance to separately
registered mismatch. No recognition-only parameter, interpolation, or extra
endpoint is allowed.

Artifacts 566--568 close the persistent scale-0.75/T-current cross. Every
candidate passes all Figure 6 gates and preserves 20 matched relay plus 621 TRN
events, but all three T scales emit only the three startup nonspecific events.
The late fourth event from the recognition-only scale-0.75 run disappears when
the same GABA scale is present during learning. Mismatch remains unconsulted and
the T grid is not extended. The remaining exact recognition-only match
survivor, projection-008 scale 0.875 at T scale 0.1875, now requires one
persistent-consistency test with no new parameter values.

Registration 569 fixes that single consistency check. Projection-008 scale
0.875 and T scale 0.1875 apply throughout fresh Figure 6 and match. Both values
were already selected in prior match-only work; no new parameter is introduced.
A complete pass may authorize one persistent mismatch. Any failure closes the
candidate without adjustment.

Artifacts 569--571 pass the persistent consistency check. With projection 008
scale 0.875 and T scale 0.1875 present throughout learning and recognition,
every Figure 6 gate passes and match yields the exact five active relay cells,
20 relay events, 576 TRN events, and four nonspecific events. Mismatch remains
unconsulted. Register one fixed persistent mismatch with match TRN total 576
locked in advance; any failed gate closes the candidate.

Registration 572 fixes that one mismatch before execution. It repeats fresh
Figure 6 with persistent projection-008 scale 0.875 and T scale 0.1875, then
runs vertical mismatch. The archived match TRN total 576 is fixed. Overlap-only
relay activity, fewer active cells than match, fewer than 576 TRN events, and
seven nonspecific events must all pass; no adjustment or repeat is allowed.

Artifacts 572--574 close the persistent candidate on its single remaining
gate. Fresh Figure 6 passes, mismatch remains overlap-only with three relay
events, and nonspecific output is exactly seven against match's four. Whole-TRN
output is still inverted at 576 match versus 595 mismatch. No adjustment or
repeat is authorized. Before another mechanism is proposed, perform one
unchanged readout-only pair that records the actual summed TRN-to-nonspecific
GABA gate and current. This tests whether raw sheet-wide event count is a valid
proxy for the convergent inhibition named in Figure 7; it cannot alter the
failed gate or promote the candidate.

Registration 575 fixes that unchanged readout repeat. It adds only summed
TRN-to-nonspecific GABA peak/integral, post-startup peak, and inhibitory-current
range to the exact persistent pair. The 20/3 relay, 576/595 TRN, and 4/7
nonspecific totals must repeat. The result can determine whether raw population
event count disagrees with effective convergent inhibition under nonlinear
synaptic timing, but it cannot alter the failed gate or reopen the candidate.

Artifacts 575--577 reproduce every event total exactly but fail the diagnostic
instrumentation contract. All four added fields are null because the runner
omitted `record_relay_diagnostics=true`, which is the existing switch that
constructs the nonspecific-pathway state monitor. The nulls are not interpreted.
One corrected, separately hash-pinned unchanged rerun is required; it must
preserve exact event identity and return finite readouts before any mechanistic
conclusion.

Registration 578 fixes that corrected rerun. The sole execution change is
enabling the existing state monitor; equations, parameters, connectivity,
learned state, stimuli, timing, and score gates remain fixed. Interpretation
requires both exact 20/3 relay, 576/595 TRN, and 4/7 nonspecific identity and
finite values for every registered direct-inhibition readout. The result remains
diagnostic and cannot reopen or promote the closed candidate.

Artifacts 578--580 pass the corrected instrumentation and identity contracts.
Both conditions reach the same bounded GABA peak, but mismatch has the larger
integrated TRN-to-nonspecific gate (1154.93 versus 1110.27 gate-ms), alongside
595 versus 576 whole-TRN events. The direction failure is therefore mechanistic,
not merely a raw-event-count proxy failure. Keep the candidate closed. The next
gate must address the upstream excess of peripheral mismatch TRN activity from a
source-justified mechanism and must preregister fresh learning, both conditions,
and direct-drive readouts together.

Registration 581 first corrects an evidence gap in that localization. Artifact
549 measured afferent gates for nine central TRN cells, not the 72 peripheral
cells that create the inversion. The fixed read-only audit records the same
source-resolved gates and currents for all 81 cells while repeating fresh
learning, match, mismatch, and the direct nonspecific GABA integral. No model or
score changes. Only a finite, exact-identity run may distinguish broad afferent
recruitment from propagation inside recurrent TRN circuitry.

Artifacts 581--583 complete the simulation but not the evidence capture: the
30,806-token all-cell YAML exceeds the result transport limit and loses its
middle. Visible event identity repeats, but missing source arrays are not
reconstructed or interpreted. Registration 584 fixes one identical full-sheet
repeat that changes only stdout serialization. It emits bounded regional sums,
per-cell order counts, and ranked peripheral deltas; all model and diagnostic
state remain fixed.

Artifacts 584--586 pass that bounded repeat and directly confirm the recurrent
localization. Peripheral match excitation exceeds mismatch through relay AMPA
(11882.13 versus 1680.78 gate-ms), layer-6II AMPA (2336.45 versus 505.51), and
layer-6II NMDA (255.46 versus 44.77); no peripheral cell has mismatch-greater
afferent integral. Yet 36 peripheral cells fire more in mismatch and the region
emits 520 versus 492 events. Broad mismatch afferent recruitment is rejected.
The next bounded effective reconstruction may cross the already tested
persistent projection-008 scale 0.75 with modest projection-011 reductions at
fixed T scale 0.1875, match-only first. This calibrates unresolved recurrent
balance and is not source recovery.

Registration 587 fixes that match-only cross. Projection 008 stays at the prior
persistent endpoint 0.75; projection 011 uses only 0.875, 0.9375, and source
control 1.0; nonspecific T scale stays 0.1875. Every value persists through a
fresh Figure 6 episode. Only exact Figure 6 plus 20-relay/five-cell/four-event
nonspecific match survivors may advance. TRN totals cannot rank candidates,
mismatch remains hidden, and no interpolation or grid extension is authorized.

Artifacts 587--589 yield one exact survivor without consulting mismatch.
Projection-011 scale 0.9375 preserves every fresh Figure 6 gate and produces the
five-cell/20-relay/697-TRN/four-nonspecific match. Scale 0.875 fails Figure 6;
source control 1.0 produces only three nonspecific events. The nonlinear grid is
closed without interpolation. Register one fixed fresh-learning match/mismatch
pair at projection-008 scale 0.75, projection-011 scale 0.9375, and T scale
0.1875, with full-sheet and direct-drive diagnostics; any failure closes it.

Registration 590 fixes that one pair before mismatch execution. Both recurrent
scales persist through fresh learning; the archived 697-event match is locked.
Mismatch must remain overlap-only, produce fewer TRN events than match, and
yield exactly seven nonspecific events against match's four. Full-sheet afferent
and direct-inhibition summaries are mandatory. No adjustment or repeat follows
a failed gate.

Artifacts 590--592 repair the former TRN mechanism but close on one remaining
numeric gate. Fresh learning and match repeat; mismatch is overlap-only. Match
now exceeds mismatch in whole TRN events (697 versus 608), central and peripheral
TRN events, and integrated TRN-to-nonspecific inhibition (1189.70 versus
1175.66 gate-ms). Match remains exactly four nonspecific events, but mismatch is
six rather than seven. Do not retune the closed pair. Next compare read-only
nonspecific timing/current traces against the prior seven-event control to
localize the missing event before defining any new parameter family.

Registration 593 fixes that read-only comparison. It repeats the archived
seven-event/wrong-TRN-order control and the six-event/repaired-TRN-order mismatch,
each after its own fresh Figure 6 episode. It records nonspecific local maxima,
detector transitions, source-separated peak currents and gates, current ranges,
and TRN GABA integrals. Exact identity is required before interpretation. The
diagnostic cannot change parameters, reopen either candidate, or promote a
baseline.

Artifacts 593--595 complete the comparison and localize the missing event.
Both independently learned arms pass Figure 6 and exactly repeat their archived
mismatch identities. At the comparable third late detector maximum, dendritic
T-type calcium, direct input, and layer-6II excitation are nearly unchanged,
whereas the repaired arm receives stronger TRN GABA in soma and both dendrites.
It never develops the control arm's fourth late positive detector maximum.
Intrinsic T-current recovery is therefore rejected as the primary difference;
the corrected inhibitory envelope suppresses the seventh event. This diagnostic
selects no parameter and reopens neither candidate. Before any new screen, audit
the source and prior calibration history of projections 047--049 under the new
recurrent balance; do not retune the T current.

Artifact 596 completes that source/history audit. The ModelDB records preserve
three all-to-one GABA contacts with 1/4-ms kinetics and fixed soma/proximal/
distal ratios; the supplement's conflicting distal 1/7-ms tuple already fails
Figure 6. The old common-scale family remains closed for its top-five upstream
model. Artifact 595 supplies new interaction evidence only under the repaired
recurrent balance, so the audit permits exactly its previously selected and
verified 0.75 endpoint—no new midpoint or grid. Registration 597 fixes one
persistent fresh-learning Figure 6-plus-match test at that endpoint. Mismatch
is hidden unless all Figure 6 gates and the exact 20-relay/four-nonspecific
match pass.

Artifacts 597--599 close that sole interaction endpoint before mismatch. Common
scale 0.75 preserves every fresh Figure 6 gate, relay cells 38--42, 20 relay
events, and 697 TRN events, but reduces integrated TRN inhibition enough to add
a fifth nonspecific match event. The official four-event match therefore fails.
Do not interpolate, extend the grid, alter one compartment, or inspect mismatch.
The source-control repaired-TRN pair remains the closest source-constrained
Figure 7 reconstruction: exact pathway, correct TRN/effective-inhibition order,
and exact match, but six rather than seven mismatch events. Further exact-rate
work now requires an independently sourced condition-sensitive mechanism;
otherwise downstream validation must lock and disclose this one-event deficit.

Artifact 600 updates the existing Figure 10 harness to accept the actual
calibrated-model contract: a fresh learned snapshot, persistent projection
scales, the learned top-five comparator, and first-category-event cue clearing.
Legacy defaults remain backward compatible, and targeted tests pass. This is
instrumentation/protocol compatibility, not a model change. Registration 601
now fixes one causal reset pair using the source-control repaired-TRN model.
Both arms share one fresh Figure 6 snapshot; the negative control disconnects
only nonspecific-thalamus-to-layer-5 projections 017/018 at mismatch onset.
Every causal gate is conjunctive, no result-driven tuning is allowed, and the
known six-versus-seven Figure 7 deficit remains explicitly locked.

Artifacts 601--603 execute the fixed causal pair and reveal a deeper but
localized reset failure. Both arms establish the identical five-cell horizontal
layer-4 assembly. In the persistent sequence, nonspecific output is exactly
four events during match and seven during mismatch. Disconnecting projections
017/018 at mismatch reduces layer-5 output from 70 to 55 events, proving that
the nonspecific burst reaches layer 5. Yet layer-6I remains 30 events and
layer-4 remains 43 events in both arms; winner suppression and alternative
release fail. Figure 10 is not reproduced. Next add read-only projection-025
gate/current diagnostics to determine whether the causal difference is erased
in layer-5 transmission or masked at layer 6I. No weight or AHP/ACh tuning is
authorized before that localization.

Artifact 604 adds default-off, read-only layer-6I source separation for the
three serialized inputs: relay projection 023, layer-2/3 projection 024, and
layer-5 projection 025. Registration 605 fixes one unchanged repeat of the
calibrated reset pair and requires complete event identity before interpreting
population-summed mismatch gate/current integrals and peaks. Greater intact
projection-025 drive with identical layer-6I output will localize masking inside
layer 6I; equal drive will localize event/transmitter loss before layer 6I.
No parameter selection or AHP/ACh change is permitted.

Artifacts 605--607 complete that localization with exact event identity.
Projection 025 is not losing the layer-5 difference: intact/control mismatch
gate integrals are 57.65/22.89, current integrals 57,937/21,003 pA·ms, and
current peaks 4,502/503 pA. Relay drive is effectively identical and layer-2/3
drive changes only slightly, yet layer-6I remains 30 events and layer-4 remains
43 in both arms. The reset signal is masked or saturated inside layer 6I, or
its event/resource timing prevents effective downstream transmission. Next
record exact layer-6I event timing, source transmitter, and projection-026
drive to layer-4 inhibitory cells without changing depletion or any weight.

Artifact 608 adds the remaining default-off reset-chain observations: exact
layer-6I events and transmitter, projection-026 drive and layer-4 inhibitory
events, then projection-036 inhibition of layer-4 excitatory cells. Registration
609 fixes one unchanged intact/control repeat and locates the earliest point at
which their difference disappears. The archived layer-6I resource parameters
(epsilon 1, 400-ms recovery) are observed but not altered. No weight, AHP/ACh,
timing, comparator, or score may change.

Artifacts 609--611 complete that trace. The intact arm still carries 2.76 times
the integrated and 8.95 times the peak layer-5 current at layer 6I, but the same
five layer-6I cells emit the same 30 events, with at most 0.03 ms timing drift
and effectively identical transmitter samples. Projection 026, layer-4
inhibitory output, and projection 036 are consequently unchanged. The causal
difference is therefore masked at layer-6I spike generation, not by transmitter
depletion or either downstream synapse. Before any calibration, compare the
layer-6I input balance and fixed-weight semantics against SMART.nml, the
supplement, and the archived KInNeSS manual. Any next executable candidate must
be a bounded source-discrete interpretation and retain every Figure 6 gate.

Artifact 612 audits the layer-2/3-to-layer-6I fixed input against all available
official materials. SMART.nml serializes fixed weight 4 and an inapplicable
plastic baseline of 2, while the paper supplement prints weight density 1 for
the same one-to-one, 1-ms, 2/2-ms AMPA route. The archived manual does not
support using 2 for a nonmodifiable projection, so that tempting midpoint is
rejected. Registration 613 selects only the supplement-literal endpoint by
scaling projection 024 from 4 to 1. It is an official-source-conflict
sensitivity, not a fitted value. Figure 6 is a hard prerequisite and the family
closes after this single endpoint without interpolation or another parameter.

Artifacts 613--615 execute and close that endpoint. All fresh Figure 6 gates
survive, and intact mismatch activity drives 136 rather than 56 layer-5 events.
Projection-025 current reaches layer 6I with a 10.89-fold integral and
56.68-fold peak intact/control contrast. Even so, both arms emit the same five
layer-6I events, all from cell 40, and all downstream reset measures remain
unchanged. Nonspecific mismatch output also shifts from seven to nine events.
The supplement-literal projection-024 value is therefore rejected and the
family closes. The next admissible work returns to the released-network value
and audits spatial layer-5 and layer-6I recruitment, because Figure 10 requires
a nonspecific layer-5 wave and broad layer-6I reset, not merely larger totals.

Artifact 616 adds default-off spatial observations to test that requirement:
mismatch layer-5 events by cell; projection-025 gate/current by layer-6I target;
and layer-6I proximal/somatic voltage peaks by target. Registration 617 returns
to released-network projection-024 weight 4 and fixes one unchanged repetition
of Artifact 610. Exact prior identity and all fresh Figure 6 gates are required
before locating the earliest spatial collapse. This audit cannot select a
parameter or reopen the failed supplement endpoint.

Artifacts 617--619 reproduce every prior identity and localize the spatial
failure. The intact mismatch activates 17 of 81 layer-5 cells versus five in
the disconnected control; each of the twelve added peripheral cells emits only
one event. Projection 025 transfers this support to exactly the same 17
layer-6I targets, but peripheral dendrites peak only around -67 to -61 mV and
somata around -69 to -68 mV. Only the original five winner-aligned layer-6I
cells spike. Thus broad sustained layer-5 recruitment is the first missing
Figure 10 phenomenon; sparse peripheral drive is then filtered below layer-6I
spike threshold. Before changing a model parameter, audit the published 300-ms
panel and approximately 175-ms reset onset against the current 200-ms total
assay duration.

Artifact 620 performs that source audit. Figure 10e spans 0--300 ms, depicts a
reset bar beginning around 175 ms, and shows replacement-winner activity mainly
after 200 ms. The current assay ends at 200 ms total. Registration 621 therefore
selects one source-visible endpoint only: retain the 100-ms winner phase and
extend mismatch from 100 to 200 ms, for 300 ms total. The runner now reports
first-100-ms and late mismatch counts separately. Every Artifact-618 prefix and
fresh Figure 6 gate must repeat before late behavior is interpreted; the family
closes after this endpoint without a duration grid or parameter change.

Artifacts 621--623 complete that endpoint and revise the localization without
recovering reset. Every fresh Figure 6 gate and the complete first-100-ms
mismatch prefix repeat. During the added interval, the intact arm recruits all
81 layer-5 cells and projection 025 delivers finite input to all 81 layer-6I
targets, versus 11 in the disconnected control; integrated projection-025
current is 4.68 times greater intact. Yet both arms emit the same 42 late
layer-6I events from the same seven cells and the same 49 late layer-4 events.
Thus 200 ms was too short to observe the published broad layer-5 wave, but
duration alone does not restore causal winner suppression or alternative
release. Close the duration family. Before changing an intrinsic parameter,
audit whether projection 025's serialized 0.2 `connectFromMany` spread is a
standard deviation or variance: the field name and archived manual conflict,
and the active standard-deviation convention makes this pathway effectively
one-to-one after the documented cutoff.

Artifact 624 completes that audit. SMART.nml fixes projection 025 as wrapped
`connectFromMany`, weight 1, spread 0.2/0.2, and calls the fields
`sigma_x/sigma_y`; the archived KInNeSS manual calls Spread X/Y variances.
With source-peak scaling and the 0.001 cutoff, standard deviation retains only
81 one-to-one contacts, while variance yields a fixed 729-contact wrapped 3x3
stencil with center/axial/diagonal factors 1/0.0821/0.00674. The supplement
omits this archived layer-5 input from its layer-6I table. Registration 625
selects the variance interpretation only for projection 025 as one effective
mixed-source sensitivity, leaving every archived number unchanged. This is not
claimed as recovered original simulator semantics: the simulator-wide variance
candidate already failed Figure 6. Run fresh Figure 6 and one 300-ms causal
pair, then close the family without another spread, weight, or threshold.

Artifacts 625--627 execute and close that family. Every fresh Figure 6 gate and
the complete upstream Figure-10 identity survive. The 3x3 stencil raises intact
projection-025 current integral from 309,566 to 419,180 pA-ms and supplies all
81 layer-6I targets, but both causal arms still emit the same 72 post-mismatch
layer-6I events from the same seven cells and the same 92 layer-4 events.
Nonspiking layer-6I somata remain near -68 mV and proximal dendrites near -67
to -66 mV. There is no winner suppression or additional alternative release.
Reject and close projection-specific variance; missing local convergence is not
sufficient. Next audit the complete layer-6I intrinsic cell, axial transfer,
and projection-025 target-compartment facts against every official source. If
that exposes no discrete source conflict, use a read-only isolated replay of
the measured input envelopes to quantify the excitability gap before defining
any effective calibration family.

Artifact 628 finds complete agreement between SMART.nml and Table 3 for the
layer-6I cell: two compartments, 0.08x0.1-mm soma and 0.05x0.1-mm proximal
dendrite, 80-kOhm-cm edge, -70-mV leaks, 0.15/0.9-mS/cm2 leak densities,
50/30-mS/cm2 somatic Na/K, and no T current or AHP. SMART.nml also explicitly
targets projection 025 to the proximal dendrite. The sole discrete conflict is
axial: the active paper-literal Equation 2 yields 62.83/24.54-nS directional
terms, whereas KInNeSS Equation 9 with the same serialized edge yields
35.30/35.30 nS. Registration 629 selects only the latter for layer 6I, retains
the source-control one-to-one projection 025, and requires fresh Figure 6
before one 300-ms causal pair. This is a mixed source-equation reconstruction,
not an axial scale fit or a recovered global simulator convention.

Artifacts 630--631 execute and close the layer-6I axial family. Every fresh
Figure-6 gate and the upstream Figure-10 identity pass. The KInNeSS axial
endpoint reduces layer-6I post-mismatch output from the paper-axial source
control's 72 events to 40, but the intact and disconnected arms still emit the
same 40 events from the same seven cells. They also produce the same 98
layer-4 events, 80 winner events, and two released alternatives. Projection
025 preserves a 4.63-fold intact/control current-integral difference, while a
common peripheral soma peaks only near -69 mV. Reject the endpoint and close
the family: every source-discrete layer-6I intrinsic, target-compartment,
spread, and axial alternative is now exhausted. Next return to paper axial and
the one-to-one projection-025 source control for a preregistered read-only
selected-cell timing assay before defining any effective calibration family.

Artifacts 632--633 implement and preregister that timing assay. The source
control remains paper-axial with one-to-one projection 025 and the fixed
300-ms protocol. The existing default-off layer-6I state monitor now also
observes the already-defined spike-detector coordinate and reduces cells 0,
31, and 40 to bounded summaries: event times; soma, proximal, and detector
peaks; the signed -20-mV detector-threshold gap; and projection-023, -024, and
-025 gate/current timing at synaptic and voltage peaks. Full traces are not
persisted. Interpret the assay only if every Figure-6 gate and all registered
Artifact-622 event counts repeat exactly. The result may define a later
isolated replay, but cannot select or change a parameter.

Artifacts 634--635 complete the timing localization after one registered run
and one unchanged output-recovery replay. All Figure-6 gates and every
Artifact-622 population count repeat exactly. Intact peripheral cell 0 gets no
projection-023 or -024 input and one projection-025 envelope (541-pA peak,
3,014-pA-ms integral). That pulse raises its proximal peak by 3.00 mV and soma
peak by only 1.43 mV; the soma remains at -68.52 mV, 48.52 mV below the -20-mV
event threshold. Active cells 31 and 40 instead receive dominant projection
024 or 023 drive, and their event counts remain six and eighteen in both
causal arms despite projection-025 differences. Thus the peripheral failure
is insufficient isolated layer-5-to-layer-6I drive, not a small threshold miss
or a local coincidence-timing error. Select no parameter. Next implement and
validate a lossless isolated replay of cell 0 before registering any effective
projection-025 scale bracket or changing the connected network.

Artifacts 636--637 implement and preregister the lossless replay gate. The
unchanged intact network will capture peripheral layer-6I cell 0 at mismatch
onset, all ten intrinsic dynamic variables, and the three integrator-visible
projection gates over the fixed 200-ms mismatch. The isolated cell recomputes
projection currents from its own voltage; no precomputed projection current is
forced. A synthetic replay and a short end-to-end Figure-10 capture/replay both
match every state to less than 1e-12. The full run must first repeat all fresh
Figure-6 gates and Artifact-634 connected-source counts, then reproduce the
empty cell-0 event train and every state within 1e-12. No scale or calibration
is authorized until this identity gate passes.

Artifacts 638--639 pass that gate exactly. Every fresh Figure-6 gate and all
six registered connected-source count pairs repeat. The isolated cell
reproduces the empty source event train and every one of ten dynamic-state
traces with zero numerical error across 20,000 mismatch samples. The replay is
therefore a lossless reduction suitable for isolated projection-025
conductance sensitivity. It has not repaired Figure 10 or selected a value.
Next add an isolated-only maximal-conductance multiplier, require scale 1 to
remain exact, and preregister the finite powers-of-two bracket 1--64 for the
first peripheral layer-6I event. Keep conductance and event-multiplicity
interpretations distinct; do not alter the connected network yet.

Artifacts 640--641 implement and preregister that isolated bracket. The only
varied quantity is the maximal conductance of layer-6I port 002, the proximal
AMPA target of projection 025. The captured receptor gate is not multiplied,
and event multiplicity is unchanged. Scale 1 must remain lossless within
1e-12 before scales 2, 4, 8, 16, 32, and 64 can be interpreted. Report only
the first event-producing scale and its preceding tested endpoint; do not
interpolate, refine, select a connected-network value, or promote a baseline.

Artifacts 642--643 locate the first-event boundary. Scale 1 remains exactly
lossless and every trial is finite. Scales 1, 2, and 4 are silent; scale 8
elicits one event at 108.35 ms into mismatch; scales 16, 32, and 64 yield one,
two, and two events. Thus the effective projection-025 conductance requirement
is coarsely greater than 4 and at most 8 times the reconstructed value. This is
a substantial transfer deficit, not a small timing adjustment, and it is not a
recovered source parameter. Next preregister one connected causal pair at the
first event-producing endpoint, scale 8. Require fresh Figure 6 and exact
upstream identity, then test broad layer-6I recruitment, winner suppression,
and alternative release before any refinement or parameter selection.

Artifact 644 preregisters that sole connected endpoint. Projection 025 is
scaled to 8 in both the intact arm and the mismatch-onset disconnection
control; persistent projection-008/011 and nonspecific-T calibrations remain
fixed. The 300-ms protocol, comparator, event rule, spread, compartments, and
all other parameters are unchanged. Fresh Figure 6 is mandatory. A reset pass
requires the existing pre-reset-winner, reset-chain, winner-suppression, and
alternative-release gates; increased layer-6I activity alone is insufficient.
No refinement, parameter selection, or baseline promotion is authorized.

Artifacts 645--646 execute the endpoint twice, with the unchanged repeat using
only a bounded final serialization filter after the first output exceeded
terminal retention. Both runs agree on the visible counts and all reset gates.
Scale 8 restores the broad intact wave: all 81 layer-5 and layer-6I cells are
active, and layer-6I output rises to 153 versus 76 control events. Projection
026 then carries 2.12 times the integrated and 8.59 times the peak excitation
to layer-4 inhibitory cells. The downstream sign is nevertheless wrong: those
cells emit 218 versus 232 events, projection-036 inhibitory-current magnitude
is 10.2% lower intact, winner events are 78 versus 76, and released
alternatives are zero versus two. Close scale 8 without selection. Before
changing another value, add time-resolved readouts for projections 026 and 036
and the currently omitted direct layer-6I-to-layer-4-excitatory projection 038
around the 208.36-ms broad wave. Do not refine projection 025 yet.

Artifacts 647--648 implement and preregister that read-only balance audit. The
source-backed port map is projection 026 to layer-4 inhibitory port 000,
projection 036 to layer-4 excitatory GABA port 001, and direct projection 038
to layer-4 excitatory AMPA port 003, all on proximal dendrites. The monitor
reduces these currents, layer-4 inhibitory events, and layer-4
winner/alternative events into fixed 10-ms bins after simulation. The focal
bins are 100--110 ms after mismatch, containing the 108.36-ms broad layer-6I
wave, and 110--120 ms immediately after it. Bounded serialization occurs only
after full scoring. Require exact Artifact-645 identity; select no parameter.

Artifacts 649--650 repeat every scale-8 identity and localize the downstream
sign failure. In the focal 100--110-ms bin, projection-026 excitation is 1.54
times control and layer-4 inhibitory events are 5 versus 2, but projection-036
inhibitory-current magnitude is already only 92.9% of control. Projection-038
direct excitation is 1.19 times control, winner events are 7 versus 5, and two
control alternatives disappear. In 110--120 ms, projection 026 rises
25.35-fold and projection 038 rises 18.71-fold, while inhibitory events are 21
versus 26 and projection-036 inhibition is only 76.2% of control. The winner
pauses transiently but no alternative is released. The first measured
wrong-sign stage is projection-036 output, with direct projection 038 opposing
reset. Next audit unresolved projection-036 `ring=true` geometry and obtain
target-resolved 036/038 balance for winner and alternative cells before any
new parameter experiment.

Artifact 651 audits projection 036 before a topology change. SMART.nml fixes a
wrapped `ring=true` Gaussian with sigma 1.5, weight 2, and 80 nonself inputs per
target; the archived manual fixes peak scaling and the 0.001 cutoff but never
defines `ring=true`. Exact legacy geometry remains unrecovered. The active
center-excluded Gaussian has incoming spatial-factor sum 13.07. The existing
parameter-free radial-annulus interpretation uses the serialized sigma as its
peak radius, retains the same 80 edges, and has sum 37.35. It is the sole
discrete alternative authorized for a projection-036-only endpoint; all other
rings must remain unchanged. Implement and preregister one scale-8 causal pair
with fresh Figure 6. Do not tune radius or weight from its result.

Artifacts 652--653 implement and preregister that endpoint. The new runtime
field is default-off and dispatches the radial-annulus kernel only for
projection 036; tests verify that projections 008, 009, and 012 remain under
the center-excluded Gaussian convention. The annulus radius remains exactly
one serialized sigma and neither edge count nor any numerical model parameter
changes. Registration 653 pins one 300-ms scale-8 intact/control pair, fresh
Figure 6, the unchanged four-gate reset score, and bounded layer-4 balance
readouts. Any failed reset gate closes this topology route; a complete pass is
only a mixed-source calibrated candidate pending confirmation and spectral
validation, never recovered original simulator semantics by itself.

Artifacts 654--655 execute and close the endpoint. All fresh Figure-6 gates
pass, and the intact mismatch recruits all 81 layer-6I cells versus 17 in the
control. Projection-026 excitation is 2.08 times control, but layer-4
inhibitory output rises by only two events and aggregate projection-036
inhibitory magnitude is effectively unchanged at 1.0002 times control. Direct
projection-038 excitation rises 2.11-fold. The winner emits 72 versus 64
events intact/control, while both arms release the same two alternatives.
Consequently winner suppression and alternative release both fail. Close the
parameter-free ring-topology route without radius or weight refinement. Next
add target-resolved, read-only projection-036/038 balance for winner and
alternative cells before authorizing any effective numerical family.

Artifacts 656--657 implement and preregister that passive audit. The monitor
reuses the existing layer-4 state recording and reduces projection-036 current,
projection-038 current, and layer-4 output into fixed 10-ms bins for cells 31,
38--42, and 49. The endpoint returns to the source-control center-excluded
Gaussian and must exactly repeat Artifact 649 before spatial interpretation.
No parameter, topology, event rule, protocol, or reset score changes.

Artifacts 658--659 complete the audit after an unchanged bounded-output
recovery repeat. Both executions exactly reproduce every Artifact-649 count.
For winner cells 38--42, intact late projection-036 inhibition is 4.9% weaker
and projection-038 excitation 10.7% stronger than control, with 39 versus 37
events; both measured directions oppose reset. Alternatives 31/49 remain
silent intact despite 44.8% less total projection-036 inhibition than control,
so excessive off-surround inhibition does not explain their silence. Their
projection-038 excitation is also 57.3% lower, but control alternatives already
emit four early events while that current is zero. Therefore neither measured
path alone explains release. Next add the omitted projection-035 relay and
projection-037 recurrent AMPA currents to the same passive target audit before
opening any numerical calibration.

Artifacts 660--661 implement and preregister that complete-port audit. The
existing target monitor now includes relay projection 035 and recurrent
projection 037 alongside projections 036/038. A new bounded output mode stores
all exact 10-ms arrays without repeated row metadata, but only after the full
result and reset score exist. The unchanged source-control model must repeat
every Artifact-658 identity before the first complete-input divergence is
interpreted. No parameter changes.

Artifacts 662--663 complete the unchanged recovery and localize the first
alternative output difference to 70--80 ms. Before that interval, relay drive
and off-surround inhibition are effectively identical across arms, recurrent
projection 037 is zero, and neither alternative fires. In the divergence bin,
each control alternative fires once and projection 037 becomes nonzero, while
both intact alternatives remain silent. The ten-ms integral cannot determine
whether recurrent current starts before or after the first spike, although its
large later control-only total establishes positive-feedback amplification.
Next preregister a read-only 65--85-ms, one-ms timing audit of voltage, events,
and projections 035--038. No parameter is selected.

Artifacts 664--665 implement and preregister that causal-order audit. For cells
31 and 49, it stores one-ms projection-current integrals and somatic-voltage
bounds, exact spike times, and projection-037 onset measured on the native
0.01-ms grid using a fixed 1e-9-pA threshold. One unchanged source-control pair
must repeat every Artifact-662 identity. The result may classify recurrent
excitation as initiation or feedback amplification but cannot authorize or
select a parameter.

Artifacts 666--667 complete the pair with exact identity. Cells 31 and 49 both
spike at 77.33 ms in control, while their projection-037 current first exceeds
the preregistered threshold at 77.45 ms. Recurrent excitation therefore follows
the first event by 0.12 ms: it amplifies release but does not initiate it. The
control somatic action potential is already present in the 76--77-ms voltage
bin. Because shunting current magnitudes depend on membrane voltage, the next
diagnostic must inspect projection-035 and projection-036 gate variables with
native-sample voltage over 74--78 ms before any parameter is opened.

Artifacts 668--669 implement and preregister that gate/voltage audit. It
records projection-035 and projection-036 gate means, the four existing
current summaries, and somatic-voltage bounds for cell 31 in 0.1-ms bins from
74--78 ms. Interpretation begins at the first bin whose arm difference in
maximum voltage reaches 0.1 mV and compares both gate means in that and the
preceding bin. Exact source identity is mandatory and no parameter may be
selected or fitted.

Artifacts 670--671 complete the pair with exact identity. The preregistered
first 0.1-mV difference is present at the left edge of the 74-ms window, so its
required preceding bin is unavailable and that endpoint is formally
left-censored. The retained trajectory still localizes the candidate mechanism:
the intact projection-036 gate surges in 75.4--75.5 ms while control remains at
baseline, and the intact voltage advantage reverses to a control advantage in
75.5--75.6 ms. Projection-035 gate differences remain about 0.001. A native-
sample gate-onset confirmation over a wider window is required before opening
a projection-036 arrival/delay family.

Artifacts 672--673 implement and preregister the native-sample confirmation.
For cell 31 over 73--77 ms, projection-035 and projection-036 gate threshold
times are extracted on the 0.01-ms integration grid at a fixed gate value of
0.1; 0.1-ms voltage summaries are retained. Confirmation requires earlier
intact projection-036 onset before the voltage sign reversal without a matching
directional explanation from projection 035. The result can authorize, but not
execute or select, a bounded arrival/delay family.

Artifacts 674--675 complete the confirmation with exact source identity. Relay
gate 035 is already above threshold at 73 ms in both arms and has no
directional onset difference. Projection 036 crosses 0.1 intact during
75.4--75.5 ms and in control at 75.55 ms, strictly before the voltage advantage
reverses in 75.5--75.6 ms. The console compacted the exact intact native field,
so its supported interval—not an invented timestamp—is archived. The ordering
test passes and authorizes one effective projection-036 delay endpoint at 0.2
ms, one 0.1-ms increment above the serialized delay. It remains unexecuted and
unselected.

Artifacts 676--677 implement and preregister the sole authorized effective-
arrival endpoint: projection 036 delay 0.2 ms, one source-delay increment above
the serialized/default 0.1 ms. The override is projection-specific, recorded in
the result, and default-off. Selection requires fresh Figure-6 gates, preserved
pre-reset winner and reset chain, successful alternative release, and no
worsening of the baseline +2 intact-minus-control winner-event difference. No
delay refinement or interpolation is allowed.

Artifacts 678--679 execute and reject that endpoint. All fresh Figure-6 gates,
the pre-reset assembly, and the upstream reset chain pass, with 81 active
layer-6I cells intact versus five control. Layer-4 inhibitory output is exactly
218 events in both arms, however, and neither arm releases an alternative cell.
The winner emits 79 versus 76 events, worsening the registered intact-minus-
control difference from +2 to +3. The control's two source-delay alternatives
also disappear, demonstrating trajectory sensitivity without causal reset.
Restore the serialized/default 0.1-ms delay and close the family without
refinement, interpolation, another delay, or a condition-specific override.
Original SMART remains unreproduced and unfrozen.

Artifacts 680--681 open a distinct primary-source conflict at projection 026.
The 2008 supplementary table prints a 0.1-ms layer-6I-to-layer-4-inhibitory
delay, while recovered `SMART.nml` prints 1.0 ms; projection 038 is 1.0 ms in
both sources. One Figure-10-only endpoint applies the exact supplementary
0.1-ms value and restores projection 036 to its serialized/default 0.1 ms. A
pass requires all four causal reset gates; a failure closes the conflict with
no interpolation. The passive target trace now includes projection-038 gate
onset at representative alternative cell 31 and winner cell 40. Even a pass is
only a mixed-source candidate until the convention survives fresh learning,
Figure 7, and all holdouts.

Artifacts 682--683 execute and reject the source cross. Fresh Figure 6, the
pre-reset assembly, the reset chain, and broad intact layer-6I recruitment all
pass. The intact winner nevertheless emits 81 events versus 80 control, and
intact releases zero alternatives versus two control. The smaller +1 winner
difference remains the wrong causal sign. Retained 73--78-ms target traces show
projection 038 below the 0.1 gate threshold in control cells 31 and 40, so it
does not initiate their early divergence. Close the p026 delay conflict without
interpolation and retain recovered `SMART.nml` 1.0 ms as executable default.
Next localize the exact layer-4-inhibitory source events that generate the
p036 surge at alternative cell 31 under unchanged source timing.

Artifacts 684--685 implement and preregister that passive audit. For each
projection-036 edge into cell 31, the reducer retains the connected inhibitory
source index, compiled weight, source spike time, and 0.1-ms-delayed arrival in
the fixed 75.2--75.8-ms window. The unchanged source-control pair must exactly
repeat every Artifact-674 population identity before attribution. The result
may distinguish a single-source timing difference from a synchronized or
topology-weighted event set, but it cannot select a parameter or change a reset
gate.

Artifacts 686--687 complete the audit with exact source identity. Both arms
deliver the same five projection-036 sources and the same total compiled weight
to alternative cell 31. Intact arrivals begin 0.15 ms earlier because flank
sources 38 and 42 spike before central source 40; control reverses that order.
The first inhibitory divergence is therefore a source-cell phase difference,
not a topology or edge-weight difference. Keep both delay families closed. The
next permissible step is a passive native-sample audit of every synaptic port,
somatic voltage, and output event for inhibitory cells 38, 40, and 42 over
75.0--75.6 ms; it must select no parameter.

Artifacts 688--689 implement and preregister that source-cell audit. Native
0.01-ms traces retain all four chemical input gates/currents (projections
026--028 and 030), projection-029 gap current, soma/proximal/detector voltage,
Na/K gate state, and emitted events for inhibitory cells 38, 40, and 42 over
75.0--75.6 ms. Interpretation requires exact repetition of Artifact 686 and
must identify a temporally preceding input/state difference without fitting.
The monitor is passive and cannot select a parameter.

Artifacts 690--691 record a successful execution whose 45,150-token console
payload was transport-compacted. Preserved fields pass Figure 6 and repeat the
failed reset score, but required control identities and native samples are
missing; no mechanism is inferred. Artifacts 692--693 preregister an exact
evidence-capture rerun using a focal three-row monitor, one hashed compressed
trace per arm, and compact console metadata. Its additional decision boundary
classifies the 75.0-ms start as left-censored if the focal cells are already on
different action-potential trajectories there.

Artifacts 694--695 complete the recovery pair with exact population and
arrival identity and two verified trace hashes. All focal somata are already
between +30.07 and +43.43 mV at 75.0 ms, so the window begins inside their
action potentials; multiple gates also already differ. The audit is formally
left-censored and cannot identify an initiating input. No parameter family is
opened. Extend the same passive, hashed trace to 73.5--75.6 ms next.

Registration 696 fixes that extension before execution. It changes only the
trace start to 73.5 ms, preserves the same three cells, variables, hashed-file
capture, source-control dynamics, and identity gates, and gives presynaptic gate
ordering priority over voltage-dependent currents. A difference already
present at 73.5 ms remains left-censored and cannot select a parameter.

Artifacts 697--698 complete the exact wider pair. The action-potential rise is
captured, but projection-026 and projection-030 gate contrasts and membrane
differences already exist at 73.5 ms. Projection 027 first differs at 73.85 ms
and projection 028 at 75.40 ms, so neither initiates the inherited split. The
gate cause remains left-censored. Registration 699 therefore avoids further
incremental windows and retains the same passive traces from mismatch onset to
75.6 ms, with an explicit 1e-12 initial-equality and first-divergence rule.
