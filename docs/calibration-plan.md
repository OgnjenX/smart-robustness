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
