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
