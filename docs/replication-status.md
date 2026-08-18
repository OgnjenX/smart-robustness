# Replication status

## What the 2008 model contains

The Methods section describes two hierarchically organized thalamocortical
loops. Each contains a six-layer cortical module, core and matrix thalamic
cells, local inhibitory cells, and TRN; the primary loop additionally contains
a nonspecific thalamic nucleus. Most sheets are 9×9. Cells use the minimum
unbranched compartments and Hodgkin–Huxley-type currents needed for their role.
Synapses use normalized dual exponentials, activity-dependent transmitter
depletion, and gated learning.

## Current milestone: M2 cell-kernel validation

Implemented:

- reproducible YAML configuration and fingerprints;
- a complete machine-readable transcription of the 12 Table 3 cell classes,
  their 30 compartments, geometry, passive properties, and channel densities;
- unit-checked conversion from Table 3 membrane densities to total soma
  capacitances and conductances;
- a Brian2 classic-HH reference population with Traub–Miles-style Na/K gates;
- conductance-based double-exponential AMPA/GABA-like synapses;
- presynaptic resource depletion and recovery;
- a minimal thalamus–cortex–TRN/reset benchmark;
- rate and spectral analysis with predeclared beta/gamma bands;
- unit tests plus an optional Brian2 smoke test;
- all 55 nonblank recovered Supplementary Table 3 connection records as a
  typed, validated catalog: 49 chemical projections, four gap junctions, and
  two external inputs;
- raw and parsed supplementary values, stable record IDs, deterministic
  serialization, and per-record verification status;
- an executable validation-target registry and a source-strength classification
  for the published figure claims;
- an accepted vectorized-multicompartment Brian2 architecture decision.
- executable vectorized dynamics for all 12 Table 3 cell classes and all 30
  compartments, including explicitly selected axial-current conventions;
- source-backed Na/K and T-type calcium equations with printed and inferred
  kinetic alternatives represented as named conventions;
- layer-5 AHP and ACh dual-exponential dynamics with the unreported maximum
  AHP conductance required as an explicit calibration parameter;
- a source-specific Figure 19 kernel assay: both paper and ModelDB profiles
  reproduce deeper AHP after two events and fast ACh suppression. The paper
  80/100-ms profile is within 2 mV of its matched control at 500 ms; the
  archived 80/150-ms, weight-4.5 profile is not and remains a documented source
  discrepancy rather than a tuned-away result;
- a predeclared Figure 8 isolated-relay protocol and qualitative tonic/burst
  scorer.
- recovered KInNeSS 2008 equation semantics, including the `Simple tau`
  function and exact directional axial Equation 7;
- separate paper and executable AHP profiles, plus source-specific Figure 8
  cell geometry and channel densities;
- explicit specific-capacitance selection instead of a hidden default.
- all 12 first-order populations from `SMART.nml` instantiated at their source
  sizes: 812 cells and 1,950 compartments. This intrinsic-only sector uses the
  executable geometry, edge-serialized KInNeSS axial values, population-specific
  reversal potentials, and the network's 5/20-ms layer-5/layer-6II AHP profile.
- structural-count reconciliation: Methods 4.1 plus Table 3 independently imply
  the executable sector's 812 cells and 1,950 compartments; the incompatible
  732/2,106 totals printed in Methods 4.2 are retained as a documented
  publication anomaly rather than used to invent a resize.
- a separate integrity-pinned derived catalog of all 55 executable ModelDB
  projections (51 chemical and four electrical) plus 11 external channels;
- projection-specific receptor ports for all 51 ModelDB chemical records, including
  AMPA/GABA dual-exponential kinetics, NMDA voltage block, and the equal-tau
  alpha-function case; executable XML `g_bar` values are used directly as
  mS/cm² densities as required by KInNeSS Equations 3 and 16;
- the 50 chemical projections whose source and target are both in the
  first-order sector, with exact ModelDB delays, weights, asymptotes,
  one/many/all methods, Gaussian sigmas, wrap/extend borders, and ring flags;
- three in-scope gap-junction records using KInNeSS Equation 8 geometry; one
  cross-area electrical record and two V2 chemical sources are reserved for
  the higher-order loop;
- all ten conductance-based KInNeSS input channels with four 0--255 source
  dimensions and Equation 5 driving-potential semantics;
- source-wide transmitter depletion on layer 5, layer 6II, and layer 6I using
  the serialized recovery/depletion values;
- typed ModelDB learning metadata and the exact piecewise Equation 6
  postsynaptic spike gate, including each record's 20/25-ms depotentiation
  interval; modifiable projections now initialize at their serialized
  asymptotic baseline and retain the separate maximum weight;
- KInNeSS Equations 25/28 adaptive-weight dynamics for the serialized
  presynaptically gated, postsynaptically gated, and dual-AND-gated rules;
  rule-level tests verify potentiation for pre-before-post and depression for
  post-before-pre timing;
- the remaining direct-current external channel using KInNeSS Equation 6
  current-density semantics; its four archived sensitivities are all zero and
  are therefore retained as an inert protocol-controlled input;
- a deterministic 0.1-ms whole-sector integration gate: all 812 cells, 1,950
  compartments, and 53 in-scope projections remain finite at rest, while all
  three in-scope adaptive projections remain exactly at baseline and within
  their serialized bounds;
- a driven Brian2 plasticity gate on the bottom-up relay-to-layer-4 projection:
  the positive Equation 6 lobe potentiates its weight, the negative lobe
  depotentiates it, and both updates remain within the serialized bounds;
- recovered horizontal and vertical 9x9 stimulus grids, each containing five
  centered green=120 pixels; the relay sensitivity of 0.4 reconstructs the
  paper's -12 mV driving potential after applying KInNeSS's leak-relative
  voltage coordinate;
- peak-normalized Gaussian connectivity whose spatial factor remains in [0,1],
  preserving each serialized `weight` as the peak/maximal receptor density;
- CI validation split into independent Brian2 processes: 163 lightweight tests,
  five sector-construction tests, six connectivity tests, and two long AHP
  tests, two protocol tests, plus one whole-sector runtime test currently pass
  (179 total).

Explicitly unresolved source anomalies:

- four supplementary records retain ambiguity flags rather than guessed
  corrections: a literal `N` receptor label, a delay printed as `01`, an NMDA
  record printed with -70 mV reversal, and a plasticity tuple printed in the
  Gaussian-spread row;
- ModelDB backup 112923 has now been recovered and integrity-pinned; its raw
  files are not vendored because the archive has no explicit redistribution license.
- the paper's printed sodium activation coefficient differs by a factor of ten
  from the standard Traub--Miles form, and its calcium steady-state expressions
  are greater than one over physiological voltages unless interpreted as
  reciprocals;
- the Figure 8 caption does not report the hyperpolarizing clamp voltage or
  exact epoch durations, and the legacy Figure 8 XML omits leak and membrane
  capacitance values.

Not yet implemented or validated:

- direct Brian2 instantiation of the supplementary catalog (the executable
  baseline instead uses the distinct, integrity-pinned ModelDB catalog while
  retaining the supplement as an independent audit);
- exact legacy meaning of the KInNeSS `ring` flag (currently retained as a
  center-excluding Gaussian candidate), long-run/protocol-specific learning
  validation, ACh vigilance,
  CSD/LFP geometry, and every published figure protocol;
- quantitative reproduction of match/gamma and mismatch/beta/reset.

The first source-defined horizontal-bar run is retained as a failed
reproduction in `validation-results/first-order-bar-001.yaml`. After a 20-ms
warmup, the five active relay cells fire at 190 Hz rather than 40 Hz and 76
off-pattern relay spikes occur. The immediate cause is a synchronized intrinsic
TRN startup spike near 0.89 ms followed by broad relay rebound. The unreported
legacy gate-initialization convention must be resolved before oscillation
spectra can be interpreted. It is now a required runtime choice:
`steady_state_at_initial_voltage` preserves the prior implementation and
`zero` is a named source-audit alternative. Neither is designated as the
classic baseline until it passes source or behavioral validation.

The corresponding zero-gate run is retained in
`validation-results/first-order-bar-002.yaml`. It removes the immediate TRN
warmup burst, but active relay cells still fire at 120 Hz and the stimulus
recruits 162 TRN spikes. Gate initialization is therefore consequential but
not sufficient to recover the published 40 Hz relay drive.

After correcting XML centimeter geometry, ligand-channel pS conversion, and
leak-relative Na/K voltage coordinates, the prior relay activity disappears;
the exact run is retained as `validation-results/first-order-bar-003.yaml`.
The archived finite-conductance gate now produces no relay spikes, while a
literal -12 mV dendritic clamp produces only one onset spike. This supersedes
the earlier numerical candidates as the active source-consistent failure and
narrows the unresolved discrepancy to legacy membrane/input defaults rather
than network topology or synaptic units.

A typed convention profile, stable configuration fingerprint, reusable
connected-bar runner, and isolated-relay discriminator now replace ad-hoc
runtime probes. The first coarse convention matrix is retained in
`validation-results/isolated-relay-sweep-001.yaml`; none of its finite members
reproduces 40 Hz, and numerically invalid members are explicitly excluded
rather than counted as low-rate candidates.

A later input-semantics discriminator identifies the archived all-zero
`EXTERNAL INHIBITION` channel as behaviorally decisive. Under the literal
framework rule it acts as a large permanent resting leak. Under the explicit
`omit_all_zero` alternative, the isolated relay fires at 40 Hz when the
unreported specific capacitance is 2 uF/cm2 and gates start at steady state;
see `isolated-relay-input-semantics-003.yaml`. The connected five-cell bar does
not yet pass both rate and selectivity: a 20-ms warmup yields five 40-Hz active
relays plus 76 inactive rebound events, while a 100-ms warmup removes inactive
events but yields 30 Hz. This is a strong partial candidate, not a reproduced
baseline; see `first-order-bar-005.yaml`.

The zero-gate branch removes that startup event and yields the first connected
drive profile that passes both gates. At 1.5 uF/cm2, all five active relays fire
at exactly 40 Hz and no inactive relay fires after a 20-ms warmup; the isolated
40-Hz spike-count plateau spans 1.5--1.9 uF/cm2. The result is retained in
`first-order-bar-006.yaml`. It validates the relay-drive scaffold but is not
yet the classic baseline: the capacitance and all-zero-input semantics remain
source-unresolved, and the untrained cortical populations are still silent.

Figure 6 and the executable XML resolve a prior learning-state conflation:
modifiable projections now initialize at serialized `weight`, while
`assymptoticWeight` is retained as Equation 25's uncorrelated baseline. The
learning ODE was also corrected to use correlation-dependent saturation
`Xpre*Xpost*(w_max-w)` plus baseline decay `(w0-w)`, which prevents overshoot of
the declared upper bound. This preserves the selective 40-Hz relay result, but
the cortical populations remain silent, so Figure 6 recruitment is not yet
reproduced.

The chemical-synapse runtime now follows KInNeSS Equations 13--15 at the
individual-connection level: only the last two presynaptic arrivals contribute,
and their normalized gates combine as `g1 + g2 - g1*g2`. The earlier aggregate
dual-exponential state could exceed one (the relay-to-layer-4 diagnostic reached
7.95) and therefore overestimated synaptic conductance. The corrected bounded
implementation still passes the five-cell 40-Hz relay/selectivity gates, while
the cortical populations remain silent.

A subsequent unit audit found that the executable ModelDB path had incorrectly
reused the supplementary table's `pS × millions/cm²` conversion. KInNeSS
Equation 3 instead defines XML channel `g_bar` directly in mS/cm², and Equation
16 multiplies it by the projection weight. Removing that extra `10^-3` factor
raises the center layer-4 dendrite's first-volley maximum from approximately
-64.5 mV to -1.4 mV in an isolated relay→layer-4 assay, while its soma remains
subthreshold over the first 20 ms at the earlier 1.5 μF/cm² candidate. An
instrumented connected run proves finite dynamics through 42 ms: relay activity
recruits TRN, layer 6I, layer-4 interneurons, and matrix cells, but not layer-4
excitatory cells. The prior apparent failure to collect results was execution
cost/output handling, not evidence of numerical instability.

At the KInNeSS source value `CM=1 μF/cm²`, an isolated exact relay→layer-4
pathway produces five bar-aligned layer-4 spikes, whereas the full E/I pathway
does not: relay spikes occur at 5.99 ms and broad layer-4 interneuron spikes
begin at 11.38 ms, suppressing the excitatory winner. A normalized-Gaussian
falsification delays the surround and yields exactly five bar-aligned layer-4
spikes at 18.16--18.33 ms. It is not promoted because Figure 6's before-learning
map and XML weight semantics support peak-scaled weights. The exact legacy
`connectFromMany`/`ring` implementation remains the active topology ambiguity.

Figure 6b's before-learning scale had suggested a normalized-density Gaussian,
but the recovered 2008 KInNeSS User Manual now directly resolves the executable
rule: the serialized `Weight` is the Gaussian peak and resulting weights below
0.001 are omitted. The executable baseline now uses this source-backed finite
`source_peak` convention; the earlier `normalized_density` result is retained
as a superseded audit. Under that earlier convention and source `CM=1 μF/cm²`,
the complete 12-population first-order sector produced
exactly five horizontal-bar-aligned layer-4 excitatory spikes at 17.87--18.04
ms, following the relay volley at 5.75 ms, with no off-bar layer-4 excitatory
spikes. This passes the qualitative selective-recruitment and positive-latency
claim only for the superseded profile (`figure6-recruitment-011.yaml`). It does
not yet reproduce the 100-ms learned weight maps or prove the separate 40-Hz
relay target under the corrected profile. The exact `ring` stencil remains
unresolved.

The corrected peak-scaled, 0.001-cutoff profile has now been run through the
predeclared 100-ms Figure 7 assay. Match and mismatch both produce 20-Hz
nonspecific output, versus the published approximately 40 and 70 Hz; relay
counts are 8 and 6, and both conditions contain 188 TRN events. The source
correction is retained, but it does not reproduce mismatch disinhibition
(`figure7-kinness-gaussian-032.yaml`).

SANNDRA's archived CVS revision history subsequently resolved ionic-gate
initialization. Matching 2004 entries for `gates.h`, `layer.h`, and `unit.cpp`
state that `TGate.init()` resets voltage-gated currents to resting potential.
The executable profile now initializes Na/K and T-type activation and
inactivation variables at their equilibrium occupancy at each compartment's
serialized resting voltage; zero initialization remains only as an audit
alternative. The exact 100-ms Figure 7 rerun gives identical match and mismatch
trajectories: 60-Hz nonspecific output, five relay spikes, 218 TRN spikes, no
layer-4 spikes, and three category spikes in each condition. This resolves an
implementation convention without reproducing the official arousal split and
leaves legacy `ring` geometry as the active topology discriminator
(`figure7-resting-gates-033.yaml`).

Figure 10 reset now has a dedicated causal validation harness instead of being
inferred from Figure 7 rates. One persistent first-order network first receives
a horizontal match and then a vertical-bottom-up/horizontal-top-down mismatch;
an otherwise identical negative control disconnects only the two nonspecific-
thalamus-to-layer-5 channels at mismatch onset. The first exact candidate fails
all reset gates. No layer-4 winner is established during the initial match;
during mismatch, nonspecific thalamus and layer 5 are silent while layer 6I
emits 111 spikes and layer 4 emits 29. The intact and disconnected trajectories
are identical, proving that this late broad activity is not the published
nonspecific-thalamus→layer-5 reset chain (`figure10-reset-034.yaml`).

The archived KInNeSS manual adds that a compartment's actual resting potential
drifts away from its configured leakage equilibrium when voltage-gated channels
are present. A new explicit Figure 7 `equilibration_ms` discriminator therefore
tests an unstimulated 100-ms settling phase without changing model parameters.
It is decisively rejected: both match and mismatch then contain 728 scored TRN
spikes, no relay or layer-4 spikes, and zero nonspecific output. TRN does not
settle to quiescence, so an undocumented warm-up cannot repair the source-backed
profile (`figure7-rest-equilibration-035.yaml`).

A direct visual audit of the paper's Table 3 exposed a previously hidden source
mixture: the connected runtime used complete `SMART.nml` cell specifications,
while its `calcium_density_convention="table3"` label only meant “use the
density already present in that selected cell.” The runtime now has an explicit
`intrinsic_cell_convention` selecting either every recovered executable cell or
every printed Table 3 cell. This matters causally. With otherwise identical
reticular channel kinetics and initialization, one isolated executable TRN cell
fires five autonomous spikes in 30 ms and remains strongly depolarized; the
Table 3 TRN fires none and settles near -66 mV. In the complete match trial,
Table 3 cells restore relay-to-layer-4 and category recruitment, although the
nonspecific cell runs at 160 Hz instead of the published approximate 40 Hz.
That conclusion is superseded by the later correction of reticular calcium
initialization into its absolute runtime coordinate. Under the current
equations the Table 3 TRN emits 18 events in 200 ms, rather than remaining
quiescent. The older assay also mixed Table 3 dimensions/densities with
ModelDB ionic reversals. Whole paper-source selection now consistently uses
the published `E_Na=50`, `E_K=-90`, and `E_Ca=180`; this correction does not by
itself remove the autonomous regime (`intrinsic-cell-source-036.yaml`).

The corrected whole-paper follow-up confirms 29 isolated TRN events in 200 ms
when Table 3 cells and paper ionic reversals are combined with the recovered
reticular kinetics. Combining paper cells with the paper's common thalamic
calcium equations does not rescue the phenotype either: the reciprocal
interpretation fires approximately every 10 ms, while the printed-literal
interpretation becomes non-finite. Thus neither complete official-source
profile is currently a quiescent-yet-recruitable TRN baseline.

Two additional source corrections supersede that assay's mixed initialization.
The recovered reticular calcium equations use absolute membrane voltage, but
their gates were initialized using voltage relative to leak; evaluating the
initial gates in the runtime equation's absolute coordinate reduces an isolated
executable TRN cell from sustained depolarization to one startup event and no
events after 1.02 ms. SMART Equation 8 was also corrected to arm above +30 mV
and emit on the subsequent fall below 0 mV, as printed, rather than emitting on
the upward threshold crossing. The paired Figure 7 rerun now recruits relay,
layer 4, and category cells, but both match and mismatch still yield 70-Hz
nonspecific output, 20 relay events, and one synchronous 81-cell startup TRN
volley with no later TRN events. These corrections are retained, while official
match/mismatch divergence remains failed (`figure7-reticular-init-event-037.yaml`).

The official 100-ms horizontal episode now yields finite activity throughout
all first-order populations and sculpts the winning layer-4 cell's incoming
LGN map horizontally when Equation 25's baseline and upper bound are treated as
projection-level parameters. Horizontal-arm mean weight is 1.063 versus 0.896
on the vertical arm, and the center remains strongest at 3.421. This passes the
qualitative Figure 6b orientation-map claim. The same run does not reproduce
Figure 6c: the central layer-6II category cell spikes only once and its wide and
narrow outgoing relay maps do not acquire positive horizontal contrast. The
artifact `figure6-learning-012.yaml` therefore records Figure 6b as passing and
Figure 6c as failing, rather than claiming the full figure.

A pathway-specific Figure 6 candidate initialized the two corticothalamic
projections at their Gaussian-scaled asymptotic baselines while retaining
projection-level maxima. Figure 6c's own approximately 0--0.3 before-learning
scale now supports this initial state over the roughly 2.1 combined center
weight produced by initializing both pathways at their serialized maxima. The
paper depicts one corticothalamic field, so the scorer now sums the archived
wide and narrow adaptive AMPA maps. This source interpretation is promoted and
documented in `figure6-map-interpretation-017.yaml`; the prior run's combined
horizontal contrast is still only about 0.00079, far below the predeclared 0.01
gate, so Figure 6c learning remains unreproduced.

The exact 100-ms runner completes successfully when retained as a persistent
process; earlier empty yields were mistakenly interpreted as termination. Its
source-level timing identifies the remaining Figure 6c mechanism failure:
layer-6II cell 40 spikes at 58.28 ms, but its 2-ms axonal delay places the LGN
teaching arrival at 60.28 ms, after the matched relay volley at 59.75--59.96
ms. Thus the immediate gamma pair has post-before-pre timing and depresses the
top-down field. The trace and assessment are archived in
`figure6-causal-timing-018.yaml`.

The archived input frame begins at time zero, and the paper specifies a 100-ms
exposure without a warmup. Removing the previously inserted 2-ms unstimulated
period restores causal timing: layer-6II teaching arrives at 60.50 ms and the
matched relay spikes at 61.99 ms. Combined top-down contrast improves to
0.00496 but remains below 0.01. Zero warmup is therefore promoted as the
source-faithful protocol correction, while Figure 6c remains a partial result;
see `figure6-zero-warmup-019.yaml`.

Numerical convergence over 0.005, 0.01, and 0.02 ms preserves all monitored
spike counts, causal ordering, Figure 6b success, and Figure 6c failure; see
`figure6-timestep-convergence-021.yaml`. A separate cross-source contradiction
was made executable: the supplement specifies layer-2/3 excitatory→layer-6II
one-to-one input, whereas SMART.nml points the excitatory channel named
`AMPA 2/3` at layer-4 interneurons with a Gaussian rule. The supplement profile
delays the sole category spike until 90.24 ms and does not improve Figure 6c,
even with a post-stimulus settling interval. It remains an explicit audit
alternative rather than replacing the literal executable profile; see
`figure6-source-resolution-020.yaml`.

Revalidation after the later intrinsic-cell and propagation corrections moves
the active profile's three synchronized relay volleys to 49.53, 65.70, and
81.78 ms. The category cell now spikes at 83.66 ms, so its 2-ms-delayed
teaching signal arrives after the final relay volley and no causal top-down
STDP pair exists. Bottom-up horizontal contrast remains positive (0.08745),
whereas combined top-down contrast is negative (-0.00166). Re-running the
supplement-corrected projection-022 candidate suppresses the category spike
entirely and is again rejected. Equations 25-28 and KInNeSS Table 3 also
confirm that the extra presynaptic factor is intentional, not an implementation
error. See `figure6-category-phase-068.yaml`.

An expanded cortical-chain trace now shows that the active profile stops even
earlier than the delayed teaching event: layer 4 fires five times, but layer
2/3 and layer 5 emit no events. The central layer-2/3 cell receives a strong
projection-032 gate (peak 2.015) and its proximal dendrite reaches -42.32 mV,
while its soma reaches only -62.98 mV. The archived blue=70 layer-6II input is
independently subthreshold. A paper-literal directional axial candidate
recruits layers 2/3 and 5, but over-recruits the network and still gives
negative Figure 6c contrast; KInNeSS Equations 8a-9 support the active symmetric
total edge-current derivation, so that candidate is rejected. The reusable
Figure 6 assessment now requires the ordered layer-4 -> 2/3 -> 5 chain rather
than accepting layer-6II activity alone (`figure6-cortical-chain-069.yaml`).

An exact isolated projection-032 sweep recruits the layer-2/3 soma for every
finite 0.1--2 uF/cm2 capacitance candidate, including the active 1 uF/cm2, so
capacitance does not explain the full-network silence. In the intact network,
five layer-2/3 interneurons fire at 77.28--77.58 ms and projection-031 GABA
peaks at 0.601 after projection-032 AMPA peaks at 2.015. A causal projection-031
knockout restores five layer-2/3 and ten layer-5 events, proving that the
published feedforward inhibitory path vetoes the category chain under current
execution. The knockout is not promotable because projection 031 agrees across
the supplement and archive. A coherent Table 3 intrinsic-cell candidate also
fails both maps and over-recruits layer 6II. The remaining discrepancy is thus
the relative execution/timing of projections 032, 039, and 031
(`figure6-cortical-ei-070.yaml`).

A direct XML ownership audit also rules out a catalog-wide endpoint reversal.
In SMART.nml, the enclosing population/compartment owns the postsynaptic
receptor and `refToPopulation` names its presynaptic source; representative
relay-to-layer-4, layer-4-to-layer-2/3, and interneuron-to-layer-2/3 records all
retain that convention in Brian2. Projection 022 is therefore a localized
ModelDB-versus-supplement inconsistency, not evidence for reversing the whole
network (`modeldb-projection-ownership-071.yaml`).

A projection-031 delay discriminator now excludes Brian2 event scheduling as
the source of that veto. Adding 4 ms—400 integration steps—to the archived
0.1-ms inhibitory delay leaves layer 2/3 and layer 5 silent. Adding 12 ms moves
inhibition beyond the latent layer-2/3 event and restores the ordered sequence:
layer 4 at 73.16 ms, layer 2/3 at 86.81 ms, and layer 5 at 90.57 ms. The altered
delay is not source-supported and is not promoted; it localizes the remaining
failure to inhibitory efficacy versus excitatory somatodendritic recruitment,
not same-step scheduling (`figure6-inhibitory-delay-072.yaml`).

The expanded KInNeSS manuscript further rules out globally saturating the five
interneuron contributions. Equation 15 combines only the last two spikes from
the same presynaptic cell; Equation 16 defines a separate weighted current for
each source. A new executable regression activates two projection-031 sources
and verifies that the postsynaptic gate equals their sum. Distinct-source
summation is therefore retained and cannot be weakened to manufacture Figure 6
recruitment (`kinness-ligand-summation-073.yaml`).

Two additional falsifications leave the single category teaching event
unchanged. Removing layer-6II's archived AHP produces bit-identical spike times
and maps; applying the spike detector in leak-relative coordinates yields only
five relay spikes and zero top-down contrast. Equation 6's positive branch is
already evaluated directly from ongoing voltage above 30 mV, so it is not
truncated to one integration step. These results are retained in
`figure6-ahp-spike-coordinate-022.yaml`; physical-voltage spike detection and
the archived AHP remain active.

A provenance-labeled Figure 6c graphical reference now permits downstream
mechanism assays without pretending that the current network learned it.
Figure 7 at explicit 600- and 1000-pA layer-6II current candidates yields
identical match and mismatch activity: five relay cells fire four times, TRN
fires 264 spikes, and nonspecific thalamus fires six times. Orientation remains
visible only in which five relay cells fire. A zero-expectation control is also
identical, localizing the failure before relay-count modulation and TRN
convergence; see `figure7-stage-localization-023.yaml`. A direct Methods 4.9
clamp interpretation is now executable without changing topology: per-step
operators pin selected relay compartments while every original projection
remains intact. None of the three direct compartment choices reproduces the
same sentence's 40-Hz calibration: soma gives 0 Hz, proximal dendrite 70 Hz,
and distal dendrite 20 Hz. In contrast, the archived green=120 KInNeSS input
gate gives exactly 40 Hz per selected relay. Direct compartment pinning is
therefore retained as a rejected audit interpretation rather than promoted to
the classic protocol; see `figure7-clamp-semantics-024.yaml`.

The lossless global-index partition primitive remains useful for later model
substitution. It maps every original edge into one source-partition × target-
partition block and reconstructs all ModelDB topologies without omissions or
duplicates. Brian2 cannot, however, let multiple presynaptic groups write to
one summed receptor variable, so partitioning is not used by this clamp assay.

The first nonzero-duration V2/pulvinar integration exposed overflow in inactive
plastic synapses rather than in membrane dynamics. Their auxiliary timestamps
had used -1e9-second sentinels; Brian2 eagerly evaluated exponential waveform
and post-spike branches before multiplying them by zero-amplitude or false
conditions. Arrival timestamps now start at zero with zero amplitude,
post-spike timestamps start just outside each serialized learning window, and
exponential elapsed ratios are capped at 100 time constants (an omitted tail
below 4e-44). A warning-as-error regression passes, and the complete isolated
V2 internal circuit remains finite and warning-free for 5 ms. This removes a
numerical initialization defect but is not yet a behavioral validation; see
`full-network-numeric-stability-028.yaml`.

An exact dense C++ standalone backend now makes the complete two-area protocol
tractable without changing equations, parameters, or topology. Its paired
100-ms Figure 7 assay uses the same 600-pA category cue and paper-constrained
Figure 6c expectation in both conditions. Match and mismatch are bit-identical
on all monitored outputs: nonspecific thalamus fires at 60 Hz, V1 layer 4 emits
39 spikes, V1 relay 20, V1 TRN 264, the category population 3, V2 layer 4 35,
and V2 relay 15. Match therefore misses the approximately 40-Hz target;
mismatch lies at the edge of the 70+/-10-Hz range but does not exceed match.
The higher-order loop does not repair the orientation-count failure previously
localized in the first-order assay; see `figure7-full-two-area-029.yaml`.

A pathway-level Figure 7 assay now records both adaptive corticothalamic AMPA
gates, fixed corticothalamic NMDA gates, relay voltages, category spikes, and
the layer-6II/relay inputs to TRN. The installed horizontal expectation reaches
the relay sheet with the intended anisotropy: its combined AMPA peak gate is
1.140 at the center, 0.690 at the outer horizontal arms, and only 0.050 at the
outer vertical arms. Nevertheless, every bottom-up-driven relay emits four
spikes in both conditions, and both conditions produce 264 TRN spikes and the
same six nonspecific spikes. An isolated zero-initialized TRN cell emits three
spikes without input, explaining 243 of the network's 264 TRN events as a
population-wide intrinsic rebound. Repeating the assay with source-peak rather
than normalized Gaussian kernels reduces both conditions equally to ten relay
and 188 TRN spikes; it does not restore divergence. Thus the missing Figure 7
effect is now localized after anisotropic distal feedback delivery but before
relay soma spike-count modulation. Distal-to-soma coupling and event efficacy
are the next source-level discriminators; see
`figure7-pathway-localization-030.yaml`.

Two additional causal controls reject simple strength and threshold fixes.
Multiplying all three archived TRN-to-relay GABA weights by 2 or 4 leaves both
conditions unchanged; a deliberately extreme 100-fold multiplier raises the
combined inhibitory gate from about 0.41 to 41 and reduces every driven relay
from four to three spikes, but still does so identically in match and mismatch.
The contemporaneous KInNeSS framework's printed -20-mV HH event threshold was
also tested as an explicit alternative to SMART Methods' +30 mV. Combined with
the framework's preceding-sample-below-zero rule it emits repeatedly during a
single upstroke, yielding 35,667 TRN events and no nonspecific spikes in either
condition. It is therefore retained as a rejected source contradiction, not a
classic-baseline setting.

Expectation timing is now an explicit Figure 7 protocol variable rather than
an implicit simultaneous onset. Giving the 600-pA layer-6II cue a 10-ms lead
changes relay recruitment: match produces five spikes in each of the five
horizontal relays, while mismatch adds one expectation-driven spike in each of
two horizontal arms to five-spike vertical relay trains. The total remains 25
versus 27 relay spikes, both conditions retain 243 TRN events, and the
nonspecific cell fires at 80 Hz in both. A 30-ms lead moves the intrinsic TRN
burst before sensory onset but exposes widespread relay startup events, again
giving equal 70-Hz nonspecific output. Both zero and steady-state ionic-gate
initialization candidates are non-quiescent when isolated relay and TRN cells
start at the XML resting voltages, and KInNeSS does not serialize the missing
gate states. Initialization is therefore still an unresolved executable
default, not a tuning parameter.

Interpreting only the two layer-6II-to-TRN `ring=true` Gaussian weights as
source peaks (multipliers `2*pi*1.2^2` and `2*pi*1.5^2`) with the 10-ms cue lead
also fails: match and mismatch each produce 25 relay spikes, 247 TRN spikes,
and 70-Hz nonspecific output. The surviving KInNeSS manuscripts define the
Gaussian spread and border modes but not the XML `ring` flag. No invented
annulus stencil is promoted to the baseline; see
`figure7-pathway-localization-030.yaml`.

A later intrinsic-channel audit found that the shared calcium compiler had
incorrectly applied the relay-cell exponent-3 `Simple_Tau` family to TRN. The
XML instead gives V1 and V2 TRN a distinct exponent-2 Destexhe reticular
T-current, and KInNeSS Table 2 supplies the exact `Reticular_Tau` formula.
After correction, an isolated zero-initialized TRN cell emits two events at
10.93 and 13.59 ms rather than three events at 20.27--25.64 ms. In the exact
100-ms Figure 7 assay, TRN output falls from 264 to 162 events, match relay
output is 23 spikes, mismatch relay output 25, and the nonspecific cell fires
at 70 Hz in both. With a 10-ms expectation lead, both nonspecific outputs are
60 Hz. Thus the source correction changes the mechanism substantially but does
not yet reproduce mismatch disinhibition; see
`figure7-reticular-calcium-031.yaml`.

The Figure 7 protocol now keeps its Methods 4.9 somatic-current category cue
separate from the blue category pixel in the recovered Figure 6 training PNG.
Its predeclared scorer independently tests the approximately 40-Hz match and
70-Hz mismatch rates of the nonspecific thalamic cell, with a ±10-Hz tolerance
declared before simulation. Passing this arousal-rate gate will not by itself
be labeled reproduction of the later qualitative beta/reset claim.

The official input archive also resolves a missing protocol channel:
`horizontal0.png` and `vertical0.png` contain blue=70 at the central green=120
pixel. Source-complete routing now drives the central layer-6II category cell
and converges the five green pixels onto nonspecific and matrix input gates.
The relay target still passes, but cortex remains silent; this negative result
is retained in `first-order-bar-008.yaml` rather than attributing Figure 6 to a
relay-only stimulus.

Figure 6a is now independently reproduced at the qualitative level for all
five KInNeSS gating families over -30 to +30 ms. Every curve depresses for a
postsynaptic-before-presynaptic pair, potentiates for the reverse order, has an
extremum within 1--3 ms, and approaches zero in the tails. The exact protocol
and extrema are archived in `figure6-timing-001.yaml`. The presynaptic gating
function is the source's `Xpre`; because Equation 28 also contains the base
`Xpre*Xpost` correlation term, the complete presynaptically gated derivative
contains `Xpre**2*Xpost`. Figure 6b/c network weight-map reproduction remains
pending.

The KInNeSS statement that reported voltage is shifted by the leak potential
also makes the spike detector's coordinate ambiguous in a physical-voltage
implementation. The explicit `relative_to_soma_leak` alternative preserves the
isolated relay's 40-Hz response, but in the connected sector it yields 20 Hz in
each driven relay and 4,617 TRN spikes. This failed discriminator is archived as
`first-order-bar-007.yaml`; `absolute_physical` remains the drive-passing
candidate convention rather than an asserted source fact.

The spike-event temporal expression is now independently executable as three
fingerprinted conventions: a peak latch followed by release below zero, a
literal immediately-previous-sample test, and a threshold-hysteretic latch.
The archived SANNDRA history confirms a dedicated `spikeevents.h` detector and
a 2005 “proper spike detecting” fix, but its source body has not survived. On
the recovered ModelDB TRN at 0.01 ms, literal `V_theta=-20 mV` produces 2,535
events in 300 ms and the hysteretic implementation produces 163 autonomous
events. Literal `V_theta=+30 mV` produces no event, because the voltage does not
fall from above +30 mV to below zero in one step. These alternatives therefore
do not restore source-faithful post-startup TRN recruitment; the +30-mV peak
latch remains the active candidate, not a frozen fact.

The archived KInNeSS Soma editor now resolves a separate ambiguity: chemical
synapses receive a binary signal converted from somatic potential in an
automatically present axon. `monitorSpikes` controls spike-file output, while
the older XML's explicit `axon=true` became unnecessary. The large recovered
TRN dendritic calcium spikes therefore cannot be promoted as projection events;
the Brian2 soma source is source-confirmed (`kinness-axon-source-052.yaml`).
The manual and surviving SANNDRA Doxygen index still do not expose the detector
body, so peak-latch versus literal previous-sample timing remains unresolved.

A third voltage-coordinate candidate now implements the paper-wide fixed
`V_internal=V_physical+67 mV` shift consistently in both neuron events and the
plasticity postsynaptic gate. In the isolated recovered TRN it arms above
-37 mV but never releases below -67 mV: control and measured receptor-driven
conditions both emit zero events despite proximal dendritic peaks above +91 mV.
It is therefore rejected before a full Figure 7 compile and remains only a
fingerprinted audit alternative (`spike-event-shifted67-053.yaml`). The active
absolute-physical candidate is unchanged and is not yet a frozen source fact.

The primary-paper PDF audit also corrects an earlier provenance overstatement:
Methods 4.4 prints the immediately preceding-sample detector and Methods 4.9
reports 100/1000-ms epochs, but no section specifies the integration timestep.
KInNeSS exposes it as a user preference that is absent from `SMART.nml`. A
literal +30-mV detector sweep from 0.005 through 0.05 ms remains finite and
emits zero control or driven TRN events. At 0.1 ms it records one crossing only
after the cell becomes non-finite; 0.2 ms is also non-finite. Coarsening the
step is therefore rejected as a numerical artifact, and 0.01 ms remains an
explicit converged project default rather than an official parameter
(`spike-event-timestep-054.yaml`).

The complete 100-ms Figure 7 test of the hysteretic KInNeSS `V_theta=-20 mV`
alternative confirms the isolated-cell rejection: match and mismatch each
produce 648 TRN events (405 after 40 ms), 90-Hz nonspecific output, and no
relay or layer-4 events. The framework threshold therefore exposes the late
autonomous TRN cycle rather than restoring match-dependent reticular
recruitment (`spike-event-rule-038.yaml`).

A source-backed axial-convention sweep now localizes the remaining reticular
failure more tightly. In every convention, all 81 TRN somata emit one common
startup event; subsequent proximal dendritic calcium spikes reach at least
+71.9 mV while somata remain between -49.10 and -30.29 mV after 5 ms. Match and
mismatch have identical post-startup somatic ranges. The paper-literal axial
form recruits five relay and ten layer-4 events but still produces no later TRN
event. Because `SMART.nml` marks only the soma with `monitorSpikes=true`, a
dendritic event source is not promoted. Axial interpretation alone is therefore
rejected as the missing Figure 7 mechanism; see
`figure7-trn-axial-propagation-040.yaml`.

An additional multiplicative axial audit tests 1x, 2x, 4x, 8x, 16x, and 32x
coupling in independent control and measured-drive TRN cells. Every trace is
finite, but none emits a post-startup soma event. Stronger coupling progressively
reduces the proximal calcium peak, while the control soma peak remains at least
as large as the driven peak at every factor. This closes a simple missing-factor
explanation without fitting a non-source parameter (`trn-axial-scale-066.yaml`).

An isolated receptor-drive matrix further excludes projection strength as the
missing mechanism. The recovered TRN receives constant layer-6II AMPA/NMDA
gates from zero through eight times the largest values observed in Figure 7,
yet produces no post-startup somatic event; its proximal +91.3-mV cycle is
present even at zero drive. With the complete Table 3 TRN, executable Na/K
rates instead produce autonomous trains, whereas the paper's printed Na/K
rates remain silent even under constant gates at 128 times the observed peaks.
Visual inspection of KInNeSS Table 2 confirms the implemented Reticular-tau
formula, and Methods 4.5 confirms the paper Na/K transcription. Thus no tested
official-source family is both quiescent and recruitable, and receptor weights
must not be fitted around this cellular discrepancy; see
`trn-source-family-recruitment-041.yaml`.
The final complete Methods-coordinate check (zero leak reversal, zero membrane
initialization, and printed ionic/synaptic reversals) is likewise silent through
128 times observed drive. Its reciprocal-calcium soma rests above zero, making
the nominal 0-mV excitatory reversal hyperpolarizing rather than recruiting.

The expanded KInNeSS manuscript explicitly identifies the XML
`inpResistance` value as pair-specific axial resistance (R_A), ruling out its
reinterpretation as conductance. A causal 0.01x--10x axial scale sweep still
produces no post-startup TRN soma event, while the known 100x raw-unit error is
numerically stiff at 0.01 ms. High-resolution visual checks of Methods 4.5 also
confirm that the dimensionless `exprel` implementation exactly preserves the
printed Na/K coefficients. Neither axial scaling nor an OCR/transcription
error explains the failure (`trn-axial-source-audit-042.yaml`).

A visual audit of the same manuscript now closes the remaining membrane-area
alternative: both Equation 2's membrane update and Equation 9's axial-density
conversion divide by `pi*D*L`. Intrinsic conductances, capacitance, synaptic
ports, and axial currents already use that cylindrical lateral area throughout
the Brian2 implementation. Adding compartment end caps is therefore rejected
as non-source-backed (`kinness-membrane-area-049.yaml`). A source-motivated
full-cell equilibrium initialization was also tested after recovering the
surviving `TEq_Membrane<T>::equilibrium()` symbol. The recovered TRN has a
zero-input stationary point near -44.16/-42.13/-40.74 mV, but it is unstable:
both control and driven cells immediately enter autonomous bursts. This rules
out complete stationary-point initialization as the missing Figure 7
convention (`trn-full-equilibrium-050.yaml`).

Figure 7 validation now encodes the caption's causal pathway separately from
its approximate output rates. A full reproduction must show more active relay
cells and more TRN events during match than mismatch, in addition to the
approximately 40-Hz versus 70-Hz nonspecific-thalamic rates. This prevents an
accidental rate fit from being mislabeled as the published mechanism. The
current 70/70-Hz candidate fails both pathway-order gates and remains
unreproduced.

The unreported layer-6II current amplitude is no longer a plausible explanation
for the missing match inhibition. In the isolated active-profile category cell,
the archived central blue=70 TD input is subthreshold, 600 pA gives three
events, and 1,000 pA gives four events over 100 ms. The independently calibrated
1,000-pA cue was then run through the complete first-order match network. It
still produces 70-Hz nonspecific output, 20 relay events, and only the common
81-cell TRN startup volley, with no TRN event after 5 ms. The stronger cue is
therefore rejected without fitting a mismatch run
(`figure7-topdown-calibration-051.yaml`).

The source-unreported specific capacitance is also no longer an ordinary
explanation for the missing TRN output. Independent recovered-TRN control and
constant-drive trials spanning 0.2--50 µF/cm² produce no stable post-startup
somatic event; the largest soma peak is 24.52 mV, below the published +30-mV
detector arm threshold, and is not input-selective. The 0.1-µF/cm² condition is
non-finite and is rejected rather than counted as recruitment
(`trn-capacitance-recruitment-056.yaml`).

The Figure 16 source audit corrects an earlier roadmap/target transcription:
the caption sets the V1-layer-2/3-to-V2-layer-4 delay to 10 ms, not 1 ms. The
recovered `SMART.nml` record independently serializes 5 ms. A named protocol
helper now overrides only that projection after archive-faithful network
construction, preserving all other cross-area delays. This passes focused
structural tests but does not yet establish the published inter-area
cross-correlation result (`figure16-delay-protocol-043.yaml`).

The exact Figure 16 Fourier-domain cross-correlation workflow is now available
for all five published bands. It returns raw and energy-normalized curves,
signed lags, band peaks, and the predeclared qualitative lower-frequency versus
gamma comparison. Synthetic delayed and mixed-band controls pass, including a
guard against normalizing roundoff-only spectral energy. This makes the
analysis reproducible but does not yet claim official behavior because
source-faithful LFP current accounting remains pending
(`figure16-cross-correlation-analysis-044.yaml`).

Each compiled compartment now exposes membrane, axial, and Equation-32
transmembrane current observables without feeding those aliases back into its
voltage equation. Equations 31 and 33 are implemented with explicit pA, µm,
mS/cm, µV, and µV/µm² units and pass analytic source/sink and quadratic-CSD
controls. A short complete-network run remains finite with these observables.
This closes current accounting and transform arithmetic, but not the stochastic
54-tip geometry or official LFP behavior (`lfp-current-accounting-045.yaml`).

Methods 4.11's stochastic electrode construction is now executable. The 54
tips span Figure 18's full 1.2-mm cortical sheet at equal intervals; all seven
cortical population classes use their illustrated absolute compartment-centre
depths; the selected cell is sampled uniformly at
10--200 µm lateral distance and every other cell at 10--1000 µm. Euclidean
tip-to-compartment distances feed Equation 31. Because the publication reports
the distributions but not its realized random coordinates, every geometry is
seed-explicit, immutable, and SHA-256 fingerprinted. Parallel aligned cells and
one lateral coordinate per cell are documented reconstruction assumptions, so
this closes reproducible geometry construction without claiming recovery of
the unpublished Figure 16 random draw
(`figure16-electrode-geometry-046.yaml`).
The population-field pipeline also enforces the cell-major current ordering
used by the geometry, preventing Brian's per-compartment monitor arrays from
being silently paired with the wrong Equation 31 distances.
Whole-cortex fields now sum every cortical population on that shared tip axis
and retain the caption's inferior and superior 0.3-mm tip matrices. The source
does not state how those regional tips were reduced to a single Figure 16
curve, so reduction remains an explicit validation convention rather than a
hidden implementation guess.

The two-area network can now attach 1-ms Equation-32 current monitors to all
seven cortical population classes in each area and convert each recovered
NeuroML population into a complete cortical field. This sampling resolves the
caption's 100-Hz upper bound, uses canonical population names to bind the
source-specific cell records to Figure 18 depths, and selects each 9x9 sheet's
central cell for the reported near-electrode distribution. Structural and
synthetic-current integration pass; the 2-s learned-stimulus run is not yet a
behavioral reproduction (`figure16-current-monitor-pipeline-047.yaml`).

An executable Figure 16 candidate runner now builds the complete archived
V1-pulvinar-V2 network, applies only the caption's 10-ms feedforward override,
requires an explicit learned-weight state, presents the learned horizontal bar
for the 1-s prestimulus and 1-s recording intervals, and activates current
monitors only for the recording epoch. A 0.01-ms + 0.01-ms full-network smoke
run completes and yields both 54-tip fields. The official-duration run and its
lower-frequency synchrony criterion remain unverified; the available Figure
6c-shaped weight field is explicitly paper-constrained because the publication
does not provide its learned arrays (`figure16-candidate-runner-048.yaml`).

Figure 16 analysis now binds the caption's exact inter-area regions before any
behavioral run: inferior 0.3 mm of higher-order V2 versus superior 0.3 mm of
lower-order V1. Regional tips are reduced by a declared arithmetic mean; a sum
would produce the same energy-normalized cross-correlation. The integrated
candidate scorer passes anatomical-selection, synthetic-band, and malformed
time-axis controls, but the official-duration behavioral run remains pending
(`figure16-region-reduction-057.yaml`).

The same complete candidate now executes through Brian2 C++ standalone. A
full-network smoke build compiled 1,199 generated translation units, reloaded
one sample for both 54-tip fields, and found all potentials finite. This proves
the practical execution path needed for the 200,000-step official protocol; it
does not substitute the smoke duration for that behavioral run
(`figure16-cpp-standalone-058.yaml`).

The full Figure 16 candidate has now completed the caption's 1-s prestimulus
and 1-s recording epochs. Both 54×1000 fields are finite. The predeclared
regional analysis finds normalized peaks of 0.914, 0.818, 0.678, 0.283, and
0.267 across the five ascending bands; 2--4 Hz is strongest and lower-frequency
coupling exceeds 20--100 Hz. This passes the published qualitative ordering.
It is labeled a reconstructed pass because the learned Figure 6c weights and
original electrode realization are unpublished (`figure16-official-duration-059.yaml`).

The Figure 14 first-order analysis is now executable without reusing the
generic point-model spectrum helper. It records cumulative spikes from all
seven V1 cortical classes for the published 1-s epoch, subtracts the mean, and
uses predeclared contiguous 200-ms Hamming windows. Both the caption's 8--20-Hz
and Methods 4.10's conflicting 8--10-Hz bands are reported. Synthetic controls
and the generated-network monitor path pass; official match/mismatch behavior
has not yet been promoted (`figure14-analysis-runner-060.yaml`).

Both official-duration Figure 14 conditions have now completed. Match peaks at
55 Hz with gamma power 17.23; mismatch peaks at 15 Hz and gamma falls to 12.40,
a 28.1% reduction. A visual audit confirms that panels (a)/(b) assert spectral
peak location while panels (c)/(d) plot band-limited time-domain components;
an integrated-power comparison across unequal-width bands is not the published
gate. The reconstructed candidate therefore passes Figure 14's qualitative
spectral directions. It does not pass the mechanism: both conditions retain
only the identical 81-event TRN startup volley and 7-Hz nonspecific output
(`figure14-official-duration-061.yaml`).

The Figure 10 reset assay has also been rerun under the current conventions and
1,000-pA cue. The horizontal five-cell layer-4 assembly is now scored as one
pre-reset representation rather than collapsed to an argmax cell. During
mismatch, nonspecific thalamus and layer 5 emit no events; intact and
nonspecific→layer-5-disconnected controls have identical 33-event layer-4 and
179-event layer-6I trajectories. Broad activity is therefore not a causal
reset, and the current Figure 10 gate remains failed
(`figure10-current-reset-062.yaml`).

An isolated source-cell assay now localizes a second Figure 10 failure. Holding
the recovered nonspecific AMPA/NMDA peak gates on the layer-5 distal apical
compartment generates a local +41.57-mV action potential but no somatic event;
the soma reaches only -66.86 mV. Sixteen times the archived drive is needed for
one output event. A 2x axial diagnostic also yields one soma event, but it is
source-unreported, non-monotonic at larger factors, and cannot repair the full
trial's independently silent mismatch nonspecific thalamus. Neither change is
promoted (`figure10-layer5-propagation-067.yaml`).

An isolated nonspecific-thalamic recovery assay also rules out the Figure 10
runner's zero-gap phase switch as a sufficient explanation. Both a 120-valued
green source and the exact five-source convergent value of 600 produce an
initial event train but no second event after 0, 20, or 100 ms without drive.
An unreported recovery gap is therefore not promoted; the silent mismatch
response remains an intrinsic thalamic-runtime discrepancy shared with Figures
7 and 8 (`figure10-nonspecific-recovery-076.yaml`).

Figure 15 local synchrony now has a source-bound runner and scorer. The
predeclared pair is adjacent stimulated layer-4 cells 39 and 40; their binned
spike trains are linearly cross-correlated, the caption's ±180-ms range is
retained for display, and the complete cross-correlogram spectrum is tested
against 44±5 Hz. A synthetic 44-Hz pair is recovered at 44.02 Hz. The
unpublished pair choice and confidence-limit
method remain explicit reconstruction gaps (`figure15-analysis-runner-063.yaml`).

The full Figure 15 candidate now passes for that predeclared pair. Cells 39 and
40 emit 9 and 10 spikes, respectively, and the complete linear-correlogram
spectrum peaks at 48.52 Hz, within the declared 44±5-Hz gate. No post-result
neighboring-pair search or network retuning is used. The earlier 27.70-Hz
failure came from cropping the correlogram to its plotted ±180-ms range before
the periodogram, thereby imposing an undocumented lag window
(`figure15-official-duration-064.yaml`).

A visual source audit also corrected Equation 33: its printed denominator is
Δx, despite prose describing a second derivative. The classic path now uses
that paper-literal uV/µm transform. A separately named Δx² implementation is
available only as an alternate robustness convention.

A separate coordinate audit tested whether KInNeSS integrates voltage at zero
relative to each compartment's serialized leak and adds the leak only when
writing physical voltage. Membrane initialization and the voltage supplied to
absolute-voltage T-current kinetics are now independent fingerprinted
conventions. A wholly internal-zero interpretation makes isolated TRN
quiescent, but its 100-ms Figure 7 trials are identical and silent downstream:
25 relay events, no TRN, nonspecific, or layer-4 events in either condition. A
split interpretation, with physical voltage supplied only to T-current gates,
also fails at network scale: over 30 ms both conditions have 10 relay events,
no TRN or layer-4 events, and seven nonspecific events. The active candidate
therefore remains the physical-membrane implementation; the KInNeSS footnote
is treated as an output transformation rather than a wholesale current-axis
shift (`voltage-coordinate-039.yaml`).

The first Figure 8 candidate (67 mV-shifted standard Traub--Miles rates,
reciprocal T gates, Table 3 calcium density, and a -80 mV pre-pulse clamp)
produces sustained tonic trains in both conditions. It therefore fails the
published transient-burst signature and is retained as a negative validation
result, not as a reproduction.

That candidate predates recovery of the dedicated `Ca_rebound.xml` model and
is now classified as a useful failed paper-only reconstruction. The recovered
file resolves the T-gate equation roles and supplies a different Figure 8 cell;
validation of that executable-source profile is the active M2 task.

The latest source-specific candidate now reproduces the transient
hyperpolarized burst signature, but not the depolarized tonic train. This
improvement followed correction of T-gate initialization into the absolute
voltage coordinate. It remains a partial result and is not promoted to a
reproduction claim.

A predeclared 20-member matrix over the two values absent from the archived
Figure 8 XML (five leak densities by four specific capacitances) confirms that
this is not resolved by ordinary passive defaults. Every depolarized condition
emits one onset event; 0.5 and 1 uF/cm2 retain the two-event transient burst,
whereas 1.5 and 2 uF/cm2 lose it. A legacy-version geometry discriminator also
fails with the archived T-current. Removing that current restores tonic firing
only as a causal diagnostic, not as a source-faithful candidate. The remaining
high-value discriminator is the caption's simultaneous hyperpolarizing-clamp
semantics (`figure8-source-defaults-065.yaml`).

That discriminator now has a named executable implementation matching the
contemporaneous KInNeSS voltage-input definition. The top condition freely
equilibrates without a clamp; the bottom condition retains a finite
hyperpolarizing conductance throughout the pulse. Conductances from 1 through
300 nS produce no pulse-evoked events in either condition. Clamp timing is
therefore ruled out as a sufficient explanation. The unresolved Figure 8 gap
is now localized to a missing version-1 `libkinmaze` membrane, current-unit, or
gate default rather than the better documented KInNeSS 1.2 network semantics.

The missing depolarized holding potential is now independently excluded over
-62.3 through -45 mV: every source-cell trial produces exactly one onset event
and no tonic train. The version-1 `Ca_rebound.xml` geometry ambiguity is also
now a first-class executable convention rather than an ad-hoc calculation;
both centimetre-schema and millimetre-schema interpretations can be selected
and reported by the Figure 8 runner (`figure8-depolarized-hold-074.yaml`).

Executing the newly explicit millimetre-schema candidate through the public
runner reproduces its negative discriminator: one 0.68-ms onset event in the
depolarized condition and one delayed 44.6-ms event in the hyperpolarized
condition. It fails both tonic and burst gates, so the centimetre candidate
remains the better partial reconstruction (`figure8-millimeter-runtime-075.yaml`).

Accordingly, output from M2 is a source/audit benchmark, not evidence that
the 2008 results have already been replicated.

The `connectFromAll` external-input translation has been corrected to sum
independent conductance currents rather than treating five green=120 pixels as
one out-of-domain value of 600. This fixes a direct conflict with KInNeSS
Equations 5-6 and changes bar-driven network trajectories. Earlier dynamic
Figure 7/10/14/15/16 results that used the scalar-sum translation are therefore
superseded and must be rerun; structural and unrelated isolated validations
remain valid. See `connectfromall-external-current-080.yaml`.

Official-duration reruns under the corrected fingerprint now establish the
current failure precisely. Figure 7 gives 60 Hz nonspecific firing in both
match and mismatch, with identical reported population counts, rather than the
published approximately 40/70 Hz split
(`figure7-corrected-input-official-081.yaml`). The persistent Figure 10 assay
now forms a five-cell pre-mismatch layer-4 winner, but nonspecific thalamus and
layer 5 are silent during mismatch; disconnecting projections 017/018 changes
nothing (`figure10-corrected-input-reset-082.yaml`). These are failed official
reproductions, not baseline validation passes.

The published Equation 2 axial profile now passes the isolated Figure 10b
apical-to-soma propagation gate without a fitted scale: it produces two
somatic events at source-strength sustained nonspecific drive, whereas the
later KInNeSS Equation 9 profile produces only a distal spike. In a 30-ms
first-order discriminator, Equation 2 also restores early layer-4 recruitment
but does not separate match from mismatch. It therefore remains a
source-backed candidate, not yet the frozen classic convention; see
`figure10-axial-source-discriminator-079.yaml`.

## Validation gates

A milestone may be marked complete only when:

1. each equation and parameter has a provenance status;
2. deterministic tests pass and stochastic results reproduce across declared seeds;
3. target figure protocols and readouts are documented before tuning;
4. expected and negative-control outcomes are both reported;
5. generated data and summaries include the exact configuration fingerprint.
