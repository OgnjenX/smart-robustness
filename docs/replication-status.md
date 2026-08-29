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

The Figure 8 readout itself has now been corrected. Figure 8 plots somatic
membrane voltage, whereas the earlier scorer counted Equation-8 axonal release
events. The literal source cell's apparent one-event tonic/two-event burst
actually contains 137 and 109 leak-relative voltage peaks, respectively, and
therefore reproduces neither plotted mode. The corrected source-unit sweep
finds that 0.2 mS/cm² calcium yields a plausible four-peak tonic trace at
48.13, 112.96, 195.31, and 288.55 ms, but its hyperpolarized trial has 15
prolonged peaks. Current-unit and sub-nS finite-clamp grids also produce no
complete survivor (`figure8-voltage-observable-audit-124.yaml`). Artifact 118
is retained as raw evidence but marked superseded for its wrong observable.

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

A same-network Figure 6-to-Figure 7 protocol now preserves the complete
training trajectory instead of copying learned weights into a cold network.
It retains membrane voltages, intrinsic gates, receptor and transmitter state,
AHP state, and adaptive weights, then disables further plasticity for the
recognition trial. Both match and mismatch nevertheless produce zero
nonspecific and TRN events, all 81 relay cells active, and three category
events. Loss of transient state at the learning/recognition boundary is thus
excluded as a sufficient explanation; the unresolved Figure 6c learning and
condition-dependent relay/TRN recruitment remain upstream gates
(`figure7-same-network-figure6-state-089.yaml`).

The executable depletion path now latches transmitter availability at the
source spike and carries that amplitude through the axonal delay. Previously,
the delayed Brian pathway resampled the continuously recovering transmitter
on arrival, incorrectly coupling delay length to release strength. The
source-corrected Figure 6 run remains a partial result: layer 4 fires at 74.09
ms, the category cell at 82.57 ms, and its 2-ms-delayed teaching event still
follows the final 81.78-ms relay volley. Figure 6b retains positive horizontal
contrast (0.07390), while Figure 6c remains negative (-0.00609), with no
layer-2/3 or layer-5 events (`figure6-emission-time-depletion-090.yaml`).

Re-running the paper-literal Equation 2 axial profile after that correction
creates a causal category-to-relay learning pair: the first category event at
4.52 ms arrives at 6.52 ms and is followed by a relay event at 26.13 ms.
However, the profile broadly over-recruits layer 6II (36 events), still emits
no layer-2/3 or layer-5 events, and produces only 0.00088 top-down horizontal
contrast. It therefore fails Figure 6c and remains an explicit paper-source
discriminator rather than replacing the KInNeSS executable profile
(`figure6-paper-axial-emission-time-091.yaml`).

Official Figure 7 and Figure 10 assays have now been regenerated under the
same emission-time transmitter semantics. Figure 7 changes from the previous
60/60-Hz result to 70/70 Hz; mismatch activates seven relay cells versus five
during match, while both conditions retain the identical 81-event TRN startup
volley. Figure 10 still establishes a five-cell layer-4 winner, but mismatch
has no nonspecific or layer-5 event and is bit-identical to the reset-pathway
disconnection control. These remain failed official reproductions, not
baseline passes (`figure7-emission-time-delay-092.yaml` and
`figure10-emission-time-delay-093.yaml`).

The published-duration Figure 14 rerun also reverses the earlier provisional
spectral pass. Under emission-time transmitter sampling, both conditions peak
at 15 Hz; mismatch gamma power (12.6895) exceeds match gamma power (10.5785).
The match-gamma and mismatch-gamma-reduction gates therefore both fail.
Artifact 083 is superseded for baseline assessment by
`figure14-emission-time-delay-094.yaml`.

Figure 15 is likewise superseded: the same predeclared adjacent layer-4 pair
now has 13 events per cell but a 27.01-Hz gamma-range peak, outside the
44+/-5-Hz acceptance interval. Artifact 084's approximate pass is replaced by
the failed source-corrected result in
`figure15-emission-time-delay-095.yaml`.

The full two-area Figure 16 rerun remains a qualitative pass. Its 1,000-ms
prestimulus and 1,000-ms recording epochs complete with finite 54-by-1,000
fields in each area; the strongest normalized inter-area correlation is in
the 2--4-Hz band (0.9617), above the 20--100-Hz band (0.2399). This result is
still underidentified by unavailable learned arrays and electrode placements,
and cannot offset the failed first-order learning, arousal, reset, and gamma
gates (`figure16-emission-time-delay-096.yaml`).

Those emission-time results are retained as an audit branch but are no longer
the current baseline evidence. A direct rereading of KInNeSS Equation 16 shows
that transmitter availability multiplies the ongoing ligand gate at each
time, rather than being sampled into an event amplitude at emission or
arrival. The runtime now implements `w * g_ij(t) * z_j(t)` while Equation 17
evolves source-wide `z_j`. Dynamic tests verify delayed unit-amplitude arrivals
and continuous 0.25 resource scaling. Artifacts 090--096 are explicitly
superseded by `transmitter-continuous-gating-097.yaml`; all affected official
protocols require one further rerun before baseline assessment.

The corrected Figure 6 rerun is now current (`figure6-continuous-transmitter-098.yaml`).
The KInNeSS axial profile retains a positive bottom-up horizontal contrast
(0.08745), but has no relay event after the 85.65-ms category teaching arrival,
no layer-2/3 or layer-5 event, and a negative combined top-down contrast
(-0.00479). The paper-literal axial discriminator also lacks the cortical chain
and now has no causal corticothalamic pair. Figure 6b therefore remains a
qualitative pass, while Figure 6c and the complete Figure 6 gate remain failed.

The corrected official Figure 7 standalone rerun is also current
(`figure7-continuous-transmitter-099.yaml`). Match and mismatch are identical:
both produce 60-Hz nonspecific output, 18 relay events across five cells, 81
TRN events, five layer-4 events, four category events, and 118 aggregate V1
cortical events. The published match/mismatch arousal and pathway divergence
therefore remains unreproduced.

The corrected Figure 10 intact/control pair remains a failed causal reproduction
(`figure10-continuous-transmitter-100.yaml`). Both conditions establish the same
five-cell pre-mismatch layer-4 winner, then produce identical mismatch
trajectories: 33 layer-4 events, no nonspecific-thalamic events, no layer-5
events, and 179 layer-6I events. Disconnecting the two published
nonspecific-thalamus-to-layer-5 reset inputs changes nothing, so the reset
chain, winner suppression, and alternative-release gates all fail.

The corrected one-second Figure 14 spectra provide a partial result
(`figure14-continuous-transmitter-101.yaml`). Match now has a gamma-dominant
55-Hz peak, but mismatch has the same 55-Hz peak and slightly greater gamma
power (20.1054 versus 19.2410). Thus the match-gamma gate passes while the
mismatch-lower-frequency and mismatch-gamma-reduction gates fail.

The corrected Figure 15 predeclared adjacent layer-4 pair has 10 and 11 events,
but its correlogram spectrum peaks at 59.03 Hz rather than the reported
approximately 44 Hz (`figure15-continuous-transmitter-102.yaml`). The
sufficient-spike gate passes and the numeric target gate fails; no pair search
or post-hoc frequency selection was used.

The corrected full-duration Figure 16 two-area run completes with finite
54-by-1000 fields in V1 and V2 (`figure16-continuous-transmitter-103.yaml`). Its
strongest normalized inter-area correlation is in 4--8 Hz (0.6438), above the
20--100-Hz band (0.3014), so the qualitative lower-frequency-dominance gate
passes. Missing original learned arrays, electrode locations, and regional
reduction details still prevent an exact numeric-reproduction claim.

Section 4.9's description of Spread X/Y as Gaussian variances conflicts with
the archive's `sigma_x`/`sigma_y` names. A new fingerprinted variance
interpretation broadens the kernels and creates a causal category-to-relay pair,
but layer 2/3 and layer 5 remain silent and Figure 6c contrast is only 0.0057,
below the 0.01 gate (`figure6-gaussian-variance-104.yaml`). It is retained as an
audit alternative and is not promoted over the archived sigma interpretation.

A contemporaneous SANNDRA report identifies 0.05 ms as an acceptable CPU HH
timestep, although SMART itself reports none. Repeating Figure 6 at 0.05 ms
under the current equations leaves the same 15/5/1 relay/layer-4/category event
regime, no layer-2/3 or layer-5 events, no causal teaching pair, and negative
Figure 6c contrast (`figure6-continuous-timestep-105.yaml`). The high-precision
0.01-ms baseline is retained.

KInNeSS Equation 17's spike term has also been checked against its accompanying
prose (`transmitter-depletion-jump-106.yaml`). The text explicitly says that a
spike depletes the resource by `-epsilon*z`, confirming the implemented discrete
jump `z <- z*(1-epsilon)`. In particular, SMART's layer-6I value `epsilon=1`
fully depletes transmitter; the alternative impulse solution
`z <- z*exp(-epsilon)` is not the stated simulator update. A Brian2 event-level
regression test now protects this invariant. Because the runtime was already
correct, no figure result changed and artifacts 098--105 remain current.

The original 2008 download page has now been recovered directly and hash-pinned
(`legacy-snapshot-recovery-107.yaml`). It identifies the exact source artifacts
as `KInNeSS-0.3.4-RC2.tar.gz` / `KINNESS_0_3_4_RC2` and
`SANNDRA-1.2.0-RC3.tar.gz` / `SANNDRA_1_2_0_RC3`. Exact-URL checks against the
Internet Archive, the contemporaneous Common Crawl index, and GitHub code
search still recover no tarball or `spikeevents.h` body. This narrows the
remaining source gap but does not justify changing the active falling-phase
Equation 8 detector or any Figure 6 trajectory.

A reusable full-sector current-balance diagnostic now records the exact first
layer-2/3 bottleneck (`figure6-l23-current-balance-108.yaml`). Projection 032's
gate peaks at 76.19 ms, within 0.03 ms of its source-event/delay/kinetics
prediction; projection 031 peaks at 79.69 ms, within 0.04 ms of its independent
prediction. Brian2 scheduling and dual-exponential execution are therefore not
the discrepancy. The excitatory current reaches +1.36 nA in the proximal
dendrite, but axial current into the soma peaks at only +57 pA; the later GABA
current reaches -2.96 nA and the soma remains below -62.97 mV. Halving the
unserialized specific capacitance still produces no layer-2/3 event, while
0.1 uF/cm2 makes the full network non-finite and also fails recruitment. No
capacitance or synaptic correction is promoted.

The Figure 7 diagnostic now follows the causal pathway through the nonspecific
cell itself (`figure7-current-balance-109.yaml`). Match and mismatch activate
horizontal versus vertical relay positions, respectively, but each has 18
events in five relay cells. All 81 TRN cells emit only once, together at 1.34
ms—before the first category event at 3.06 ms and the first driven relay event
at 49.15 ms—and never respond afterward. The resulting TRN GABA gate, its
169.7246 gate-ms integral, layer-6II AMPA/NMDA drive, direct image current,
nonspecific voltages, and six output events are numerically identical between
conditions. This rules out the nonspecific-cell current balance as the first
discrepancy: the missing source-level operation is still matched-relay
recruitment of post-startup TRN output. No unsupported correction is promoted.

A second reusable decomposition resolves the TRN driven-window current balance
itself (`figure7-trn-current-balance-110.yaml`). Relay AMPA is present and soma
axial inward current reaches 3.282 nA, while recurrent GABA is only about 3.13
pA; recurrent inhibition and absent dendrite-to-soma transfer are therefore
excluded as primary explanations. Instead, the executable-source intrinsic
currents dominate: somatic calcium, sodium, and potassium reach approximately
+20.08, +47.49, and -74.58 nA. Both conditions peak near +7.10 mV and never arm
the +30-mV event detector. This also makes the official source conflict
material: SMART.nml includes 100 mS/cm2 somatic reticular calcium, while Table 3
omits it and specifies 10 mS/cm2 only in each dendrite. Prior complete
paper-profile discriminators still fail, so no hybrid conductance edit is
promoted.

The earliest Figure 7 failure is now quantified directly at the relay
(`figure7-relay-current-balance-111.yaml`). As the caption requires,
expectation-only cells remain inhibited. The failure is instead that all five
bottom-up cells keep firing during mismatch, rather than only the single
bottom-up/top-down overlap cell. Directly driven cells receive approximately
0.65 nA, while expectation-only cells receive at most tens of picoamps and
remain below about -54.5 mV. Scaling the horizontal learned field by all of its
shape-preserving 1.2-fold headroom, until its center reaches the serialized 1.5
maximum in both adaptive projections, still leaves five active relay cells in
each condition (20 match versus 18 mismatch events), 81 startup-only TRN
events, and 60-Hz nonspecific output. No unsupported weight or inhibition
tuning is promoted.

The layer-6II-to-TRN off-surround is now protected by projection-specific
topology tests and a measured-delivery audit
(`figure7-off-surround-topology-112.yaml`). With the provisional
center-excluding interpretation required by the otherwise parameter-free
`ring=true` records, projections 009/012 omit overlap cell 40 and deliver their
largest NMDA/AMPA factors to nearby nonoverlap TRN cells. Runtime gates and
currents preserve that ordering: center current is zero, near nonoverlap cells
receive up to about 158 pA, and farther cells up to about 82 pA. Nevertheless,
all TRN output remains the 81-cell startup volley with no later event. Thus
missing connectivity, accidental center inclusion, and failed ligand delivery
are excluded; exact legacy ring geometry remains unverified but cannot explain
the already demonstrated failure under constant measured drive through 128x.

The last documented Gaussian-spread ambiguity has also been carried through a
complete Figure 7 pair (`figure7-gaussian-variance-113.yaml`). Interpreting all
serialized Spread X/Y values as variances broadens the whole network, including
the corticoreticular rings, but match and mismatch still activate all five of
their bottom-up relay cells, each with 20 events. Both retain 81 startup-only
TRN events and 60-Hz nonspecific output. This source-supported alternative does
not restore the caption's overlap-only mismatch response and is not promoted.

The constrained calibration branch now executes the cellular ambiguity space
as deterministic, fingerprinted candidates. The initial Stage A TRN matrix
evaluated 96 combinations of cell source, calcium source, axial equation,
membrane initialization, and event rule. It produced no candidate that was
both quiescent without drive and recruitable by the largest measured
layer-6II AMPA/NMDA gates (`calibration-stage-a-trn-114.yaml`). Following the
predeclared no-survivor rule, the contract was revised to include the material
official-source conflict between the paper's printed Na/K equations and the
archived ModelDB/Traub–Miles rates. The complete revised 192-candidate matrix
also produced zero survivors (`calibration-stage-a-trn-115.yaml`). All 192
control and driven trials were finite; 158 candidates had a quiescent control,
but every such candidate remained unrecruitable. The remaining candidates
produced autonomous control events and were rejected even when drive increased
their event count. No synaptic weight, conductance, threshold, or holdout
result was exposed to rescue the gate. Network calibration therefore remains
causally gated on resolving this cellular source discrepancy.

The next contract revision exposed the article's separate calcium-density
conflict rather than fitting a new value: cell-specific densities from Table 3
and ModelDB versus the Methods 4.6 global 250 mS/cm² statement. The 192 newly
exposed global-density candidates were retained as an incremental matrix
(`calibration-stage-a-trn-116.yaml`). All trials were finite; 174 candidates
were quiescent but also unrecruitable, while the other 18 were autonomous and
remained active with drive. Zero were promoted. Together with the preceding
cell-specific matrix, all 384 registered Stage A TRN combinations fail the
same causal gate.

The remaining isolated gates have now been regenerated together under the
current calibration fingerprint (`calibration-stage-a-remaining-117.yaml`).
The complete 5×4 Figure 8 passive-default matrix again yields zero full
tonic/burst survivors: every depolarized trace has one onset event rather than
a sustained tonic train, while only the 0.5 and 1.0 µF/cm² candidates retain
the two-event transient burst. The paper Figure 19 profile passes frequency
dependence, ACh suppression, and 500-ms recovery (1.302 mV error); the ModelDB
profile passes the first two but fails recovery (6.807 mV error). For isolated
layer-5 propagation, only paper-literal Equation 2 produces somatic output (two
events); both KInNeSS axial forms and the symmetric cable candidate produce
none. These results preserve genuine partial reproductions but provide no
complete Stage A candidate because the Figure 8 and TRN gates still fail.

The unrecovered version-1 calcium conductance unit has now been calibrated as
a registered Figure 8 training variable rather than guessed from a successful
plot (`figure8-legacy-unit-calibration-118.yaml`). A ten-point logarithmic grid
between the two documented interpretations of serialized `g_bar=250` produced
zero full survivors. At 0.25 mS/cm², the top condition is a sustained six-event
tonic train but the bottom condition is an eighteen-event sustained train. At
250 mS/cm², the bottom condition is the desired two-event transient burst but
the top condition has only one onset event. Intermediate values never pass
both gates. Thus a single version-1 calcium-unit scale is insufficient, and no
calibrated value is promoted.

The paper/archive Na-rate discrepancy has now been decomposed rather than
treated as one all-or-nothing profile. SMART Methods and ModelDB differ in both
the sodium activation scale (0.128 versus 1.28) and inactivation offset (27
versus 17 mV), so the two cross-source hybrids are legitimate discrete
discriminators. Their complete 384-candidate TRN matrix is finite throughout
but has zero survivors (`calibration-stage-a-trn-hybrid-nak-119.yaml`). Of 384
controls, 314 are quiescent; every one remains unrecruitable. Candidates using
archived activation instead support autonomous events, sometimes modulated by
drive, while printed activation remains silent regardless of inactivation
offset. The missing TRN mechanism is therefore not either Na-rate constant in
isolation.

A protocol audit then found that every preceding isolated-TRN matrix omitted
projection 010, the relay→TRN AMPA port, while applying only the two layer-6II
ports. This contradicted the assay's stated relay/layer-6II drive and the
connected Figure 7 current-balance evidence. A new network diagnostic measured
the missing relay gate at 20.60478, compared with 1.73256 layer-6II AMPA and
0.11284 layer-6II NMDA. Under all three measured gates, the complete current
768-candidate matrix produces six quiescent-yet-recruitable survivors
(`calibration-stage-a-trn-complete-drive-120.yaml`). The lexicographically
first candidate uses Table 3 intrinsic cells, ModelDB calcium kinetics,
Methods-global 250 mS/cm² calcium density, archived Na/K rates, serialized
KInNeSS axial coupling, physical initialization, and the latched event rule.
Its control has zero events and peaks at +24.27 mV; drive reaches +31.21 mV and
produces four events. This profile is frozen as
`configs/calibration/trn_stage_a_survivor_v1.yaml`. The prior 114–116 and 119
matrices are superseded as incomplete-drive assays, not evidence against this
candidate. The selected profile does not yet pass the layer-5 propagation or
dedicated Figure 8 gates, so full Stage A remains incomplete.

The selected TRN survivor does not reproduce the official Figure 6 training
target (`calibration-network-trn-survivor-121.yaml`). The network is finite and
recruits relay, layer 4, layer-2/3 interneurons, layers 6I/6II, TRN, and
nonspecific thalamus, but layer-2/3 pyramidal and layer-5 populations remain
silent. The learned bottom-up and top-down maps both fail their predeclared
orientation gates. At layer-2/3 cell 40, proximal AMPA reaches +1.369 nA while
the delayed GABA trough reaches -4.423 nA; only +58.1 pA reaches the soma and
its peak remains -63.2 mV.

The measured-gate isolated discriminator in
`figure6-layer23-transfer-discriminator-122.yaml` further localizes this result.
Excitation alone causes somatic output under every admissible axial convention.
When inhibition is introduced after the observed 2.30-ms current-peak delay,
the three KInNeSS/cable profiles remain silent and paper-literal Equation 2
emits one event. However, the real transient full Equation-2 network previously
also failed to recruit layer 2/3. Therefore neither an axial convention switch
nor a peak-gate substitution is justified; the source-defined transient
excitation/inhibition waveform is the earliest unresolved Figure 6 gate.

The alternative KInNeSS-primary-source reading of SMART.nml Spread X/Y as
Gaussian variances has also been executed under candidate fingerprint
`0b118fb5...` (`calibration-network-gaussian-variance-123.yaml`). It advances
the first layer-4 event from 23.26 to 19.24 ms but produces 162 layer-4 and 162
interneuron events, 1053 TRN events, and 245 category events while layer 2/3 and
layer 5 remain silent. Bottom-up and top-down orientation gates still fail.
This rules out the registered Gaussian interpretation as a Figure 6 rescue.

The relay-level intact/control assay now moves the earliest Figure 6
discriminator upstream (`figure6-relay-current-balance-125.yaml`). Cell 40
receives the archived external input at an effective -12 mV reversal, peaks at
+46.72 mV, emits once at 1.89 ms, and recruits layer 4 at 23.26 ms. With all
network projections absent, the unchanged relay emits 19 events during the
same 100-ms drive. Removing only the three serialized TRN-to-relay GABA_A
projections (records 000, 001, and 004) likewise restores 19 relay events.
Therefore the immediate relay-repetition failure is caused by the intact TRN
feedback pathway, not the external-input conversion, relay intrinsic
HH/T-current implementation, or local thalamic-interneuron inhibition. The
TRN projection-removal control still produces only one layer-4 event, so it is
a causal localization control rather than a candidate SMART reproduction.

The same 20-ms intact-network relay gate has been run for all six candidates
that passed the independently registered TRN control/recruitment screen
(`figure6-relay-survivor-screen-126.yaml`). Every combination of the three
admissible axial conventions and two admissible Equation-8 event rules emits
exactly one center-relay event at 1.89 ms; none repeats. Thus survivor selection,
axial coupling, and event-rule hysteresis are excluded as sufficient rescues.
The next source-level audit is the TRN recruitment/transmitter/GABA pathway,
while retaining all three official TRN-to-relay projections.

Adding the 5-ms pre-drive used by the isolated TRN screen does not resolve this
network result (`figure6-relay-equilibration-127.yaml`). During the following
20-ms bar drive, center relay cell 40 emits no event and remains below -68.95
mV. Equilibration therefore lets the connected network enter the suppressive
state before stimulus onset and is rejected as a Figure 6 rescue.

A population-resolved official-source hybrid now uses the archived SMART.nml
relay cell and retains the calibrated Table 3 cells for every other population
(`figure6-relay-source-hybrid-128.yaml`). The archived relay's inactive
condition is quiescent while a green-120 drive emits at 5.28 ms. In the full
100-ms episode, exactly the five stimulated relay cells and five layer-4 cells
emit, rather than the prior full-sheet 81/81 initialization volley. The
bottom-up map has positive horizontal contrast 0.006593 and passes Figure 6b.
TRN output falls from 810 to 320 events. Layer 2/3 pyramidal and layer 5 remain
silent, no relay event follows teaching arrival, and top-down contrast is
-0.0003045. The profile is therefore retained as a partial source-derived
candidate but is not promoted as Figure 6 or classic SMART reproduction.

Combining that population-resolved relay source with the paper-literal
Equation 2 axial profile produces the leading Figure 6 candidate
(`figure6-relay-axial-source-hybrid-129.yaml`). The ordered layer-4 -> 2/3 -> 5
chain now occurs at 5.79, 24.73, and 27.44 ms. Teaching arrives at 9.88 ms and
is followed by relay output at 28.47 ms, a causal 18.59-ms pair. Bottom-up
horizontal contrast is 0.34016. Combined top-down contrast is also positive at
0.007370, but remains below the predeclared 0.01 gate. Thus only the Figure 6c
contrast gate fails, and the candidate remains explicitly unpromoted.

The leading candidate's map decomposition is deterministic
(`figure6-relay-axial-map-decomposition-130.yaml`). Category cell 40 emits at
7.88, 14.93, 39.27, 62.02, and 82.92 ms, while each horizontal relay follows a
closely related seven-event rhythm. The narrow field supplies 0.006526 of the
combined contrast. The wide field supplies only 0.000845 because its nearest
horizontal targets depress while farther horizontal targets potentiate;
inactive vertical-arm weights do not change. This localizes the shortfall to
causal/anti-causal phase balance rather than isotropic tail growth.

All remaining registered intrinsic alternatives have been screened around
that exact profile (`figure6-leading-source-alternatives-131.yaml`). The
hysteretic event rule is bit-identical; the literal rule, internal-zero
initialization, and two printed-activation Na/K families fail upstream. Printed
paper calcium kinetics, the surviving Na/K hybrid, and an archived layer-6II
cell all worsen the top-down or causal-pair gates. Restoring source-serialized
top-down initial weights starts the combined center at 3.0, drives 162 relay
events, and produces negative top-down contrast, contradicting the weak
Figure 6c before-map. None is promoted.

An exact Equation 25/28 term audit now closes the remaining Figure 6c
weight changes to floating-point precision
(`figure6-top-down-learning-phase-132.yaml`, maximum additive residual
`1.52e-16`). For wide projection 005, the nearest horizontal targets change by
`-0.003763`: their positive correlation contribution (`+0.057777`) is exceeded
by the anti-causal contribution (`-0.068561`), despite a small positive
baseline term (`+0.007021`). The farther horizontal targets instead change by
`+0.005452` because their anti-causal contribution is only `-0.028240`.
Narrow projection 007 correctly potentiates the nearest horizontal targets by
`+0.013052`. Vertical targets emit no relay events and remain unchanged. This
proves that the last Figure 6c failure is the wide pathway's spatially unequal
post-spike depression, not numerical integration, inactive-tail growth, or a
missing presynaptic category gate. The passive audit accumulators do not feed
back into SMART state. The leading candidate remains unpromoted.

The resulting source-literal bounds discriminator has also been executed
(`figure6-projection-level-learning-bounds-133.yaml`). It preserves the
complete cortical chain, causal teaching pair, and positive Figure 6b contrast
(`0.32355`), but fails Figure 6c with combined contrast `-0.01374`. Because
Equation 25's uniform projection-level `w0=0.05` acts during the presynaptic
category gate even when a relay target does not spike, inactive vertical-arm
weights grow toward `w0`; both wide and narrow fields become vertically
biased. This independently reproduces the previously inferred inactive-tail
problem and rejects projection-level bounds as the complete Figure 6 runtime
interpretation. It is retained as a source-literal negative control, not
promoted.

Decoupling only Equation 6's `D` from the local Equation 25 baseline is also
rejected (`figure6-projection-depression-scale-134.yaml`). The candidate uses
the serialized projection ratio `D=-0.05/1.5` while leaving inactive spatial
baselines local. It retains the cortical chain, causal pair, and Figure 6b,
but combined Figure 6c contrast falls to `-0.01531`; every active horizontal
wide weight depresses, and active narrow weights cross below zero. This is not
a bounded KInNeSS learning system and cannot be promoted. Together with
artifact 133, it shows that projection-level `D` cannot be separated from its
projection-level Equation 25 bounds to rescue the graphical local-baseline
interpretation.

The exact teaching-volley decomposition
(`figure6-teaching-volley-decomposition-135.yaml`) partitions every integrated
weight term with no residual. In the prior global-paper-axial leader, the
onset relay spike at 1.14 ms leaves a negative Equation 6 tail that overlaps
the first category arrivals at 9.88 and 16.93 ms. For wide target 39 this first
window contributes `-0.005070`, exceeding the connection's final
`-0.003763` change. Direct visual and textual review of the primary Figure 6
caption confirms that learning is active during the same 100-ms episode; it
does not authorize suppressing this onset window. The Results text instead
states that the oriented top-down weights are learned during gamma
oscillations.

A source-coherent population-resolved axial profile initially passed the
shape-only registered Figure 6 target
(`figure6-population-resolved-axial-136.yaml`). The archived SMART.nml relay
cell uses its archived KInNeSS serialized axial edges, while cells selected
from paper Table 3 use paper-literal Equation 2 coupling. Each of the five
stimulated relays emits exactly four events in 100 ms (40 Hz), rather than the
seven events caused by applying paper axial coupling to the archived relay.
The ordered layer-4 -> layer-2/3 -> layer-5 chain occurs at 9.93, 28.87, and
31.59 ms. Later resonant teaching pairs include 90.00 -> 95.70 ms. Bottom-up
horizontal contrast is `0.23252`, and combined top-down contrast is `0.03045`,
above the fixed `0.01` gate; inactive vertical arms remain unchanged. The
complete run is deterministic across the initial and confirmation executions.
That promotion is superseded by the amplitude audit in
`figure6-population-axial-amplitude-audit-140.yaml`. The combined top-down map
peaks at only `0.11853`, below the primary Figure 6c after-learning range of
approximately `0.5--2.5`, despite its positive `0.03045` orientation contrast.
The profile is retained as the leading rate/timing/shape candidate but
retracted as a complete Figure 6 reproduction. Classic SMART remains unfrozen.

The first direct Figure 7 holdout using the genuinely learned Figure 6 weights
has now been completed
(`calibration-figure6-figure7-population-axial-137.yaml`). Under the historical
shape-only contract, Figure 6 passed for a third deterministic run, but match
and mismatch each produce only one
nonspecific-thalamus event at 0.74 ms, or 10 Hz, rather than approximately
40/70 Hz. Both conditions recruit five relay cells in the bottom-up input
shape, all 81 TRN cells emit the same synchronized event at 5.53 ms, and TRN
does not express the required match-greater-than-mismatch ordering. The
artifact is correctly labeled `holdouts_consulted: true`; Figure 7 is not
reproduced. Its Figure 6 pass field is historical evidence under that older
contract, not evidence against the later amplitude-audit retraction.

Two source-driven post-holdout TRN candidates were rejected at the mandatory
Figure 6 prerequisite. Applying KInNeSS axial edges to all thalamic populations
while retaining paper Equation 2 in cortex yields 319 TRN events, only the five
relay onset events, and no Figure 6c learning
(`figure6-thalamocortical-axial-138.yaml`). Applying the complete isolated TRN
survivor package--KInNeSS axial plus Methods-global calcium density only in
TRN--causes 810 TRN events, zero relay events, and collapses the whole cortical
chain (`figure6-population-resolved-trn-139.yaml`). Isolated
quiescent/recruited TRN viability therefore does not predict full-network
spatial recruitment. Neither candidate is promoted.

The exact learning-term audit under the retracted population-axial profile is
archived in `figure6-population-axial-learning-phase-141.yaml`. Five category
volleys and four spikes per active relay produce only `0.01366` wide-field and
`0.02640` narrow-field growth at representative target 39. All Equation 6
terms close with maximum residual `1.22e-16`; the positive postsynaptic overlap
is only `0.411--0.423 ms` across the entire 100-ms episode. The amplitude gap
is therefore not a numerical integration error. The next discriminator must
separate the published +30-mV output-event detector from the Equation 6
postsynaptic learning-threshold interpretation, while leaving spike emission
unchanged.

That discriminator is complete in artifacts 142--145. Absolute learning
thresholds of `0 mV` and the archived KInNeSS `-20 mV` alternative preserve
the five-cell, four-spike relay pattern but raise the combined map only to
`0.15232` and `0.16854`. Evaluating Equation 6 in the paper's leak-relative
coordinate raises the map to `0.52741`, but recruits 58 relay events and still
falls far below Figure 6c's approximately `2.5` peak. The acceptance contract
now requires a conservative `2.0` peak plus exactly four events in each of the
five horizontal relay cells and no off-bar relay events. Threshold value and
coordinate are rejected as sufficient explanations; no profile is promoted.

The paper-Methods learning-rule contradiction is now tested in artifacts 146
and 147. Methods 4.3 dual-AND gating with the absolute learning coordinate
preserves confined 40-Hz relay recruitment but peaks at only `0.19319`.
Crossing dual-AND with leak-relative voltage increases the peak to `0.96876`,
but spreads 58 relay events across 43 cells. Both fail the `2.0` amplitude and
spatial-confinement contract. The Figure 6 caption/serialized presynaptic rule
and Methods dual-AND rule remain provenance-labeled alternatives, but neither
currently reproduces Figure 6c.

Artifact `figure6-dual-and-leak-learning-phase-149.yaml` decomposes the
strongest rejected interaction to maximum residual `5.55e-16`. Horizontal
targets receive `5.83--5.97 ms` positive overlap and grow to approximately
`0.48`, but vertical targets that emit no relay spikes still receive
`2.23--2.28 ms` positive overlap and grow to approximately `0.23--0.25`.
Leak-relative Equation 6 gating therefore treats subthreshold surround
depolarization as a positive postsynaptic signal, mechanistically explaining
the loss of spatial confinement. The next discriminator returns upstream to
the unresolved action-potential waveform/legacy membrane implementation; it
must not weaken spatial gates to retain this candidate.

The connected relay waveform audit
(`figure6-relay-waveform-nak-audit-150.yaml`) measures all four registered
Na/K source families during the first 20 ms of the official episode. Standard
Traub-Miles and archived-activation/printed-inactivation both emit one relay
event and have nearly identical widths: `0.18/0.17 ms` above +30 mV and
`0.33/0.32 ms` above 0 mV. The two printed-activation families peak near
`-50.05 mV`, emit no relay event, and recruit no layer-4 cell. Na/K family
choice therefore cannot supply the missing positive Equation 6 duration; no
waveform candidate is promoted.

Equation 6's spike timestamp has also been separated from Equation 8 emission
(`figure6-learning-timestamp-upward-151.yaml`). Timestamping the postsynaptic
spike at the upward +30-mV crossing leaves every population spike count and
the confined five-cell relay pattern unchanged, but lowers the combined map
peak from `0.11853` to `0.10750`. The earlier timestamp starts the depressive
tail sooner and increases its overlap with presynaptic activity. Thus reusing
the later falling-phase event was not the source of insufficient potentiation;
the upward alternative is retained as rejected provenance, not promoted.

The remaining paper-coordinate combination is tested in
`figure6-learning-coordinate-leak-plus30-152.yaml`: Equation 6 uses
leak-relative voltage with its printed `+30 mV` threshold while Equation 8
emission remains absolute. This preserves exactly four events in each of the
five horizontal relay cells, but its combined map peaks at only `0.18606`.
Unlike leak-relative zero, it does not potentiate the surround. The registered
threshold/coordinate family is now exhausted without a Figure 6c survivor.

### Figure 7 numeric-target correction

A direct audit of the official Figure 7 caption and Results Sections 2.2-2.3
retracts the repository's earlier 40/70-Hz nonspecific-thalamus acceptance
pair. The paper states only a directional mechanism: a match recruits more
specific-thalamic cells and therefore more TRN inhibition, while mismatch
disinhibits the nonspecific thalamus and increases its firing. The paper's
40-Hz value belongs to Methods 4.9 relay-input calibration, and its 70-Hz
values occur in unrelated transmitter/firing protocols. Historical artifacts
retain their predeclared numeric scorer for provenance, but the active Figure
7 contract now requires the published spatial relay subset, stronger TRN
output during match, and higher nonspecific output during mismatch. Artifact
156 still fails the corrected contract because both conditions produce five
active relay cells, 81 TRN events, and 10-Hz nonspecific output.

Population-specific TRN event-coordinate discriminators are now executable.
The leak-relative detector preserves the Figure 6 relay/cortical trajectory
and changes the first TRN volley, but Figure 7 remains condition-invariant at
10/10 Hz without a cue lead (`figure7-trn-kinness-event-coordinate-pair-159.yaml`).
With a 10-ms lead it produces 441 identical TRN events, suppresses every relay,
and yields 40/40 Hz (`figure7-trn-kinness-event-coordinate-lead10-160.yaml`).
The causal cue screen then isolates the failure: after 20-ms equilibration,
300 pA causes no category or relay event but all 81 TRN cells emit at
cue-relative 4.43 ms (`figure7-cue-current-screen-161.yaml`). The separate
fixed +67-mV KInNeSS output coordinate produces the same autonomous volley
(`figure7-shifted67-cue-screen-162.yaml`). Both shifted detectors are rejected;
the absolute physical event coordinate remains active pending a source-backed
resolution of why published-source TRN waveforms peak near +7 mV rather than
crossing the printed +30-mV arm threshold.

The absolute-coordinate cue screen (`figure7-absolute-cue-current-screen-163.yaml`)
finds the first clean selective top-down operating point at 600 pA: the center
category cell emits during the 10-ms lead, no off-source category cell emits,
and no TRN or relay event precedes bottom-up onset. A full pair without a rest
epoch (`figure7-absolute-event-coordinate-lead10-164.yaml`) is rejected as an
initialization-contaminated diagnostic because all 81 TRN cells emit before the
category event. The runner now exposes an explicit Figure 7 equilibration
epoch, defaulting to the 20 ms used by the causal cue screen.

With that equilibration (`figure7-absolute-event-coordinate-equilibrated-lead10-165.yaml`),
the pre-cue TRN volley disappears and category recruitment remains selective,
but both conditions eventually recruit all 81 relay cells, produce zero scored
TRN events, and leave nonspecific thalamus silent at 0 Hz. The pathway is not
under-driven: sampled TRN relay-AMPA gates peak near 84--100 and layer-6II AMPA
gates near 16--25. TRN proximal voltage reaches approximately +37.09 mV, while
sampled soma maxima remain between approximately -3.64 and -12.74 mV. The next
discriminator is therefore the source meaning of TRN compartmental output and
axial propagation, not a fitted increase in excitatory synaptic strength.

The next two finite official-source matrices close additional intrinsic
explanations without fitting channel parameters. The Table 3/SMART.nml TRN
potassium matrix (`figure7-trn-potassium-source-screen-166.yaml`) crosses soma
potassium density 100/80 mS/cm2 with reversal -90/-100 mV. Every candidate is
quiescent after 20 ms, but all produce zero post-bottom-up TRN events. The best
sampled soma maximum is -12.11 mV, still 42.11 mV below the printed +30-mV arm
threshold, while proximal dendrites reach +37.64 mV. Potassium density and
reversal are therefore rejected as the missing propagation convention.

The calcium source matrix (`figure7-trn-calcium-source-screen-167.yaml`) crosses
the Table 3 absence versus SMART.nml presence of a somatic T channel with
calcium reversal 180/120 mV. Adding the somatic channel does not restore output.
The 120-mV reversal without that channel does cross the somatic event threshold,
but all 81 TRN cells emit during the 10-ms top-down cue lead, suppress relay
output completely, and fail the predeclared causal gate. This is a real
excitability transition but not the Figure 7 match mechanism.

Primary-paper review then corrects the protocol role of that lead interval.
The Figure 7 Results text explicitly describes simultaneous bottom-up and
top-down excitation; the 10-ms lead remains a diagnostic, not the canonical
trial. The frozen 120-mV candidate was therefore tested once under simultaneous
onset (`figure7-trn-calcium-reversal-simultaneous-pair-168.yaml`). Match and
mismatch are identical: zero active relay cells, 229 TRN events across all 81
cells, and 30-Hz nonspecific output. The source change restores TRN somatic
events but destroys the two-against-one relay match and is rejected. No
intermediate calcium reversal may be fitted to Figure 7. The next discrete
source discriminator is the still-unfactored TRN dendritic calcium density
conflict (Table 3 10 versus SMART.nml 100 mS/cm2), preserving the same
equilibration and simultaneous-onset contract.

That final discrete density crossing is complete in
`figure7-trn-dendritic-calcium-source-screen-169.yaml`. At the archived
100-mS/cm2 dendritic density, all four soma-channel/reversal combinations
remain post-equilibration quiescent and emit zero connected TRN events. This is
not insufficient dendritic activation: sampled proximal peaks span
approximately +85.99 to +134.33 mV, while the best sampled soma remains only
-23.43 mV. Combined with artifacts 167 and 168, the complete public-source
10/100 density x soma-channel x 180/120-mV reversal cube has no causal Figure 7
survivor. The surviving sources therefore do not identify a discrete legacy
TRN propagation convention that reproduces SMART. Any next density search must
be labeled behavior calibration, predeclare a finite grid between the two
official endpoints, and relinquish an exact-source parameter claim while
retaining Figures 10 and 14--16 as untouched holdouts.

The registered behavior-calibration density grid is complete in artifacts 170
and 171. Stage 1 crosses 10, 15, 20, 30, 40, 60, 80, and 100 mS/cm2 at the
archived 120-mV reversal. The 10-mS/cm2 endpoint repeats the 81-cell
top-down-only TRN volley and is rejected; every value from 15 through 100 is
cue-safe. Stage 2a then applies simultaneous match onset for 50 ms. All seven
survivors preserve exactly the five horizontal relay cells, with total relay
events increasing from five to nine, but every value produces zero TRN and
zero nonspecific events. No value reaches the predeclared match gate, so no
mismatch run is authorized. A scalar dendritic calcium density between the two
official endpoints is now rejected as a sufficient behavioral calibration.
Artifact 172 then resolves the suspected topology ambiguity: the executable
source explicitly declares a linear Soma -> Dendrite 0 -> Dendrite 1 cable,
only the soma monitors spikes, and the implementation already builds those two
adjacent edges. A star topology or dendritic chemical output is not admissible.
The active discrepancy is now confined to unidentified legacy KInNeSS
propagation/somatic-event behavior: strong dendritic spikes coexist with silent
somatic output despite source-matched ordering and output location.

The first explicitly behavior-calibrated propagation test is now complete in
artifacts 173 and 174. It multiplies both directional conductances on only the
TRN soma--proximal edge over the predeclared grid 1, 1.25, 1.5, 2, 3, 4, 6, 8,
12, and 16, while fixing the linear topology, proximal--distal edge, archived
120-mV calcium reversal, 100-mS/cm2 dendritic calcium density, intrinsic
channels, synapses, and event detector. Every scale is cue-safe. In the
simultaneous match assay, 1--3x preserve exactly relay indices 38--42 but give
zero TRN output; 4--16x activate all 81 relay cells and still give zero TRN
output. Sampled somatic peaks range from approximately -35.15 to -23.77 mV and
never approach the printed +30-mV arm threshold. No candidate advances to
mismatch. Local axial gain is rejected, and no finer edge-scale search is
authorized.

Artifacts 175 and 176 then test the remaining detector-coordinate hypothesis
without lowering SMART's +30/0-mV arm/release landmarks independently. A
numeric TRN-only voltage-origin offset is predeclared from 0 through 69 mV,
bounded by the absolute, fixed +67-mV, and TRN leak-relative +69-mV source
coordinates. Figure 7 instrumentation now records equilibration-window output;
the cue gate additionally requires the final 10 ms of equilibration to be
quiescent. Offsets 0--40 mV retain the synchronized 81-event startup TRN volley,
settle, and remain cue-safe. The 50-mV offset is rejected with 405 equilibration
TRN events (162 in the tail) and 162 cue-lead events. Offsets 60/67/69 mV remove
the startup volley entirely and are tail/cue quiet. During simultaneous match,
0--40 mV preserve relay indices 38--42 but produce zero TRN; 60/67/69 mV
activate all 81 relay cells and also produce zero TRN. No candidate advances to
mismatch. A detector-origin offset cannot recover Figure 7 without either
missing TRN output or destroying the prepared relay state.

The registered density-by-offset interaction screen is complete in
`figure7-trn-density-event-offset-cue-grid-177.yaml`. It fixes the first
event-generating 50-mV detector offset and crosses the already registered
10, 15, 20, 30, 40, 60, 80, and 100 mS/cm2 dendritic calcium densities. No
candidate is cue-safe: equilibration TRN output is 162, 162, 162, 243, 243,
324, 324, and 405 events, respectively, and all candidates add at least one
population-wide 81-event volley during the top-down-only lead. The
simultaneous match stage is therefore not run. This closes the simplest
second-order scalar rescue; the missing behavior is not recoverable by trading
dendritic calcium strength against a shifted somatic detector origin.

The soma--proximal event-transfer grid in artifacts 178 and 179 produces the
first calibrated early-match survivor. Every blend from 0 through 1 is quiet
during the original cue gate. Under simultaneous 50-ms match, blend 0.5 alone
preserves relay indices 38--42 while generating 81 TRN events. Values 0--0.1
remain selective but TRN-silent; 0.2--0.4 lose relay selectivity without TRN
output; 0.6--0.9 recruit all relay cells and 81 TRN cells; the proximal endpoint
is nonselective and silent. Artifact 180 independently repeats the 0.5 match,
but mismatch activates the vertical input indices 22, 31, 40, 49, and 58 with
the identical TRN event train and no nonspecific output. It is not yet a Figure
7 reproduction.

Primary-source audit 181 shows that two earlier gates were too restrictive.
Grossberg and Versace explicitly route top-down layer-6II excitation through
TRN to inhibit top-down-only LGN cells, so cue-period TRN output is compatible
with the intended one-against-one mechanism. Their complete-module mismatch
simulation uses 300-ms epochs and places the first nonspecific increase around
50 ms, so a 50-ms pair cannot close the arousal claim. The next registered test
holds blend 0.5 fixed, reopens only the existing 600/800/1000-pA undocumented
top-down-current grid, permits causal cue-period TRN output, and evaluates final
match/mismatch directionality over 300 ms.

The corrected current screen is complete in artifacts 182--184. Currents 600,
800, and 1000 pA are all cue-safe and all pass the 50-ms match gate with five
horizontal relay events and 81 TRN events. Over the primary-paper-motivated
300-ms epoch, sustained relay activity emerges. Counts are 389/432, 441/419,
and 418/426 for match/mismatch at the three currents; only 800 pA has the
published match-greater-than-mismatch relay direction. TRN output remains
81/81 and nonspecific output remains 0/0 at every current, so no amplitude is a
Figure 7 survivor. Raising the undocumented current within its registered
range cannot recover arousal.

The next failure is cellular rather than an absence of nonspecific drive.
Full pathway diagnostics in artifact 165 show nonspecific-thalamic proximal
voltage above +129 mV under both conditions, with somatic peaks near +13 mV and
no emitted event. This mirrors the legacy compartment-to-event propagation gap
already found in TRN. A nonspecific-population event-transfer calibration is
therefore the next isolated mechanism test; synaptic strengths, delays, and
holdout figures remain locked.

## Validation gates

A milestone may be marked complete only when:

1. each equation and parameter has a provenance status;
2. deterministic tests pass and stochastic results reproduce across declared seeds;
3. target figure protocols and readouts are documented before tuning;
4. expected and negative-control outcomes are both reported;
5. generated data and summaries include the exact configuration fingerprint.
