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

The Figure 16 source audit corrects an earlier roadmap/target transcription:
the caption sets the V1-layer-2/3-to-V2-layer-4 delay to 10 ms, not 1 ms. The
recovered `SMART.nml` record independently serializes 5 ms. A named protocol
helper now overrides only that projection after archive-faithful network
construction, preserving all other cross-area delays. This passes focused
structural tests but does not yet establish the published inter-area
cross-correlation result (`figure16-delay-protocol-043.yaml`).

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

Accordingly, output from M2 is a source/audit benchmark, not evidence that
the 2008 results have already been replicated.

## Validation gates

A milestone may be marked complete only when:

1. each equation and parameter has a provenance status;
2. deterministic tests pass and stochastic results reproduce across declared seeds;
3. target figure protocols and readouts are documented before tuning;
4. expected and negative-control outcomes are both reported;
5. generated data and summaries include the exact configuration fingerprint.
