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

Figure 6b's before-learning scale resolves the leading Gaussian-amplitude
ambiguity. The relay→layer-4 XML weight is 6 with X/Y spread 0.5, while the
published map peaks near 4; a normalized bivariate Gaussian gives
`6/(2*pi*0.5^2)=3.82`. With this fingerprinted `normalized_density` convention
and source `CM=1 μF/cm²`, the complete 12-population first-order sector produces
exactly five horizontal-bar-aligned layer-4 excitatory spikes at 17.87--18.04
ms, following the relay volley at 5.75 ms, with no off-bar layer-4 excitatory
spikes. This passes the qualitative selective-recruitment and positive-latency
claim in Figure 6b (`figure6-recruitment-011.yaml`). It does not yet reproduce
the 100-ms learned weight maps or prove the separate 40-Hz relay target under
the same profile. KInNeSS's benchmark confirms finite connection kernels, but
the exact truncation radius and `ring` stencil remain unresolved.

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
