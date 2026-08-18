# ModelDB 112923 executable-source audit

ModelDB alternate version 112923 is the preserved executable backup associated
with Grossberg and Versace (2008). It was recovered on 2026-08-11 from
`https://modeldb.science/download/112923`.

## Integrity and licensing

| Item | SHA-256 |
|---|---|
| Archive | `5fc6e0042ef093665ad2f1c24c87f9cc0d796629c896ee8a4efe34827a4462c5` |
| `SMART.nml` | `16fc196f2ed68d1c02a39395fa83bb0ffab8fcf8edffe6fd11022bb6c6e299e1` |
| `Ca_rebound.xml` | `2fc9615bf119e7602e75c00b5cd29e74dfaf89ba96d8eb6b0f17e36932f1ad0d` |
| `Layer_5_and_Maynert_AHP_ACh.nml` | `682355d883b98f727087c5f959f2c776d9e47edfd6ff5896d1cf394d283ff560` |

The README states copyright but does not give an open-source license. Raw files
are therefore not committed. The archive is used as a primary-source audit and
its numerical/model facts are transcribed with provenance.

## Structural facts

`SMART.nml` contains 24 populations: 12 in the first-order V1 sector and 12 in
the V2 sector. In each sector, ten externally sized grids correspond to the 9×9
model sheets and the intralaminar/nonspecific and matrix populations are fixed
1×1 sheets. The first-order sector therefore contains 812 cells and 1,950
compartments when the external grids are instantiated as 9×9. The full
two-area assembly contains 1,624 cells, 3,900 compartments, 118 serialized
projections (109 chemical and nine gap-junction records), 16 external channels,
and nine projections crossing the V1/V2 boundary. The first 12 populations
contain 56 serialized projection records, one of which is the self-triggered
layer-5 AHP channel; the inter-population first-order catalog therefore has 55
records.

These executable counts also agree with the article's detailed Methods 4.1
population-size statement and the compartment classes in Table 3. They conflict
with Methods 4.2's separately printed totals of 732 cells and 2,106
compartments, which cannot be obtained from the stated population sizes and
two-/three-compartment cell classes. The archive controls construction; the
printed totals remain a documented publication anomaly rather than a reason to
invent a resize.

The complete 24-population topology now builds and executes a zero-duration
Brian2 network check. This is structural validation only: it does not establish
that the published match/gamma or mismatch/beta/reset trajectories have been
reproduced. A hash-pinned independent extraction now verifies all 24 runtime
cell definitions against the raw XML for compartment geometry, leak, axial
fields, Na/K/T-calcium densities and reversals, AHP kinetics, and transmitter
depletion fields. Eleven V2 populations are exact intrinsic homologues of their
V1 counterparts. V2 layer 5 is the sole exception: its leak reversal is -72
rather than -73 mV and its two serialized child-edge axial resistances are 5
rather than 6 kOhm*cm.

## Resolved equation ambiguities

- Sodium activation uses the standard Traub–Miles coefficient `0.32`, not the
  paper's printed `0.032`; sodium inactivation uses a 17 mV offset, not 27 mV.
- KInNeSS defines zero on the voltage-gate axis as the compartment's resting
  potential. The classic network therefore evaluates Na/K rates using
  `V_membrane - E_leak`, rather than a cell-independent 67 mV shift. T-type
  gates retain their separately serialized absolute-voltage formulas.
- SANNDRA's archived CVS revision history resolves the missing gate initial
  state. The same 2004 change is recorded in `gates.h`, `layer.h`, and
  `unit.cpp`: `TGate.init()` resets voltage-gated currents to resting
  potential. The classic runtime therefore initializes each activation and
  inactivation gate to its equilibrium occupancy at the compartment's
  Table 3 initialization voltage. Literal zero initialization remains an audit
  alternative and is not the executable baseline. The KInNeSS manual warns
  that voltage-gated channels shift actual membrane rest away from configured
  leakage equilibrium, so this interpretation remains explicitly auditable
  pending recovery of the missing `TGate.init()` source.
- For relay T-current, the XML marks the sigmoid with `V0=-63, B=7.8` as
  `m_inf`, while `Simple_Tau(A=2.44, B=2.506, V0=-9.84)` is the activation time
  constant. Likewise, the `V0=-83.5, B=-6.3` sigmoid is `h_inf`, while the
  `A=19.15, B=7.171, V0=-10.54` expression is the inactivation time constant.
  The paper labels these roles in the opposite order and prints 19.5 rather
  than the executable's 19.15.
- TRN does not use that relay T-current family. `SMART.nml` gives all three
  reticular compartments a distinct Destexhe et al. channel with activation
  exponent 2, `m_inf(V0=-52, B=7.4)`, `h_inf(V0=-80, B=-5)`, and the
  six-parameter KInNeSS `Reticular_Tau` function. The runtime now compiles this
  family separately for V1 and V2 TRN. Applying the relay exponent-3 and
  `Simple_Tau` equations to TRN was an executable-source transcription error.
- The contemporaneous KInNeSS paper defines `Simple tau` as
  `A + B*10^-2*exp(C*10^-2*V)`, confirming the ModelDB time-constant
  transcription. It also supplies the exact directional inter-compartment
  Equation 7, now available as the named `kinness_2008` axial convention.
- `SMART.nml` serializes `inpResistance` on child compartments while root
  somata omit it. The `kinness_serialized_edge` convention therefore applies
  the child connection value in both current directions, preserving the XML
  topology rather than manufacturing a root parameter.
- The archived KInNeSS User Manual resolves `connectFromMany`: the Gaussian is
  scaled so its peak equals the projection's `Weight`, and shoulders whose
  resulting weight is below 0.001 are cut. The executable baseline therefore
  uses finite, peak-scaled `source_peak` kernels. The formerly promoted
  `normalized_density` interpretation was an inference from Figure 6b's plotted
  scale and is retained only as an audit alternative. Exact `ring` geometry
  remains unresolved.
- The framework defines many-to-one connectivity as a Gaussian centered at the
  selected cell and the archived adaptive layer-6II-to-LGN records explicitly
  use wrapped, non-ring kernels. Their finite support is determined by the
  documented 0.001 resulting-weight cutoff rather than a serialized radius.
  The separate layer-6II-to-TRN off-surround records carry `ring=true`; their
  exact legacy center/surround stencil remains unresolved.
- Figure 6's caption and archived NML records 005/007 call corticothalamic
  learning presynaptically gated, but Methods 4.3 says layer-6II projections to
  specific thalamus use dual-AND gating. The executable baseline follows the
  figure-specific caption and NML while retaining this publication conflict as
  unresolved provenance.
- The paper's Equation 8 defines an emitted spike on the falling phase: the
  current sample is below 0 mV after the preceding spike exceeded 30 mV. The
  runtime arms above 30 mV and emits once on the subsequent fall below 0 mV,
  so ligand release, transmitter depletion, AHP, and STDP timestamps share the
  official causal event. Both physical and leak-relative voltage-coordinate
  interpretations remain executable audit profiles.
- The contemporaneous KInNeSS framework manuscript instead prints -20 mV as
  the Hodgkin-Huxley event threshold while retaining the preceding-sample-below-
  zero condition. This combination emits repeatedly during one upstroke in the
  source-defined SMART network (35,667 TRN events in 100 ms) and abolishes the
  nonspecific response. The threshold is now an explicit runtime convention;
  -20 mV is a rejected audit alternative and SMART's +30 mV remains active.
- The executable first-order area contains 55 projected gates after excluding
  the self-triggered AHP gate: 51 chemical and four gap-junction records. This
  set is not identical to the 55-row supplementary catalog; it includes
  narrow/wide channels and different weights, asymptotes, delays, and topology
  flags. A generated, integrity-pinned ModelDB catalog is therefore the
  executable baseline, while the supplement remains an independent audit.
- The supplementary table prints chemical-channel conductance in pS and
  projection weight as receptor density in millions/cm², requiring a `10^-3`
  conversion in that independent transcription. The executable `SMART.nml`
  path is different: KInNeSS Equation 3 defines channel `g_bar` as maximum
  conductance density in mS/cm², and Equation 16 multiplies it by the
  dimensionless projection weight and ligand gate. ModelDB channels therefore
  use their serialized `g_bar` directly; applying the supplementary conversion
  to them would make all chemical synapses 1000-fold too weak.
- Ligand kinetics now implement KInNeSS Equations 13--15 per connection. Each
  synapse tracks its last two arrivals and combines their unit-normalized
  waveforms as `g1 + g2 - g1*g2`, rather than linearly accumulating every past
  event in one postsynaptic state. This preserves the required `[0,1]` gate
  bound before receptor-density weighting.
- For modifiable projections, XML `weight` is the upper bound and
  `assymptoticWeight` is Equation 25's decorrelated baseline `w0`. Figure 6
  resolves pathway-specific initial states: the strong Figure 6b bottom-up map
  starts at serialized `weight`, whereas Figure 6c's approximately 0--0.3
  before-learning scale matches the sum of the wide and narrow Gaussian-scaled
  0.05 asymptotes, not their roughly 2.1 combined serialized center weight.
- Section 4.9's Gaussian calculates the initial many-to-one synaptic weights,
  while Equation 25 defines projection-level upper bounds. Figure 6b retains a
  projection-level baseline; Figure 6c identifies a Gaussian-scaled local
  baseline for the two corticothalamic adaptive fields. Fully spatially scaled
  maxima remain an audit alternative.
- Ten voltage-driven input channels now follow KInNeSS Equations 4--5: their
  four serialized sensitivities shift the effective driving potential from
  red/green/blue/alpha source values in [0,255]. The eleventh external channel
  is a direct current-injection gate governed by Equation 6. It is compiled
  separately from conductance inputs; all four archived sensitivities are zero,
  so it remains inert until explicitly configured by an experiment protocol.
- Methods 4.9 does not identify which relay compartment is fixed at -12 mV.
  Direct per-step clamps of soma, proximal dendrite, and distal dendrite produce
  0, 70, and 20 Hz respectively, whereas recovered green=120 input through the
  archived proximal shunting gate produces the stated 40 Hz. Direct ODE
  replacement is retained as an audit alternative, not the executable default.
- All-zero voltage-input mappings are now an explicit runtime convention. The
  literal framework interpretation makes them permanent resting-potential
  leaks; omitting them as inactive legacy mappings changes the isolated relay
  from 0 to 60 Hz at 1 uF/cm2 and exposes a 40-Hz candidate at 2 uF/cm2.
  This behavior is fingerprinted and remains unpromoted pending network and
  source validation.
- The archived 9x9 stimulus PNGs contain five green=120 bar pixels plus
  blue=70 at the central pixel. The executable protocol now routes green to
  relay, nonspecific, and matrix input gates and blue to the central V1
  layer-6II category cell. `connectFromAll` gates sum individually valid pixel
  values instead of treating the sum as one out-of-range pixel.
- Gap-junction totals follow KInNeSS Equation 8 rather than treating XML
  `g_bar` as an already converted membrane density.
- KInNeSS serializes compartment dimensions in centimeters. The executable
  runtime converts those values to the internal millimeter representation;
  for example, the SMART relay soma `0.005×0.006` cm is the Table 3
  `0.05×0.06` mm soma. Treating the raw XML values as millimeters leaves total
  membrane-current ratios unchanged but makes axial coupling 100-fold too
  strong relative to membrane current and capacitance.

- Intercompartmental coupling follows Equation 9 of the expanded KInNeSS
  manuscript. A visual equation audit corrected an earlier summed-geometry
  transcription to the published harmonic geometry ratio. With one serialized
  child-edge resistance used in both directions, the resulting total currents
  are exactly equal and opposite even though their density effects differ by
  receiving-compartment area.
- The dedicated Figure 8 cell is not the Table 3 relay cell. It uses soma
  geometry 0.2×0.4 mm, 50/30 mS/cm² Na/K, and 250 mS/cm² T-current in soma
  and both dendrites. Its leak density is not serialized and remains a required
  KInNeSS-default/calibration input.
- The dedicated AHP/ACh file uses an AHP channel density of 0.1 mS/cm²,
  reversal -90 mV, AHP rise/fall 80/150 ms, connection weight 4.5, and 3 ms
  delay. These differ from the paper's 80/100 ms text and were not recoverable
  from the supplement alone.
- Its layer-5 cell has a 0.1×0.15 mm soma and 0.01×0.1/0.2 mm
  dendrites. Dendritic input-resistance fields are 35 and 30 kΩ·cm, but the
  root soma field is omitted. The executable profile therefore requires an
  explicit soma candidate instead of silently inheriting a Table 3 value.
- The self-projection weight 4.5 scales the AHP current while each output spike
  adds one normalized dual-exponential event. This preserves the source's
  distinction between channel gating and projection strength under overlap.

The dedicated executable AHP profile is implemented separately from the paper
profile: 80/150 ms kinetics and event weight 4.5 versus 80/100 ms and unit
events. Specific membrane capacitance remains an explicit runtime parameter
because neither Table 3 nor the recovered Figure 8 XML serializes its value.

## Current Figure 8 result

After correcting ModelDB T-gate initialization to use absolute membrane
voltage, the source-specific cell passes the predeclared hyperpolarized burst
criterion (two early spikes and no late spikes) under the current candidate
protocol. The depolarized condition still emits only one qualifying action
potential and fails the tonic-train criterion. This is partial validation, not
an official Figure 8 reproduction. Remaining audit targets are the legacy
clamp timing, membrane-capacitance default, and voltage-gate coordinate
handling.

## Consequence for baseline status

The paper transcription remains valuable as an audit target, but the classic
executable baseline must be reconstructed from the ModelDB files wherever they
specify a value or mechanism. Paper-versus-executable differences remain named
profiles so robustness experiments can test them rather than silently mixing
them.

The executable first-order cell library is also retained separately from the
printed Table 3 library. After converting XML centimeters to Table 3
millimeters, several geometries agree exactly; the relay distal dendrite still
has a distinct resistance and leak, relay/TRN potassium
reversals are -100 mV, and the full-network AHP profile is 5/20 ms rather than
the dedicated cholinergic demonstration's 80/150 ms.
