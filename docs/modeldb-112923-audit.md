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
compartments when the external grids are instantiated as 9×9. It contains 56
serialized channel projections; the full two-area file contains 24 populations.

## Resolved equation ambiguities

- Sodium activation uses the standard Traub–Miles coefficient `0.32`, not the
  paper's printed `0.032`; sodium inactivation uses a 17 mV offset, not 27 mV.
- KInNeSS defines zero on the voltage-gate axis as the compartment's resting
  potential. The classic network therefore evaluates Na/K rates using
  `V_membrane - E_leak`, rather than a cell-independent 67 mV shift. T-type
  gates retain their separately serialized absolute-voltage formulas.
- For relay T-current, the XML marks the sigmoid with `V0=-63, B=7.8` as
  `m_inf`, while `Simple_Tau(A=2.44, B=2.506, V0=-9.84)` is the activation time
  constant. Likewise, the `V0=-83.5, B=-6.3` sigmoid is `h_inf`, while the
  `A=19.15, B=7.171, V0=-10.54` expression is the inactivation time constant.
  The paper labels these roles in the opposite order and prints 19.5 rather
  than the executable's 19.15.
- The contemporaneous KInNeSS paper defines `Simple tau` as
  `A + B*10^-2*exp(C*10^-2*V)`, confirming the ModelDB time-constant
  transcription. It also supplies the exact directional inter-compartment
  Equation 7, now available as the named `kinness_2008` axial convention.
- `SMART.nml` serializes `inpResistance` on child compartments while root
  somata omit it. The `kinness_serialized_edge` convention therefore applies
  the child connection value in both current directions, preserving the XML
  topology rather than manufacturing a root parameter.
- The executable first-order area contains 55 projected gates after excluding
  the self-triggered AHP gate: 51 chemical and four gap-junction records. This
  set is not identical to the 55-row supplementary catalog; it includes
  narrow/wide channels and different weights, asymptotes, delays, and topology
  flags. A generated, integrity-pinned ModelDB catalog is therefore the
  executable baseline, while the supplement remains an independent audit.
- Chemical-channel `g_bar` is an individual-channel conductance in pS, while
  each serialized projection weight is receptor density in millions/cm².
  Executable ligand conductance density therefore includes the required
  `10^-3` conversion from `pS × 10^6/cm²` to `mS/cm²`; input gates remain
  conductance densities and do not use this receptor-density conversion.
- Ten voltage-driven input channels now follow KInNeSS Equations 4--5: their
  four serialized sensitivities shift the effective driving potential from
  red/green/blue/alpha source values in [0,255]. The eleventh external channel
  is a direct current-injection gate governed by Equation 6. It is compiled
  separately from conductance inputs; all four archived sensitivities are zero,
  so it remains inert until explicitly configured by an experiment protocol.
- Gap-junction totals follow KInNeSS Equation 8 rather than treating XML
  `g_bar` as an already converted membrane density.
- KInNeSS serializes compartment dimensions in centimeters. The executable
  runtime converts those values to the internal millimeter representation;
  for example, the SMART relay soma `0.005×0.006` cm is the Table 3
  `0.05×0.06` mm soma. Treating the raw XML values as millimeters leaves total
  membrane-current ratios unchanged but makes axial coupling 100-fold too
  strong relative to membrane current and capacitance.
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
