# Primary sources

## Baseline article

Grossberg, S. and Versace, M. (2008). *Spikes, synchrony, and attentive learning
by laminar thalamocortical circuits*. Brain Research 1218, 278-312.
DOI: [10.1016/j.brainres.2008.04.024](https://doi.org/10.1016/j.brainres.2008.04.024).
[Author-hosted manuscript](https://sites.bu.edu/steveg/files/2016/06/GroVer2008BR.pdf).

Visual inspection of Methods 4.4, 4.9, and 4.12 confirms that the article
defines the preceding-sample spike expression and 100/1000-ms protocol epochs,
but does not report the numerical integration timestep. The KInNeSS manual
stores timestep as a user preference, and the recovered NeuroML does not carry
that preference. The repository's 0.01-ms default is therefore a converged,
explicit reconstruction parameter rather than a published SMART value
(`validation-results/spike-event-timestep-054.yaml`).

## Original supplementary parameter table

Elsevier attachment:
[1-s2.0-S0006899308008883-mmc1.doc](https://ars.els-cdn.com/content/image/1-s2.0-S0006899308008883-mmc1.doc).

The main article refers to this attachment as "Supplementary Table 4". The
attachment itself captions the connection table as "Supplementary Table 3".
This repository records both names so future searches do not mistake them for
different sources.

The supplement supplies projection-specific receptor types, reversal
potentials, maximal channel conductances, receptor-density weights, Gaussian
spreads, axonal delays, rise/fall constants, plastic-weight annotations, and
transmitter-depletion annotations for the first-order thalamocortical loop.

Source documents are used for transcription and verification but are not
redistributed in this repository.

## Original executable-model backup (recovered)

[ModelDB accession 112922](https://modeldb.science/112922) records the SMART
implementation and points to alternate version
[112923](https://modeldb.science/112923), labeled “backup of web link.” The
backup remains downloadable at
`https://modeldb.science/download/112923` and contains:

- `SMART.nml`, the two-area KInNeSS network;
- `Ca_rebound.xml`, the dedicated thalamic rebound network used for Figure 8;
  it contains Reticular, PBN, Relay, and Layer VI populations but no serialized
  current-pulse or voltage-clamp experiment schedule;
- `Layer_5_and_Maynert_AHP_ACh.nml`, the dedicated AHP/ACh network;
- horizontal and vertical stimulus files and the author README.

Pinned SHA-256 hashes are recorded in `models/modeldb112923.py`. The archive's
README contains a copyright notice but no explicit redistribution license, so
the raw files are downloaded into the ignored source-audit directory and are
not vendored. Extracted model facts and source hashes are reproducible without
republishing the archive.

## Contemporaneous simulator specification

Versace, M., Ames, H., Léveillé, J., Fortenberry, B., and Gorchetchnikov, A.
(2008). *KInNeSS: A modular framework for computational neuroscience*.
Neuroinformatics 6, 291–309. DOI:
[10.1007/s12021-008-9021-2](https://doi.org/10.1007/s12021-008-9021-2).
The [Boston University manuscript](https://open.bu.edu/items/0e83be53-0aa9-4e52-90a0-d4c071205d62)
defines the exact KInNeSS axial Equation 9, `Simple tau` function, normalized
dual-exponential synapse, transmitter depletion, and fourth-order Runge–Kutta
integration used to interpret the SMART XML. The audited PDF has SHA-256
`0f445537cc2e47a21c525f9cbd59cb5d7bd56d86bd195d0a772a4413c522024c`.

The expanded BU manuscript at the same repository item spells out learning
Equations 25--28, including the negative equilibrium scale
`D = -w0 / w_max` when the minimum is zero and the pre-, post-, and dual-gated
extensions. The locally audited download has SHA-256
`15a7f9166e301c9740e7cff4258810d315bd95b196ea5dff3419c5cc2879f177`.
Visual checks of its Equations 2 and 9 independently show the compartment
membrane area as `pi*D*L`. This confirms that KInNeSS uses cylindrical lateral
area for capacitance and current-density conversion and excludes circular end
caps (`validation-results/kinness-membrane-area-049.yaml`).

The archived 2008 download page identifies KInNeSS 0.3.4 RC2 and SANNDRA 1.2.0
RC3 as the contemporaneous releases. A newly recovered raw capture gives the
exact linked filenames `KInNeSS-0.3.4-RC2.tar.gz` (CVS tag
`KINNESS_0_3_4_RC2`) and `SANNDRA-1.2.0-RC3.tar.gz` (CVS tag
`SANNDRA_1_2_0_RC3`); the page hash and archive queries are recorded in
`validation-results/legacy-snapshot-recovery-107.yaml`. The Internet Archive preserves the
KInNeSS CVS2HTML directory index, release metadata, and SANNDRA revision
history, but not the GPL source archives or linked SANNDRA API pages. The
revision history independently records for `gates.h` revision 1.17,
`layer.h` revision 1.9, and `unit.cpp` revision 1.29 that `TGate.init()` was
changed to reset to resting potential. Together with the gate equations, this
supports initializing activation and inactivation variables at their
steady-state occupancy at each compartment's resting voltage. A literal
all-zero gate state is retained only as an audit alternative. No simulator
source code is copied into this repository.

The same revision history lists `spikeevents.h` revisions 1.1--1.7 and says
revision 1.2 (2005-03-07) “Fixed proper spike detecting”, but it does not expose
the file body or diff. Consequently the printed `V(t-dt)` spike expression can
be tested literally, but cannot by itself establish whether the legacy event
handler latched a peak or imposed threshold hysteresis.

A later public build-log trace independently reports a SANNDRA 1.0.0 download
from the KInNeSS site and exposes original header/source names through compiler
diagnostics. It contains no source bodies and cannot resolve the missing spike
event handler. The exact evidence and inference boundary are recorded in
`validation-results/legacy-source-recovery-followup-198.yaml`; no code from
that trace is treated as the legacy implementation.

The same user manual distinguishes the configured leakage equilibrium from a
compartment's actual resting potential: voltage-gated channels can move the
latter away from the former. Consequently `E_leak`, the initialized membrane
voltage, and an ionic gate's resting state are tracked as separate provenance
concepts; a source-unreported settling phase cannot be assumed silently.

The archived manual's Soma editor section also resolves the source compartment
for chemical projections. KInNeSS converts somatic potential to the binary
spike signal consumed by chemical synapses; that conversion and delayed release
occur in an automatically present axon. The `Output Spikes To File` option only
records axon activity. This explains the newer `SMART.nml` combination of
`monitorSpikes=true` on each soma and no explicit `axon` attribute, while the
older `Ca_rebound.xml` still writes `axon=true`. Dendritic thresholding is
therefore not a source-faithful way to rescue TRN recruitment
(`validation-results/kinness-axon-source-052.yaml`).

The 2008 KInNeSS site was actually served at `symphony.bu.edu`; `kinness.net`
was a frame redirect. Its archived User Manual remains available and resolves
the ordinary `connectFromMany` rule more precisely than the article: `Weight`
is the peak of the Gaussian, and shoulders whose resulting weight is below
0.001 are omitted. The locally audited `neteditor.html` capture has SHA-256
`6bf71e0aee1720694ca4c0942068fc7b73258c92f3a29cf7e5a4d6757c77afcd`.
The manual does not explain the separate `Kernel ring` XML flag.

## Known source ambiguity

The article is internally inconsistent about T-type calcium density. Table 3
prints compartment-specific values (including 10 mS/cm² in relay and TRN
dendrites), whereas Methods 4.6 states after Equation 27 that every neuron with
calcium current uses 250 mS/cm². Both are executable named conventions; neither
is silently rewritten into the other. The recovered `SMART.nml` supplies a
third set of cell-specific densities, including 100 mS/cm² in all three TRN
compartments.

Table 3 reports a cell-specific `E_L` and identifies it as the leakage-current
equilibrium potential. The Methods 4.5 text surrounding Equation 20 also states
`E_leak = 0 mV` while writing the leak current as proportional to `-V`. The
implementation currently treats the Table 3 values as physical leak reversals
and records the voltage-coordinate reconciliation as derived, pending full
behavioral validation. The recovered ModelDB files and KInNeSS framework paper
now resolve the serialized equation roles and axial formula, while the legacy
Figure 8 leak/capacitance defaults and clamp semantics remain open; see
`modeldb-112923-audit.md`.

The Internet Archive preserves the 2008 KInNeSS KBrain CVS file-list page,
including `kgatesparse.cpp`, `ksimparse.cpp`, and `kunit.cpp`, but no source-file
captures have been recovered. SANNDRA's surviving revision history resolves
ionic-gate initialization, while the simulator's unrecorded specific
capacitance remains a named runtime convention and may not be silently fitted.
The archive also preserves the SANNDRA 1.2.0 Doxygen main page, but none of its
linked class or source pages. An exact Software Heritage origin search likewise
found no relevant SANNDRA/KInNeSS source tree, so the `spikeevents.h` temporal
algorithm remains unavailable even though the manual now fixes its input to
somatic voltage.

KInNeSS Equation 27 introduces a further executable ambiguity for the archived
adaptive projections. The framework article defines the interface Transition
Time through `A=-1/T` and independently fixes `C=-0.04D` (a 25-ms depressive
tail), while `SMART.nml` serializes a field named `depotentiationLength` with
values 20/25 and the SMART paper prints a literal 0.1-ms transition followed by
a 25-ms tail. Both interpretations remain named runtime conventions. Artifact
155 also proves that the Figure 6c colorbar endpoint cannot be interpreted as
an attainable single-synapse value under the printed Equation 5 and
Supplementary Table 3 bounds; the numeric Figure 6 matrix was not published.

A direct Figure 7 audit also retracts the earlier 40/70-Hz nonspecific-thalamus
target. The caption and Results Sections 2.2-2.3 publish only directional
relay, TRN, and nonspecific-thalamus effects. The paper's 40-Hz value calibrates
the Methods 4.9 relay input, while 70-Hz values belong to unrelated protocols.
Artifact 157 records the corrected structural/qualitative contract.

The same primary Figure 7 text states that matched LGN cells receive
simultaneous bottom-up and top-down excitation. A top-down-only lead interval
is therefore retained as a causal diagnostic, not treated as the canonical
published protocol. Artifact 168 applies simultaneous onset after explicit
equilibration when testing the archived TRN calcium reversal.

The recovered Reticular entry also removes a possible compartment-topology
ambiguity. Its `CableNeuron` structure is explicitly `linear`, with ordered
substructures Soma, Dendrite 0, and Dendrite 1. Only Soma serializes
`monitorSpikes=true`; both dendrites serialize `monitor=false`. Together with
the manual's somatic-to-axon event description, this fixes adjacent chain
coupling and somatic chemical output. Artifact 172 records why neither a star
topology nor dendritic event emission is an admissible calibration route.
