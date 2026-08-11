# Primary sources

## Baseline article

Grossberg, S. and Versace, M. (2008). *Spikes, synchrony, and attentive learning
by laminar thalamocortical circuits*. Brain Research 1218, 278-312.
DOI: [10.1016/j.brainres.2008.04.024](https://doi.org/10.1016/j.brainres.2008.04.024).
[Author-hosted manuscript](https://sites.bu.edu/steveg/files/2016/06/GroVer2008BR.pdf).

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
defines the exact KInNeSS axial Equation 7, `Simple tau` function, normalized
dual-exponential synapse, transmitter depletion, and fourth-order Runge–Kutta
integration used to interpret the SMART XML. The audited PDF has SHA-256
`0f445537cc2e47a21c525f9cbd59cb5d7bd56d86bd195d0a772a4413c522024c`.

The expanded BU manuscript at the same repository item spells out learning
Equations 25--28, including the negative equilibrium scale
`D = -w0 / w_max` when the minimum is zero and the pre-, post-, and dual-gated
extensions. The locally audited download has SHA-256
`15a7f9166e301c9740e7cff4258810d315bd95b196ea5dff3419c5cc2879f177`.

The archived 2008 download page identifies KInNeSS 0.3.4 RC2 and SANNDRA 1.2.0
RC3 as the contemporaneous releases. Their GPL source archives were not
captured by the Internet Archive, but the archived SANNDRA API landing page and
CVS history remain useful corroborating records. No simulator source code is
copied into this repository.

## Known source ambiguity

Table 3 reports a cell-specific `E_L` and identifies it as the leakage-current
equilibrium potential. The Methods 4.5 text surrounding Equation 20 also states
`E_leak = 0 mV` while writing the leak current as proportional to `-V`. The
implementation currently treats the Table 3 values as physical leak reversals
and records the voltage-coordinate reconciliation as derived, pending full
behavioral validation. The recovered ModelDB files and KInNeSS framework paper
now resolve the serialized equation roles and axial formula, while the legacy
Figure 8 leak/capacitance defaults and clamp semantics remain open; see
`modeldb-112923-audit.md`.
