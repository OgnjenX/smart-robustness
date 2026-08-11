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

## Original executable-model record

[ModelDB accession 112922](https://modeldb.science/112922) records the SMART
implementation as a KInNeSS/NeuroML model. Its preserved author-era page links
to `Brain_Research_Paper_KINNESS_SMART_network.rar`, described as the network
description and input stimuli used for the article. The surviving
[ModelDB GitHub mirror](https://github.com/ModelDBRepository/112922) contains the
catalog page and assets, but not that executable payload.

The original BU/KInNeSS host is no longer reliably reachable, and an initial
exact-URL Internet Archive query found no archived copy. The implementation
therefore uses the paper and recovered Elsevier supplement as primary sources
while keeping undocumented initial states, seeds, and exact connection
realizations explicitly non-identifiable. If the bundle is later recovered, it
will be audited against the reconstruction and inspected for a model-specific
license before any redistribution.

## Known source ambiguity

Table 3 reports a cell-specific `E_L` and identifies it as the leakage-current
equilibrium potential. The Methods 4.5 text surrounding Equation 20 also states
`E_leak = 0 mV` while writing the leak current as proportional to `-V`. The
implementation currently treats the Table 3 values as physical leak reversals
and records the voltage-coordinate reconciliation as derived, pending
independent validation against the original KInNeSS/NeuroML implementation.
