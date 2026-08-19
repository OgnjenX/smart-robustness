# Classic SMART source-constrained freeze v0.1.0

## Release identity

- Git tag: `classic-smart-source-constrained-v0.1.0`
- Intended branch point for calibration: this tag
- Language/simulator: Python 3.11 and Brian2 2.9
- Primary reference: Grossberg and Versace (2008)
- Executable reference: ModelDB 112923 `SMART.nml`
- Legacy framework references: surviving KInNeSS/SANNDRA documentation and revision metadata

The annotated Git tag is the authoritative content identifier. A checkout of
that tag, rather than a moving branch name, defines this release.

## Meaning of “frozen”

This tag is immutable. Source facts, equations, conventions, tests, and
validation artifacts at the tag will not be edited in place. Corrections are
new commits and releases. Calibration starts from the tag on a separate branch
and must label every inferred value.

This is a **source-constrained reconstruction freeze**, not a validated
behavioral freeze and not proof of bitwise equivalence to KInNeSS 0.3.4 RC2.

## Included implementation

- all 12 printed Table 3 cell classes and the recovered executable cell profiles;
- vectorized two- and three-compartment membrane dynamics;
- source-labeled Na/K and T-type calcium families;
- layer-5 AHP and acetylcholine mechanisms;
- complete first-order 9×9 and higher-order V1–pulvinar–V2 projection catalogs;
- dual-exponential/alpha synapses and continuous transmitter depletion;
- bottom-up and top-down STDP rules;
- Figure 6, 7, 10, and 14–16 protocols and analyses;
- parameter fingerprints, predeclared scorers, tests, and validation records.

## Current official-result assessment

| Target | Frozen result |
|---|---|
| Figure 6 bottom-up orientation | qualitative pass |
| Figure 6 cortical chain/top-down map | fail |
| Figure 7 directional arousal split and overlap-only mismatch | fail |
| Figure 10 causal reset chain | fail |
| Figure 14 match gamma | partial pass |
| Figure 14 mismatch slowing/gamma reduction | fail |
| Figure 15 approximately 44 Hz local peak | fail |
| Figure 16 lower-frequency long-range dominance | qualitative pass, underidentified |

The current evidence is summarized in validation artifacts 098–113 and
`docs/replication-status.md`. Failed gates are retained as results; they are not
converted into passes by relaxing criteria after inspection.

## Unresolved authoritative gaps

- exact KInNeSS ring stencil and spike-event detector behavior;
- complete membrane/gate initialization and simulator-wide numerical defaults;
- original learned arrays and persistent initial state;
- electrode placements and regional reductions for exact spectral traces.

The historical download page confirms the unavailable release names
`KInNeSS-0.3.4-RC2.tar.gz` and `SANNDRA-1.2.0-RC3.tar.gz`; their source bodies
have not been recovered.

## Claim boundary

Permitted description:

> Source-constrained Brian2 reconstruction of Grossberg–Versace SMART with
> documented reproduction failures and explicit legacy ambiguities.

Not permitted: exact reproduction, verified original SMART implementation, or
validated classic behavioral baseline.

Calibration may produce a separately versioned behavioral baseline, but it
must never retroactively change this tag or relabel inferred parameters as
published values.

## Reproduction commands

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,plots]'
pytest
```

Official figure simulations are represented by fingerprinted records in
`docs/validation-results/`; several full runs require minutes per condition and
a C++ toolchain.
