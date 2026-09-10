# Calibrated classic-SMART behavioral freeze v1.0.0

## Release identity

- Git tag: `classic-smart-calibrated-v1.0.0`
- Frozen model implementation: commit `f6dc9de948586597d13d544a50e2b1bf48ca5514`
- Executable manifest: `configs/baselines/classic_smart_calibrated_v1.yaml`
- Environment lock: `requirements/classic-smart-calibrated-v1.txt`
- Simulator: Python 3.11.9 and Brian2 2.9.0
- Primary reference: Grossberg and Versace (2008)

The tag is the immutable release identifier. The manifest reconstructs the
runtime convention fingerprint, verifies its source profile and all decisive
validation records, and exposes the exact projection-scale tuple used by the
control condition.

## What “frozen” means

The baseline's topology, calibrated parameters, protocols, seeds, analysis,
acceptance gates, expected metrics, and known failures will not be edited in
place. Corrections or new interpretations require a new version. Alternative
neuron experiments must use this release as their unchanged control and alter
only the preregistered neuron-model factor.

This is a **calibrated behavioral reconstruction**, not a bit-identical replay
of the unavailable 2008 KInNeSS/SANNDRA execution and not an exact numerical
reproduction claim.

## Frozen endpoint

The control uses the source-constrained vectorized multicompartment HH kernel,
coupled RK4, the projection-036 variance interpretation, nonspecific thalamic
T-current scale 0.203125, and the complete projection-scale mapping in the
manifest. The effective projection-025 scale 8 and projection-026 scale 0.5 are
calibrated departures and must never be presented as published values.

## Validation outcome

| Target | Frozen result |
|---|---|
| Figure 6 learning and oriented maps | pass |
| Figure 7 match/mismatch and 4/7 nonspecific output | pass |
| Figure 10 causal reset and negative control | pass |
| Figure 14 match gamma versus slower mismatch | pass; 55 Hz versus 10 Hz |
| Figure 15 nearby layer-4 gamma synchrony | source-identifiable phenotype passes |
| Figure 15 graphical 44-Hz value | numeric gate fails; fixed reconstruction is 53.0265 Hz |
| Figure 16 V1–pulvinar–V2 lower-frequency dominance | pass; lower/gamma peak ratio 4.8244 |
| Repository-wide technical verification | 1062 passed, 0 failed |

## Preserved limitations

The 2008 Figure-8 clamp/input schedule and raw voltage trace are not preserved.
The exact normalized 8-bit input hypothesis also fails against the independent
2004 KInNeSS tonic/burst archive. For Figure 15, the publication does not
preserve the original cell identities, spikes, spectral estimator, or
confidence procedure. Registered source-plausible analyses all retain a
52.63–55 Hz peak rather than 44 Hz.

Accordingly, these phrases are not permitted for this release:

- “exact Grossberg–Versace 2008 numerical reproduction”;
- “bit-identical KInNeSS/SANNDRA recovery”;
- “published projection-025 scale 8” or “published projection-026 scale 0.5”.

## Reproduce the frozen environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/classic-smart-calibrated-v1.txt
python -m pip install -e .
pytest
```

The formal machine-readable decision is validation artifact 884. The next
phase compares neuron models against this fixed control. Post-2008 anatomy is a
separate later intervention family and cannot be used to rescue or redefine the
classic baseline.

