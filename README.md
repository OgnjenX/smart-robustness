# SMART robustness

An open, reproducible implementation program for testing which predictions of
**Synchronous Matching Adaptive Resonance Theory (SMART)** survive changes in
the microscopic neuron and synapse models.

The central question is:

> Which system-level SMART mechanisms are robust consequences of the laminar
> thalamocortical architecture, and which depend on the original cellular
> implementation?

The primary baseline reference is Grossberg and Versace (2008), *Spikes,
synchrony, and attentive learning by laminar thalamocortical circuits*, Brain
Research 1218, 278–312, [doi:10.1016/j.brainres.2008.04.024](https://doi.org/10.1016/j.brainres.2008.04.024).
An [author-hosted PDF](https://sites.bu.edu/steveg/files/2016/06/GroVer2008BR.pdf)
is available from Stephen Grossberg's publication page.

## Scientific milestones

The first baseline is considered validated only when it reproduces, under
documented stimulus and analysis procedures:

1. a sufficiently good bottom-up/top-down match producing sustained,
   synchronized gamma-band resonance and enabling learning;
2. a sufficiently large mismatch recruiting nonspecific thalamic reset,
   suppressing learning, and producing slower beta-band dynamics;
3. the relevant cell-level firing modes and the direction of the paper's
   ablation/parameter effects.

The source-constrained implementation and the complete first- and higher-order
networks are implemented. The calibrated classic-SMART behavioral baseline is
now frozen as `classic-smart-calibrated-v1.0.0`. It passes the registered
Figure 6 learning, Figure 7 match/mismatch, Figure 10 reset, Figure 14
gamma-versus-slower mismatch, and Figure 16 long-range synchrony gates. It also
reproduces the source-identifiable Figure 15 behavior of nearby layer-4 gamma
synchrony, but its fixed pair peaks at 53.0265 Hz rather than the panel's
graphically labeled 44 Hz. The original pair, spike arrays, estimator, and
legacy simulator runtime were not released. The frozen release is therefore a
**calibrated behavioral reconstruction, not an exact numerical reproduction**.
See [`docs/calibrated-behavioral-freeze-v1.0.0.md`](docs/calibrated-behavioral-freeze-v1.0.0.md)
and [`docs/replication-status.md`](docs/replication-status.md).
The immutable `classic-smart-source-constrained-v0.1.0` tag preserves the first
complete source-constrained reconstruction, including its failed official
reproduction gates. It is a provenance baseline, not a claim that classic
SMART has been reproduced; see
[`docs/source-constrained-freeze-v0.1.0.md`](docs/source-constrained-freeze-v0.1.0.md).
The acceptance criteria are fixed in
[`docs/validation-matrix.md`](docs/validation-matrix.md), and the staged path to
the frozen baseline is in
[`docs/reproduction-roadmap.md`](docs/reproduction-roadmap.md).

## Why Brian2?

Classic SMART uses minimal multi-compartment Hodgkin–Huxley-type cells—not a
plain leaky integrate-and-fire model. Brian2 can state those differential
equations directly, model event-driven conductance synapses and plasticity, and
still lets experiments swap in AdEx, GIF, point HH, or later a detailed
multicompartment backend. The architecture is simulator-light: configurations,
measurements, and validation criteria do not depend on a particular cell model.

## Architecture

```text
configs/                 reproducible experiment definitions
src/smart_robustness/
  models/                swappable neuron registry, Table 3 cells, and Brian2 equations
  projections.py         typed Supplementary Table 3 connection catalog
  data/                  packaged source-backed projection records
  synapses.py            dual-exponential conductance + transmitter depletion
  circuit.py             minimal SMART benchmark assembly
  experiment.py          seeded execution and artifact writing
  validation/first_order.py  fingerprinted relay/bar reproduction runners
  analysis/spectra.py    gamma/beta and synchrony measurements
tests/                   equations, configuration, analysis, optional smoke test
docs/                    provenance ledger and replication roadmap
```

The boundary for neuron-model substitution is `models.create_population(...)`.
Circuit code refers to declared receptor ports (`exc`, `inh`, and `reset`) rather
than embedding a neuron's equations. `classic_hh` is the reference target;
The frozen baseline uses the source-constrained vectorized multicompartment HH
kernel directly. `adex`, `gif`, and registry-level alternative HH adapters
remain explicit planned backends rather than silently approximated aliases.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,plots]'

smart-run configs/match.yaml
smart-run configs/mismatch.yaml
pytest
```

Each run writes a compressed data file plus a JSON summary beneath `results/`.
Set `output_dir` and `seed` in YAML to preserve independent runs. The summary
contains package versions, a configuration fingerprint, rates, dominant
frequency, beta/gamma power, and the predeclared validation checks.

## Experiment contract

All comparisons should keep the circuit, stimuli, random seeds, run duration,
and analysis fixed while changing only `model.name` and its parameters. A model
comparison should report all seeds, not only a representative trace. New
parameters require a source in `docs/parameter-provenance.yaml`; calibrated
values must be labeled as such and must not be described as published values.

## Scope and next steps

The immediate next phase is deliberately controlled:

- preserve both immutable classic releases and the failed Figure 15 numeric
  holdout;
- compare AdEx, GIF, alternative HH, and multicompartment variants against the
  frozen behavioral control, changing one preregistered neuron-model factor at
  a time;
- keep topology, stimuli, learning, projection scales, seeds, analysis, and
  acceptance gates unchanged across the first comparison matrix;
- evaluate post-2008 anatomical revisions only in a later, separately labeled
  robustness phase.

Contributions should preserve the distinction between **published**, **derived**,
**calibrated**, and **exploratory** parameters.
