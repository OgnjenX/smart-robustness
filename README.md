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
and [`docs/replication-status.md`](docs/replication-status.md). The machine-readable
requirement audit is
[`docs/validation-results/classic-smart-objective-audit-885.yaml`](docs/validation-results/classic-smart-objective-audit-885.yaml).
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

### Post-2008 intrinsic physiology screen

The Patchseq screen is an **isolated-cell fitting study**, not a new validated
SMART baseline. It is designed to fit one effective Na/K/M parameter vector per frozen
Pvalb/Sst/Vip and L23/L5 stratum, using training recordings only. The current
search, initialization, tolerances, and numerical controls are sealed in
[`protocol 1436`](docs/validation-results/post2008-patchseq-isolated-fit-registration-1436.yaml).
The calibrated classic baseline is unchanged.

#### Current numerical-reliability gate (2026-10-15)

**Both production fitters are stopped. Do not resume them on the strength of
passing unit tests, synthetic equivalence checks, or an available checkpoint.**
The original 20 completed training-candidate rows are preserved. A compiled
transition stopped before adding any new candidates because candidate 19's
retained score could not be reconstructed within the sealed tolerance.

The [verified mismatch diagnosis](docs/validation-results/post2008-patchseq-mismatch-localization-assessment-1482.yaml)
localized the disagreement upstream of scoring: both feature extractors and
scorers agreed on identical predicted voltages, but different solvers and time
steps produced materially different trajectories. On one fixed positive input,
Brian2 detected six events at the original step and five at half-step. Half-step
waveforms still disagreed across engines. This is numerical-reliability evidence
for that candidate, not a conclusion that SMART or HH neurons fail.

The next gate is the separately registered
[fixed-input reference-convergence study](docs/validation-results/post2008-patchseq-reference-convergence-registration-1483.yaml):
the same candidate and two previously diagnosed commands, two adaptive solvers
at three preset accuracy levels, and three further RK4 refinements. The
[independent terminal audit](docs/validation-results/post2008-patchseq-reference-audit-registration-1485.yaml)
reconstructs saved-array features, losses, input-edge accounting and convergence
decisions. It does not rerun the solvers. Neither a solver's successful exit nor
agreement on spike counts alone establishes a qualified reference.

```bash
# Diagnostic only: first verify that no identical runner is active.
# Resume requires unchanged source/input/build hashes; never overwrite orphan arrays.
MPLCONFIGDIR=/private/tmp/mpl HDF5_PLUGIN_PRELOAD=:: .venv/bin/python scripts/run_patchseq_reference_convergence.py

# Only after that diagnostic is terminal and its exclusive lock is released:
HDF5_PLUGIN_PRELOAD=:: .venv/bin/python scripts/verify_patchseq_reference_convergence.py
```

No diagnostic result automatically authorizes a production time-step change,
parameter replacement, fitting restart, held-out response scoring, or network
promotion. Unqualified reference comparisons remain unresolved evidence; the
old scores and failed transition must not be discarded or silently repaired.

#### Historical fitting and verification routes

The commands below are retained for reproducibility, **not current execution
instructions**. Their terminal fitting prerequisites are not currently met.
They require the pinned local recording artifacts and exact runtime versions
in protocol 1436. Raw recordings/results are ignored and are not distributed
with the repository:

```bash
# Historical original fitter; currently stopped pending a separate numerical decision.
MPLCONFIGDIR=/private/tmp/mpl HDF5_PLUGIN_PRELOAD=:: .venv/bin/python scripts/run_patchseq_isolated_fit.py --output results/patchseq-isolated-fit-1438

# Only after the main runner is terminal-complete:
.venv/bin/python scripts/verify_patchseq_fit_checkpoint_structure.py

# Only after that terminal source/structure audit succeeds:
.venv/bin/python scripts/verify_patchseq_isolated_fit_replay.py

# Only after every independent candidate reconstruction agrees:
.venv/bin/python scripts/verify_patchseq_selected_fit.py

# After terminal verification: record training classifications, not promotion.
.venv/bin/python scripts/assess_patchseq_training.py
```

The first audit checks completeness, hashes, source identities, accounting and
selection; it does not verify simulated physiology. The next independently
reconstructs every training candidate's forward response, features and losses.
The final stage compares the selected waveforms/features and independently
checks exact repeats, time-step refinement and initialization sensitivity.
Each stage retains failures and disagreements; none permits refitting,
candidate replacement, wider searches, or weakened gates after outcomes.
If the independent table disagrees, the assessment records unresolved
verification without requiring or running the selected stage. If it agrees,
the assessment requires complete selected-stage evidence and separates numerical
failure, observable-fit failure, and eligibility for a held-out study design.

An explicitly separate historical verifier-only performance route is sealed in
[`amendment 1457`](docs/validation-results/post2008-patchseq-compiled-verifier-amendment-1457.yaml).
It uses the same independent equations and gates with a retained, fingerprinted
compiled binary. The original NumPy route remains available. After the same
terminal source/structure audit, choose **one** full-table verification route;
do not run both concurrently:

```bash
.venv/bin/python scripts/verify_patchseq_compiled_fit.py
.venv/bin/python scripts/verify_patchseq_selected_compiled_table.py
.venv/bin/python scripts/assess_patchseq_compiled_training.py
```

These commands use distinct result directories and explicit backend matching.
The selected waveform comparison still uses its original Brian2 and NumPy
engines. This verifier-only route is distinct from the later failed compiled
fitting transition. Its existence does not establish production-backend
equivalence, authorize either fitter to restart, or permit a compiled fitter
to use its own kernel as its sole independent forward verifier. Numerical
disagreement is retained rather than resolved by changing scientific parameters.

**Even successful training verification does not authorize held-out response
scoring or network promotion.** Those require a separately sealed access and
assessment step. Earlier outcome-independent source QC is distinct from
held-out response features used for model validation. A sparse stratum cannot
become population-wide evidence through pooling. Experimental holding current
is a conditional isolated-fit input, not extra background drive in SMART.

Long computations are observed at the user's scheduled interval; an observation
timeout is not permission to restart. Unit tests, CI and consistent checkpoints
are engineering evidence, not proof that the scientific program is complete.

### Circuit and model layout

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

The complete SMART builders accept an explicit `population_factory` and pass
the same source-backed population names, sizes, parameters, receptor ports, and
Brian2 context to every cell population in V1 and V2. The default factory is
the source-constrained vectorized multicompartment HH kernel used by the frozen
control. A robustness backend must return the same population/port adapter
interface, allowing the circuit topology, synapses, stimuli, seeds, learning,
and analyses to remain fixed while only the cellular equations change.
The Figure 6, 7, 10, 14, 15, and 16 validation entry points forward the same
factory, so alternative models are scored by the frozen protocols rather than
by model-specific copies of those experiments.

`models.create_population(...)` remains the registry for the reduced benchmark
runner. Its `adex`, `gif`, and alternative HH adapters remain explicit planned
backends rather than silently approximated aliases; complete SMART uses the
separate model-neutral registry described below. See
[`docs/adr/0002-full-network-population-factory.md`](docs/adr/0002-full-network-population-factory.md)
for the full-network substitution contract.

The complete SMART network now has a separate model-neutral registry. Its first
alternative is a morphology-preserving somatic AdEx substitution: dendrites,
compartment-targeted ports, T-type calcium, AHP/ACh, learning, topology, and
protocols stay fixed while the somatic Na/K spike generator changes. Phase 1
and its network-blind fitting rule are preregistered in
[`configs/robustness/adex_phase1_v1.yaml`](configs/robustness/adex_phase1_v1.yaml).

The first controlled AdEx comparison is now recorded. The fixed literature arm
and the per-class direct-current-matched arm both fail all frozen Figure 6
gates. A separately registered relay-transfer calibration restores the cortical
feedforward chain, causal learning timing, and positive top-down contrast, but
produces spatially diffuse relay activity and therefore fails the three relay
gates. No failed arm is passed into downstream match/mismatch protocols, and no
network outcome is used to retune its parameters. See validation artifacts
891, 897, 903, 905, and 907 for the complete positive and negative record.

The stochastic GIF comparison is also complete through its registered Figure 6
prerequisite. It uses the Pozzorini--Mensi GIF formulation with spike-triggered
current, moving threshold, reset/refractoriness, and escape-rate firing. Both
the fixed literature transfer and a network-blind per-class current-step fit
were evaluated over the same 20 frozen seeds. Neither produced a valid Figure 6
learned state in any seed. The matched arm preserved positive top-down contrast
in 18/20 trials but did not preserve the selective repeated relay volleys,
layer-4-led cortical chain, or causal learning pair. It therefore does not enter
the downstream match/mismatch, reset, or spectral protocols. See the registered
design and result in validation artifacts 908--911.

The follow-up population-localization study keeps the calibrated thalamus
classic while replacing all cortical, excitatory-only, or inhibitory-only
somatic spike generators. None of the frozen current-step-matched AdEx arms
passed Figure 6; the corresponding GIF arms each scored 0/20 complete
successes. This rejects those registered parameterizations, not the AdEx/GIF
families in general and not all non-HH explanations of ART. A rebound-matched
relay/TRN arm was withheld because the classic relay failed the independently
registered source-port rebound target, although TRN passed. See validation
artifacts 912 and 915--919.

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

The registered neuron-model and compartment/mechanism comparison is closed in
[`assessment 936`](docs/validation-results/neuron-model-mechanism-phase-assessment-936.yaml).
That finite study does not establish that ART requires HH neurons or that
AdEx/GIF cannot support ART under other independently specified mappings.
Qualitative match/mismatch preservation and exact calibrated-gate survival
remain distinct outcomes.

The ongoing contemporary-anatomy phase is deliberately controlled:

- preserve both immutable classic releases and the failed Figure 15 numeric
  holdout;
- test direct visual thalamus-to-L5 input, active apical integration, updated
  pulvinar routing, and PV/SST/VIP-like inhibitory specificity one module at a
  time, with independently justified parameter ranges;
- distinguish an assessed negative candidate from a completed broader anatomy
  question, retaining source/representation boundaries and untested endpoints;
- preserve the registered stimuli, learning, seeds, analyses, and acceptance
  gates; change topology or receptor placement only when it is the explicitly
  registered intervention;
- test only eligible, separately sealed interactions after single-module
  characterization, then complete multi-seed confirmation and staged releases.

The [sequential program plan](docs/sequential-robustness-program-plan.md) maps
completed assessments and remaining work. The SST-like local-synchrony runner
is authorized by its separate
[pre-outcome execution seal](docs/validation-results/post2008-l5-sst-like-stage4c-seal-1015.yaml);
authorization or a running process is not evidence that its scientific gates
passed. No later phenotype pass may rescue an earlier failed prerequisite.

Contributions should preserve the distinction between **published**, **derived**,
**calibrated**, and **exploratory** parameters.
