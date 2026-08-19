# Classic SMART validation matrix

This document is the acceptance contract for the Grossberg and Versace (2008)
baseline. A result is never called an exact reproduction merely because its
trace looks similar to a published figure.

## Evidence classes

| Class | Meaning |
|---|---|
| `exact-source` | Equations, dimensions, parameters, and protocol values agree with the article or recovered supplement. |
| `structural` | Populations, compartments, projections, and causal mechanisms agree with the published architecture. |
| `qualitative` | The direction and characteristic form of a published effect are reproduced. |
| `approximate-numeric` | A value stated in the article or digitized from a figure is met within a predeclared tolerance. |
| `not-identifiable` | The publication does not contain enough information for a unique numerical target. |

All generated validation records must name the evidence class, source page or
table, configuration fingerprint, simulator version, integration time step,
seed, and tolerance. `not-identifiable` values may be calibrated, but must stay
clearly separate from published values.

## Structural acceptance gates

| Gate | Required baseline property | Primary source |
|---|---|---|
| Network scale | Two six-layer thalamocortical loops. Methods 4.1 and Table 3 imply ten 9×9 sheets plus two singleton nuclei, or 812 cells and 1,950 compartments per sector, matching `SMART.nml`. Methods 4.2 separately prints incompatible totals of 732 cells, 2,106 compartments, and 17,415 differential equations; construction retains the official executable dimensions and reports the anomaly. | Methods 4.1–4.2; Table 3; ModelDB 112923 |
| Cell structure | Published two- or three-compartment cells with passive axial coupling and the Supplementary Table 3 cell classes. | Table 3; Fig. 17; Methods 4.2 |
| First-order loop | LGN core/matrix/interneurons, TRN, nonspecific thalamus, and the complete V1 laminar loop. | Fig. 2; Table 2; supplement |
| Higher-order loop | V1–pulvinar–V2 loop, including higher-order feedback and TRN interactions. | Fig. 3; Table 2 |
| Ionic mechanisms | Traub–Miles-style Na/K currents; T-type calcium in the published thalamic classes; layer-5 AHP and cholinergic modulation. | Methods 4.5–4.7; Figs. 8, 12, 19 |
| Synapses | Normalized dual exponentials, the equal-time-constant alpha special case, compartment targets, delays, spatial kernels, and supplement annotations. | Methods 4.2–4.4; supplement |
| Learning | Postsynaptically gated bottom-up learning and dual-AND gated top-down learning. | Eq. 6; Fig. 6; Methods 4.3 |
| Reset | Mismatch disinhibits nonspecific thalamus; layer-1/layer-5 apical activation propagates through layer 6I to reset layer 4. | Figs. 7 and 10 |

## Published protocol targets

| Target | Protocol | Acceptance criterion | Evidence class |
|---|---|---|---|
| Relay tonic/burst mode (Fig. 8) | Isolated LGN relay cell; 0.3 nA injection under depolarized and hyperpolarized conditions. | Tonic firing in the depolarized condition and a T-current-dependent rebound burst after hyperpolarization. | `qualitative`; exact latency and spike count are `not-identifiable` pending digitization/source recovery. |
| Depletion (Fig. 11) | 150 ms traces; firing regimes 2/3 Hz and 7/12 Hz; `(epsilon, tau)` of `(0.5, 50 ms)`, `(1, 50 ms)`, and `(1, 10 ms)`. | Greater use or epsilon causes greater depletion; faster recovery reduces depletion. | `exact-source` protocol, `qualitative` trace. |
| AHP/ACh (Figs. 12 and 19) | Isolated layer-5 cell; about 80 Hz current-driven firing; 100 ms ACh event; AHP rise/fall 80/100 ms; ACh rise/fall 5/6 ms. | AHP adapts and hyperpolarizes the cell; ACh suppresses AHP and increases excitability; recovery is near complete by about 500 ms. | Mixed `exact-source`, `qualitative`, and digitized `approximate-numeric`. |
| Match/mismatch arousal (Fig. 7) | Horizontal bottom-up plus horizontal top-down match versus vertical bottom-up plus horizontal top-down mismatch. | Nonspecific thalamus is approximately 40 Hz during match and approximately 70 Hz during mismatch. | `approximate-numeric`; default rate tolerance must be declared before tuning. |
| Reset (Fig. 10) | A persistent network establishes a horizontal layer-4 winner, then receives vertical bottom-up plus horizontal top-down mismatch; compare intact nonspecific-thalamus→layer-5 input with a disconnected negative control. | The nonspecific→layer-5→layer-6I chain is recruited; the current layer-4 winner is suppressed more than in the control and more previously inhibited alternatives are released. | `structural` and `qualitative`; latency is `not-identifiable`. |
| Match/mismatch spectra (Fig. 14) | Five-cell bar; 1,000 ms trials; subtract mean; 200 ms Hamming window before Fourier analysis. | Match is gamma-dominant (20–70 Hz); mismatch increases slower activity and reduces gamma. | `qualitative` plus digitized `approximate-numeric`. |
| Local synchrony (Fig. 15) | Nearby V1 layer-4 cells with overlapping receptive fields. | Simulated correlation peak near 44 Hz; comparison experiment near 50 Hz; local range 300 micrometres. | `approximate-numeric`. |
| Long-range synchrony (Fig. 16) | Two thalamocortical areas; 10 ms V1-layer-2/3-to-V2-layer-4 delay; 1 s input before recording. | Long-range coupling is stronger in lower bands than local gamma; bands are 2–4, 4–8, 8–12, 12–20, and 20–100 Hz. | `qualitative` and `exact-source` protocol; amplitudes are `not-identifiable`. The recovered executable record independently serializes 5 ms. |
| Learning (Fig. 6) | Relative spike timing from -30 to +30 ms and horizontal-bar training. | Gating-family timing curves and oriented bottom-up/top-down weight maps have the published direction and selectivity. | `qualitative`; amplitudes require digitization or original source. |

## Declared publication ambiguities

- Methods 4.1 plus Table 3 imply 812 cells and 1,950 compartments per sector,
  matching the official executable archive, whereas Methods 4.2 separately
  prints 732 cells and 2,106 compartments. The archive-consistent dimensions
  control construction; `network-structural-counts-055.yaml` records the full
  reconciliation and preserves the printed totals as a publication anomaly.
- The attachment calls the connection catalog “Supplementary Table 3,” while
  the main article calls it “Supplementary Table 4.” They are the same source.
- Methods 4.10 describes 2–8, 8–10, and 20–70 Hz bands, whereas the Figure 14
  caption uses 2–8, 8–20, and 20–70 Hz. Validation must report both analyses;
  neither definition may be silently selected after looking at results.
- Methods 4.5 writes the leak current in a zero-reversal voltage coordinate,
  while Table 3 lists cell-specific negative leak equilibria. Both conventions
  must be reconciled in an isolated-cell test before network tuning.
- The article provides plotted traces, not raw figure data, exact initial
  states, complete cell placements, or random seeds. Those values are not
  exact-source targets unless recovered from the original KInNeSS archive.

## Claim gate

The repository may use the phrase **classic SMART baseline reproduced** only
after all structural gates pass and a versioned validation report demonstrates
the isolated-cell, learning, match/mismatch arousal, reset, and oscillation
targets. The report must include negative controls, tolerances, all declared
seeds, and failures as well as successes.
