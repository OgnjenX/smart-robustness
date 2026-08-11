"""Source-backed ionic and modulatory equations for classic SMART.

The numeric helpers in this module use the voltage coordinate printed in
Grossberg and Versace (2008), Methods 4.5--4.7.  They deliberately do not apply
the derived ``V + 67 mV`` transform used by the exploratory M0 point-cell
implementation.

The T-type calcium steady-state equations are ambiguous in the paper: Equations
24 and 27 are printed as additive expressions that are greater than one over
physiological voltages.  Both the literal transcription and the commonly
inferred reciprocal interpretation are exposed, and callers must choose one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True)
class EquationSource:
    """Primary-source location and fidelity notes for an equation family."""

    citation: str
    doi: str
    methods_section: str
    equations: str
    article_pages: str
    local_pdf: str
    notes: tuple[str, ...] = ()


GV2008_CITATION = (
    "Grossberg, S. & Versace, M. (2008). Spikes, synchrony, and attentive learning "
    "by laminar thalamocortical circuits. Brain Research, 1218, 278-312."
)
GV2008_DOI = "10.1016/j.brainres.2008.04.024"
GV2008_LOCAL_PDF = "tmp/pdfs/GroVer2008BR.pdf"

TRAUB_MILES_SOURCE = EquationSource(
    citation=GV2008_CITATION,
    doi=GV2008_DOI,
    methods_section="4.5 Ionic currents",
    equations="9-19",
    article_pages="40-41",
    local_pdf=GV2008_LOCAL_PDF,
    notes=(
        "Rates use the voltage coordinate V printed in the equations.",
        "No derived absolute-voltage shift is applied in this module.",
        "Printed Eq. 15 uses 0.032, tenfold below the standard Traub-Miles alpha-m coefficient.",
        "Printed Eq. 18 uses a 27 mV offset; the standard Traub-Miles form uses 17 mV.",
    ),
)

LEAK_SOURCE = EquationSource(
    citation=GV2008_CITATION,
    doi=GV2008_DOI,
    methods_section="4.5 Ionic currents",
    equations="20",
    article_pages="41",
    local_pdf=GV2008_LOCAL_PDF,
    notes=(
        "Equation 20 prints a zero-reversal leak current proportional to -V.",
        "Table 3 separately reports cell-specific leak equilibrium potentials.",
        "The later equation builder must select and record a leak convention.",
    ),
)

T_TYPE_CALCIUM_SOURCE = EquationSource(
    citation=GV2008_CITATION,
    doi=GV2008_DOI,
    methods_section="4.6 Calcium currents in thalamic cells",
    equations="21-27",
    article_pages="41",
    local_pdf=GV2008_LOCAL_PDF,
    notes=(
        "Equations 24 and 27 are printed as additive expressions without reciprocals.",
        "A reciprocal interpretation is exposed separately and is not silently selected.",
    ),
)

AHP_ACH_SOURCE = EquationSource(
    citation=GV2008_CITATION,
    doi=GV2008_DOI,
    methods_section="4.7 Cholinergic modulation and after-hyperpolarization currents",
    equations="3 (reused for AHP) and 28 (ACh complement)",
    article_pages="42-43",
    local_pdf=GV2008_LOCAL_PDF,
    notes=(
        "The AHP maximal conductance is not numerically specified in the Methods text.",
        "Only the published dual-exponential time constants are fixed here.",
    ),
)


# Published reversal potentials and the paper-wide calcium-density statement.
E_K_MV = -90.0
E_NA_MV = 50.0
E_CA_MV = 180.0
G_CA_MSIEMENS_CM2 = 250.0

# Methods 4.7 dual-exponential time constants.
AHP_RISE_MS = 80.0
AHP_FALL_MS = 100.0
ACH_RISE_MS = 5.0
ACH_FALL_MS = 6.0


def biexponential_peak_time_ms(rise_ms: float, fall_ms: float) -> float:
    """Time of the peak of ``exp(-t/fall)-exp(-t/rise)``.

    SMART reuses its dual-exponential synaptic waveform for AHP and ACh.
    Keeping the normalization here makes the event amplitude independent of
    the published time constants and gives the Brian equations an auditable
    scalar rather than an embedded fitted value.
    """

    if rise_ms <= 0 or fall_ms <= 0 or rise_ms >= fall_ms:
        raise ValueError("biexponential constants must satisfy 0 < rise_ms < fall_ms")
    return rise_ms * fall_ms * math.log(fall_ms / rise_ms) / (fall_ms - rise_ms)


def biexponential_normalization(rise_ms: float, fall_ms: float) -> float:
    """Multiplier that gives a unit-height dual-exponential event."""

    peak_ms = biexponential_peak_time_ms(rise_ms, fall_ms)
    unscaled_peak = math.exp(-peak_ms / fall_ms) - math.exp(-peak_ms / rise_ms)
    return 1.0 / unscaled_peak


AHP_NORMALIZATION = biexponential_normalization(AHP_RISE_MS, AHP_FALL_MS)
ACH_NORMALIZATION = biexponential_normalization(ACH_RISE_MS, ACH_FALL_MS)


@dataclass(frozen=True)
class TraubMilesRates:
    """Na/K gate rates in inverse milliseconds."""

    alpha_n: float
    beta_n: float
    alpha_m: float
    beta_m: float
    alpha_h: float
    beta_h: float


class TTypeGateConvention(StrEnum):
    """Interpretation of the steady-state expressions in printed Eqs. 24 and 27."""

    PRINTED_LITERAL = "printed_literal"
    RECIPROCAL = "reciprocal"


class NaKRateConvention(StrEnum):
    """Printed SMART rates versus the standard Traub--Miles rate corrections."""

    PRINTED_SMART = "printed_smart"
    STANDARD_TRAUB_MILES = "standard_traub_miles"


def _x_over_expm1(x: float) -> float:
    """Return x/(exp(x)-1), including its removable singularity at zero."""

    if x == 0.0:
        return 1.0
    try:
        return x / math.expm1(x)
    except OverflowError:
        if x > 0.0:
            return 0.0
        raise


def _inverse_one_plus_exp(x: float) -> float:
    """Stable evaluation of 1/(1+exp(x))."""

    if x >= 0.0:
        try:
            exponential = math.exp(-x)
        except OverflowError:  # pragma: no cover - exp(-x) cannot overflow for x >= 0
            return 0.0
        return exponential / (1.0 + exponential)
    try:
        return 1.0 / (1.0 + math.exp(x))
    except OverflowError:  # pragma: no cover - exp(x) cannot overflow for x < 0
        return 1.0


def alpha_n_per_ms(v_mV: float) -> float:
    """Potassium activation alpha rate from printed Eq. 11."""

    return 0.16 * _x_over_expm1((15.0 - v_mV) / 5.0)


def beta_n_per_ms(v_mV: float) -> float:
    """Potassium activation beta rate from printed Eq. 12."""

    return 0.5 * math.exp((10.0 - v_mV) / 40.0)


def alpha_m_per_ms(v_mV: float) -> float:
    """Sodium activation alpha rate from printed Eq. 15."""

    return 0.128 * _x_over_expm1((13.0 - v_mV) / 4.0)


def beta_m_per_ms(v_mV: float) -> float:
    """Sodium activation beta rate from printed Eq. 16."""

    return 1.4 * _x_over_expm1((v_mV - 40.0) / 5.0)


def alpha_h_per_ms(v_mV: float) -> float:
    """Sodium inactivation alpha rate from printed Eq. 18."""

    return 0.128 * math.exp((27.0 - v_mV) / 18.0)


def beta_h_per_ms(v_mV: float) -> float:
    """Sodium inactivation beta rate from printed Eq. 19."""

    return 4.0 * _inverse_one_plus_exp((40.0 - v_mV) / 5.0)


def traub_miles_rates(
    v_mV: float,
    convention: NaKRateConvention = NaKRateConvention.PRINTED_SMART,
) -> TraubMilesRates:
    """Return Na/K rates under a declared SMART or standard-Traub convention."""

    if not isinstance(convention, NaKRateConvention):
        raise TypeError("convention must be an explicit NaKRateConvention member")
    alpha_m = alpha_m_per_ms(v_mV)
    alpha_h = alpha_h_per_ms(v_mV)
    if convention is NaKRateConvention.STANDARD_TRAUB_MILES:
        alpha_m *= 10.0
        alpha_h = 0.128 * math.exp((17.0 - v_mV) / 18.0)

    return TraubMilesRates(
        alpha_n=alpha_n_per_ms(v_mV),
        beta_n=beta_n_per_ms(v_mV),
        alpha_m=alpha_m,
        beta_m=beta_m_per_ms(v_mV),
        alpha_h=alpha_h,
        beta_h=beta_h_per_ms(v_mV),
    )


def t_type_tau_m_ms(v_mV: float) -> float:
    """T-type activation time constant from printed Eq. 23, in milliseconds."""

    return _inverse_one_plus_exp((-63.0 - v_mV) / 7.8)


def t_type_tau_h_ms(v_mV: float) -> float:
    """T-type inactivation time constant from printed Eq. 26, in milliseconds."""

    return _inverse_one_plus_exp((-83.0 - v_mV) / 6.3)


def _require_t_type_convention(convention: TTypeGateConvention) -> TTypeGateConvention:
    if not isinstance(convention, TTypeGateConvention):
        raise TypeError("convention must be an explicit TTypeGateConvention member")
    return convention


def _t_type_m_literal(v_mV: float) -> float:
    return 2.44 + 2.506e-2 * math.exp(-9.84e-2 * v_mV)


def _t_type_h_literal(v_mV: float) -> float:
    return 19.5 + 7.171e-2 * math.exp(-10.54e-2 * v_mV)


def t_type_m_inf(v_mV: float, convention: TTypeGateConvention) -> float:
    """Eq. 24 under a required literal or whole-expression reciprocal convention."""

    selected = _require_t_type_convention(convention)
    literal = _t_type_m_literal(v_mV)
    if selected is TTypeGateConvention.PRINTED_LITERAL:
        return literal
    return 1.0 / literal


def t_type_h_inf(v_mV: float, convention: TTypeGateConvention) -> float:
    """Eq. 27 under a required literal or whole-expression reciprocal convention."""

    selected = _require_t_type_convention(convention)
    literal = _t_type_h_literal(v_mV)
    if selected is TTypeGateConvention.PRINTED_LITERAL:
        return literal
    return 1.0 / literal


# Brian2-ready fragments. ``v_paper`` is intentionally an external voltage so
# the later equation builder must make its coordinate transform explicit.
TRAUB_MILES_EQUATIONS = r"""
i_k = g_k*n**4*(e_k-v_membrane) : amp
dn/dt = alpha_n*(1-n) - beta_n*n : 1
alpha_n = 0.16/exprel((15*mV-v_paper)/(5*mV))/ms : Hz
beta_n = 0.5*exp((10*mV-v_paper)/(40*mV))/ms : Hz
i_na = g_na*m**3*h*(e_na-v_membrane) : amp
dm/dt = alpha_m*(1-m) - beta_m*m : 1
alpha_m = 0.128/exprel((13*mV-v_paper)/(4*mV))/ms : Hz
beta_m = 1.4/exprel((v_paper-40*mV)/(5*mV))/ms : Hz
dh/dt = alpha_h*(1-h) - beta_h*h : 1
alpha_h = 0.128*exp((27*mV-v_paper)/(18*mV))/ms : Hz
beta_h = 4/(exp((40*mV-v_paper)/(5*mV))+1)/ms : Hz
"""

LEAK_CURRENT_EQUATION = r"""
i_leak_printed = -g_leak_channel*leak_density*v_membrane : amp
"""


def t_type_calcium_equations(convention: TTypeGateConvention) -> str:
    """Return Brian2-ready Eqs. 21-27 under an explicit gate convention."""

    selected = _require_t_type_convention(convention)
    m_literal = "2.44 + 2.506e-2*exp(-9.84e-2*v_paper/mV)"
    h_literal = "19.5 + 7.171e-2*exp(-10.54e-2*v_paper/mV)"
    if selected is TTypeGateConvention.RECIPROCAL:
        m_expression = f"1/({m_literal})"
        h_expression = f"1/({h_literal})"
    else:
        m_expression = m_literal
        h_expression = h_literal
    return f"""
i_ca = g_ca*m_ca**3*h_ca*(e_ca-v_membrane) : amp
dm_ca/dt = (m_ca_inf-m_ca)/tau_m_ca : 1
tau_m_ca = 1/(exp((-63*mV-v_paper)/(7.8*mV))+1)*ms : second
m_ca_inf = {m_expression} : 1
dh_ca/dt = (h_ca_inf-h_ca)/tau_h_ca : 1
tau_h_ca = 1/(exp((-83*mV-v_paper)/(6.3*mV))+1)*ms : second
h_ca_inf = {h_expression} : 1
"""
