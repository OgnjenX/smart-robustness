from __future__ import annotations

from typing import Any

from .table3 import get_cell_spec

# Traub--Miles-style Na/K kinetics transcribed from Grossberg & Versace (2008),
# Methods 4.5. This M0 population is a point-cell reference component; the full
# baseline will assign the same channel family to paper-specific compartments.
EQUATIONS = r"""
dv/dt = (g_na*m**3*h*(e_na-v) + g_k*n**4*(e_k-v) + g_l*(e_l-v)
         + g_exc*(e_exc-v) + g_inh*(e_inh-v) + i_drive) / C_m : volt
dm/dt = alpha_m*(1-m) - beta_m*m : 1
dh/dt = alpha_h*(1-h) - beta_h*h : 1
dn/dt = alpha_n*(1-n) - beta_n*n : 1
v_rate = v + 67*mV : volt
alpha_m = 1.28/exprel((13*mV-v_rate)/(4*mV))/ms : Hz
beta_m = 1.4/exprel((v_rate-40*mV)/(5*mV))/ms : Hz
alpha_h = 0.128*exp((17*mV-v_rate)/(18*mV))/ms : Hz
beta_h = 4/(1+exp((40*mV-v_rate)/(5*mV)))/ms : Hz
alpha_n = 0.16/exprel((15*mV-v_rate)/(5*mV))/ms : Hz
beta_n = 0.5*exp((10*mV-v_rate)/(40*mV))/ms : Hz
g_exc = norm_exc*(g_exc_decay-g_exc_rise) : siemens
dg_exc_rise/dt = -g_exc_rise/tau_exc_rise : siemens
dg_exc_decay/dt = -g_exc_decay/tau_exc_decay : siemens
g_inh = norm_inh*(g_inh_decay-g_inh_rise) : siemens
dg_inh_rise/dt = -g_inh_rise/tau_inh_rise : siemens
dg_inh_decay/dt = -g_inh_decay/tau_inh_decay : siemens
i_drive : amp
C_m : farad (constant)
g_na : siemens (constant)
g_k : siemens (constant)
g_l : siemens (constant)
e_na : volt (constant)
e_k : volt (constant)
e_l : volt (constant)
e_exc : volt (constant)
e_inh : volt (constant)
tau_exc_rise : second (constant)
tau_exc_decay : second (constant)
tau_inh_rise : second (constant)
tau_inh_decay : second (constant)
norm_exc : 1 (constant)
norm_inh : 1 (constant)
"""


def _dual_exponential_normalizer(rise_ms: float, decay_ms: float) -> float:
    import math

    if rise_ms <= 0 or decay_ms <= rise_ms:
        raise ValueError("dual-exponential constants require 0 < rise_ms < decay_ms")
    peak_ms = rise_ms * decay_ms / (decay_ms - rise_ms) * math.log(decay_ms / rise_ms)
    return 1.0 / (math.exp(-peak_ms / decay_ms) - math.exp(-peak_ms / rise_ms))


def create_classic_hh_population(
    *, name: str, size: int, params: dict[str, Any], brian=None
):
    if brian is None:
        import brian2 as brian

    group = brian.NeuronGroup(
        size,
        EQUATIONS,
        threshold="v > 30*mV",
        refractory="v > 0*mV",
        method="exponential_euler",
        name=name,
    )
    cell_spec = get_cell_spec(params["cell_class"]) if "cell_class" in params else None
    soma = cell_spec.soma if cell_spec is not None else None
    default_c_m = soma.capacitance_pF() if soma is not None else 200.0
    default_g_na = soma.conductance_nS("na") if soma is not None else 2000.0
    default_g_k = soma.conductance_nS("k") if soma is not None else 600.0
    default_g_l = soma.conductance_nS("leak") if soma is not None else 10.0
    default_e_l = soma.e_leak_mV if soma is not None else -65.0
    group.C_m = params.get("C_m_pF", default_c_m) * brian.pfarad
    group.g_na = params.get("g_na_nS", default_g_na) * brian.nsiemens
    group.g_k = params.get("g_k_nS", default_g_k) * brian.nsiemens
    group.g_l = params.get("g_l_nS", default_g_l) * brian.nsiemens
    group.e_na = 50.0 * brian.mV
    group.e_k = -90.0 * brian.mV
    group.e_l = params.get("e_l_mV", default_e_l) * brian.mV
    group.e_exc = 0.0 * brian.mV
    group.e_inh = -80.0 * brian.mV
    exc_rise = params.get("tau_exc_rise_ms", 0.5)
    exc_decay = params.get("tau_exc_decay_ms", 5.0)
    inh_rise = params.get("tau_inh_rise_ms", 1.0)
    inh_decay = params.get("tau_inh_decay_ms", 10.0)
    group.tau_exc_rise = exc_rise * brian.ms
    group.tau_exc_decay = exc_decay * brian.ms
    group.tau_inh_rise = inh_rise * brian.ms
    group.tau_inh_decay = inh_decay * brian.ms
    group.norm_exc = _dual_exponential_normalizer(exc_rise, exc_decay)
    group.norm_inh = _dual_exponential_normalizer(inh_rise, inh_decay)
    group.v = params.get("v_init_mV", default_e_l) * brian.mV
    group.m = "alpha_m/(alpha_m+beta_m)"
    group.h = "alpha_h/(alpha_h+beta_h)"
    group.n = "alpha_n/(alpha_n+beta_n)"
    group.g_exc_rise = 0 * brian.nsiemens
    group.g_exc_decay = 0 * brian.nsiemens
    group.g_inh_rise = 0 * brian.nsiemens
    group.g_inh_decay = 0 * brian.nsiemens
    group.i_drive = 0 * brian.pA
    return group
