"""Opt-in effective single-compartment Na/K/M hypothesis; no SMART imports."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Parameters:
    capacitance_pF: float
    leak_nS: float
    leak_reversal_mV: float
    sodium_nS: float
    potassium_nS: float
    m_current_nS: float
    threshold_shift_mV: float
    m_tau_max_ms: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError('nonfinite model parameters')
        if min(self.capacitance_pF, self.leak_nS, self.m_tau_max_ms) <= 0:
            raise ValueError('nonpositive capacitance, leak or time constant')
        if min(self.sodium_nS, self.potassium_nS, self.m_current_nS) < 0:
            raise ValueError('negative conductance')


EQUATIONS = '''
dv/dt = (gL*(EL-v)+gNa*m**3*h*(ENa-v)+gK*n**4*(EK-v)+gM*p*(EK-v)+sampled)/C : volt
sampled : amp
dm/dt = am*(1-m)-bm*m : 1
dh/dt = ah*(1-h)-bh*h : 1
dn/dt = an*(1-n)-bn*n : 1
dp/dt = (pinf-p)/taup : 1
am = 1.28/exprel((13*mV-(v-VT))/(4*mV))/ms : Hz
bm = 1.4/exprel(((v-VT)-40*mV)/(5*mV))/ms : Hz
ah = .128*exp((17*mV-(v-VT))/(18*mV))/ms : Hz
bh = 4/(1+exp((40*mV-(v-VT))/(5*mV)))/ms : Hz
an = .16/exprel((15*mV-(v-VT))/(5*mV))/ms : Hz
bn = .5*exp((10*mV-(v-VT))/(40*mV))/ms : Hz
pinf = 1/(1+exp(-(v+35*mV)/(10*mV))) : 1
taup = taumax/(3.3*exp((v+35*mV)/(20*mV))+exp(-(v+35*mV)/(20*mV))) : second
C : farad (constant)
gL : siemens (constant)
EL : volt (constant)
gNa : siemens (constant)
gK : siemens (constant)
gM : siemens (constant)
VT : volt (constant)
taumax : second (constant)
'''


def population(parameters, clock, drive, initial_mV, brian):
    """Build explicit-network cells; effective total conductances, not densities."""
    if not parameters or not all(isinstance(p, Parameters) for p in parameters):
        raise ValueError('explicit parameter list required')
    initial = np.asarray(initial_mV, dtype=float)
    if initial.shape != (len(parameters),) or not np.isfinite(initial).all():
        raise ValueError('one finite initial voltage per cell required')
    group = brian.NeuronGroup(len(parameters), EQUATIONS, clock=clock, method='rk4',
                              namespace={'drive': drive, 'ENa': 50*brian.mV,
                                         'EK': -90*brian.mV})
    fields = {'C': ('capacitance_pF', brian.pfarad), 'gL': ('leak_nS', brian.nsiemens),
              'EL': ('leak_reversal_mV', brian.mV), 'gNa': ('sodium_nS', brian.nsiemens),
              'gK': ('potassium_nS', brian.nsiemens), 'gM': ('m_current_nS', brian.nsiemens),
              'VT': ('threshold_shift_mV', brian.mV), 'taumax': ('m_tau_max_ms', brian.ms)}
    for name, (field, unit) in fields.items():
        setattr(group, name, np.array([getattr(p, field) for p in parameters])*unit)
    group.v = initial*brian.mV
    # Equilibrium gates are an explicit initialization choice, not archive zero gates.
    group.m = 'am/(am+bm)'
    group.h = 'ah/(ah+bh)'
    group.n = 'an/(an+bn)'
    group.p = 'pinf'
    group.run_regularly('sampled=drive(t,i)', dt=clock.dt, when='start', order=-1)
    return group


def simulate(parameters, nominal_current_pA, rate_hz, initial_mV):
    """Replay exact sampled nominal current without interpolating source voltages."""
    import brian2 as brian

    current = np.asarray(nominal_current_pA, dtype=float)
    if (current.ndim != 2 or current.shape[1] != len(parameters) or len(current) < 2
        or len(current) > 10_000_000 or not np.isfinite(current).all()
        or not math.isfinite(rate_hz) or not 20000 < rate_hz <= 1_000_000):
        raise ValueError('invalid sampled current or rate')
    dt = (1/rate_hz)*brian.second
    clock = brian.Clock(dt=dt)
    drive = brian.TimedArray(current*brian.pA, dt=dt)
    group = population(parameters, clock, drive, initial_mV, brian)
    monitor = brian.StateMonitor(group, ('v', 'm', 'h', 'n', 'p'), record=True, when='start')
    network = brian.Network(group, monitor)
    previous = brian.prefs.codegen.target
    try:
        brian.prefs.codegen.target = 'numpy'
        network.run(len(current)*dt, namespace={})
    finally:
        brian.prefs.codegen.target = previous
    voltage = np.array(monitor.v/brian.mV)
    if voltage.shape != current.T.shape or not np.isfinite(voltage).all():
        raise ValueError('numerical voltage failure')
    for name in ('m', 'h', 'n', 'p'):
        gate = np.asarray(getattr(monitor, name))
        if not np.isfinite(gate).all() or np.any(gate < -1e-8) or np.any(gate > 1+1e-8):
            raise ValueError('numerical gate failure')
    return voltage
