"""Predeclared isolated-cell protocols for SMART figure validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..models.compartmental_hh import create_compartmental_hh_population
from ..models.modeldb112923 import (
    ahp_ach_layer5_spec,
    ahp_density_to_total_nS,
    figure8_relay_spec,
)


@dataclass(frozen=True, slots=True)
class Figure8Protocol:
    """Source-backed parts of the Figure 8 relay-cell protocol.

    The paper specifies the 0.3 nA pulse and the presence/absence of a
    hyperpolarizing voltage clamp, but does not report the clamp voltage or
    exact epoch durations. Those values remain explicit calibration inputs.
    """

    pulse_pA: float = 300.0
    precondition_ms: float = 100.0
    pulse_ms: float = 300.0
    depolarized_hold_mV: float = -60.0
    hyperpolarized_hold_mV: float = -80.0
    hyperpolarizing_bias_pA: float = 0.0
    clamp_interpretation: str = "released_exact_preclamp"
    hyperpolarizing_clamp_conductance_nS: float = 0.0
    dt_ms: float = 0.01

    def __post_init__(self) -> None:
        if self.clamp_interpretation not in {
            "released_exact_preclamp",
            "caption_finite_conductance",
        }:
            raise ValueError("unknown Figure 8 clamp interpretation")
        values = (
            self.pulse_pA,
            self.precondition_ms,
            self.pulse_ms,
            self.depolarized_hold_mV,
            self.hyperpolarized_hold_mV,
            self.hyperpolarizing_bias_pA,
            self.hyperpolarizing_clamp_conductance_nS,
            self.dt_ms,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Figure 8 protocol values must be finite")
        if self.precondition_ms < 0 or self.pulse_ms <= 0 or self.dt_ms <= 0:
            raise ValueError("Figure 8 durations and dt must be positive")
        if self.hyperpolarizing_clamp_conductance_nS < 0:
            raise ValueError("Figure 8 clamp conductance cannot be negative")
        if (
            self.clamp_interpretation == "caption_finite_conductance"
            and self.hyperpolarizing_clamp_conductance_nS <= 0
        ):
            raise ValueError("caption finite-conductance protocol requires a positive clamp")


@dataclass(frozen=True, slots=True)
class IsolatedCellTrace:
    condition: str
    time_ms: np.ndarray
    soma_voltage_mV: np.ndarray
    spike_times_ms: np.ndarray


@dataclass(frozen=True, slots=True)
class Figure8Assessment:
    tonic_pass: bool
    burst_pass: bool
    tonic_spike_count: int
    burst_spike_count: int
    notes: tuple[str, ...]

    @property
    def reproduced(self) -> bool:
        return self.tonic_pass and self.burst_pass


@dataclass(frozen=True, slots=True)
class Figure8SourceCandidate:
    leak_density_mS_cm2: float
    specific_capacitance_uF_cm2: float
    tonic: IsolatedCellTrace
    burst: IsolatedCellTrace
    assessment: Figure8Assessment


@dataclass(frozen=True, slots=True)
class Figure19Protocol:
    """Source-backed AHP/ACh kernel assay derived from Figure 19 and ModelDB.

    Figure 19 generated spikes with 10 ms current injections, but the exact
    injected current and KInNeSS soma axial default are not reported. This
    assay delivers the resulting spike events directly, isolating the
    published AHP/ACh mechanism from those unresolved spike-generation inputs.
    """

    duration_ms: float = 500.0
    dt_ms: float = 0.1
    soma_axial_resistance_kohm_cm: float = 35.0
    ahp_density_mS_cm2: float = 0.1
    ahp_event_weight: float = 1.0
    ahp_convention: str = "paper_text"


@dataclass(frozen=True, slots=True)
class Figure19Assessment:
    frequency_dependence_pass: bool
    ach_suppression_pass: bool
    recovery_pass: bool
    one_spike_minimum_mV: float
    two_spike_minimum_mV: float
    ach_minimum_mV: float
    two_spike_recovery_error_mV: float

    @property
    def reproduced(self) -> bool:
        return self.frequency_dependence_pass and self.ach_suppression_pass and self.recovery_pass


@dataclass(frozen=True, slots=True)
class TrnRecruitmentProtocol:
    """Isolated promotion gate for a candidate classic-SMART TRN cell.

    Gate amplitudes default to the largest layer-6II AMPA and NMDA values
    measured at the diagnostic cells in the current Figure 7 network assay.
    They are explicit assay inputs, not fitted replacement synaptic weights.
    """

    pre_drive_ms: float = 5.0
    drive_ms: float = 45.0
    dt_ms: float = 0.01
    layer6ii_ampa_gate: float = 1.66427
    layer6ii_nmda_gate: float = 0.10913
    drive_multiplier: float = 1.0
    axial_conductance_scale: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.pre_drive_ms,
            self.drive_ms,
            self.dt_ms,
            self.layer6ii_ampa_gate,
            self.layer6ii_nmda_gate,
            self.drive_multiplier,
            self.axial_conductance_scale,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("TRN recruitment protocol values must be finite")
        if self.pre_drive_ms < 0 or self.drive_ms <= 0 or self.dt_ms <= 0:
            raise ValueError("TRN recruitment durations and dt must be positive")
        if self.layer6ii_ampa_gate < 0 or self.layer6ii_nmda_gate < 0:
            raise ValueError("TRN recruitment gates cannot be negative")
        if self.drive_multiplier < 0 or self.axial_conductance_scale <= 0:
            raise ValueError("TRN recruitment scales must be positive")


@dataclass(frozen=True, slots=True)
class TrnRecruitmentResult:
    driven: bool
    spike_times_ms: tuple[float, ...]
    post_drive_spike_times_ms: tuple[float, ...]
    soma_voltage_range_mV: tuple[float, float]
    proximal_voltage_range_mV: tuple[float, float]
    finite: bool
    convention_fingerprint: str
    applied_layer6ii_ampa_gate: float
    applied_layer6ii_nmda_gate: float

    @property
    def post_drive_spike_count(self) -> int:
        return len(self.post_drive_spike_times_ms)


def run_trn_recruitment_condition(
    *,
    driven: bool,
    conventions=None,
    protocol: TrnRecruitmentProtocol | None = None,
    brian=None,
) -> TrnRecruitmentResult:
    """Run one independent control or constant receptor-drive TRN trial."""

    if brian is None:
        import brian2 as brian
    from ..classic_sector import figure6_runtime_conventions, first_order_population_parameters
    from ..models.modeldb112923 import first_order_population_facts

    protocol = protocol or TrnRecruitmentProtocol()
    conventions = conventions or figure6_runtime_conventions()
    fact = next(fact for fact in first_order_population_facts() if fact.canonical_name == "trn")
    params = first_order_population_parameters(fact, conventions=conventions)
    brian.start_scope()
    brian.defaultclock.dt = protocol.dt_ms * brian.ms
    population = create_compartmental_hh_population(
        name="isolated_trn_recruitment", size=1, params=params, brian=brian
    )
    for parameter in population.compiled.axial_parameter_names:
        setattr(
            population.group,
            parameter,
            getattr(population.group, parameter) * protocol.axial_conductance_scale,
        )
    spikes = brian.SpikeMonitor(population.group)
    voltage = brian.StateMonitor(
        population.group, ("v_soma", "v_proximal_dendrite"), record=True
    )
    network = brian.Network(population.group, spikes, voltage)
    if protocol.pre_drive_ms:
        network.run(protocol.pre_drive_ms * brian.ms)
    if driven:
        population.group.port_004_gate = (
            protocol.layer6ii_ampa_gate * protocol.drive_multiplier
        )
        population.group.port_001_gate = (
            protocol.layer6ii_nmda_gate * protocol.drive_multiplier
        )
    network.run(protocol.drive_ms * brian.ms)

    spike_times = np.asarray(spikes.t / brian.ms, dtype=float)
    time_ms = np.asarray(voltage.t / brian.ms, dtype=float)
    drive_window = time_ms >= protocol.pre_drive_ms
    soma = np.asarray(voltage.v_soma[0] / brian.mV, dtype=float)[drive_window]
    proximal = np.asarray(voltage.v_proximal_dendrite[0] / brian.mV, dtype=float)[
        drive_window
    ]
    finite = bool(np.all(np.isfinite(soma)) and np.all(np.isfinite(proximal)))

    def voltage_range(values: np.ndarray) -> tuple[float, float]:
        if not finite:
            return (float("nan"), float("nan"))
        return (float(np.min(values)), float(np.max(values)))

    return TrnRecruitmentResult(
        driven=driven,
        spike_times_ms=tuple(float(value) for value in spike_times),
        post_drive_spike_times_ms=tuple(
            float(value) for value in spike_times if value >= protocol.pre_drive_ms
        ),
        soma_voltage_range_mV=voltage_range(soma),
        proximal_voltage_range_mV=voltage_range(proximal),
        finite=finite,
        convention_fingerprint=conventions.fingerprint,
        applied_layer6ii_ampa_gate=(
            protocol.layer6ii_ampa_gate * protocol.drive_multiplier if driven else 0.0
        ),
        applied_layer6ii_nmda_gate=(
            protocol.layer6ii_nmda_gate * protocol.drive_multiplier if driven else 0.0
        ),
    )


def run_figure8_condition(
    *,
    hyperpolarized: bool,
    model_params: dict[str, Any],
    protocol: Figure8Protocol | None = None,
    brian=None,
) -> IsolatedCellTrace:
    """Run one Figure 8 condition under a declared caption interpretation.

    The original reconstruction exactly preclamps either condition and releases
    the clamp at pulse onset. The caption-literal alternative freely equilibrates
    the top condition and keeps a finite KInNeSS-style hyperpolarizing voltage
    conductance active in the bottom condition. The missing legacy experiment
    file prevents either interpretation from being silently designated official.
    """

    if brian is None:
        import brian2 as brian

    protocol = protocol or Figure8Protocol()
    brian.start_scope()
    brian.defaultclock.dt = protocol.dt_ms * brian.ms
    hold_mV = protocol.hyperpolarized_hold_mV if hyperpolarized else protocol.depolarized_hold_mV
    params = dict(model_params)
    params["cell_class"] = "thalamic_relay"
    if protocol.clamp_interpretation == "released_exact_preclamp":
        params["v_init_mV"] = hold_mV
    population = create_compartmental_hh_population(
        name="figure8_relay", size=1, params=params, brian=brian
    )
    group = population.group

    def apply_voltage_clamp() -> None:
        for compartment in population.compartments:
            setattr(group, f"v_{compartment}", hold_mV * brian.mV)

    exact_preclamp = protocol.clamp_interpretation == "released_exact_preclamp"

    def apply_finite_voltage_input() -> None:
        if not hyperpolarized:
            return
        group.i_syn_soma = protocol.hyperpolarizing_clamp_conductance_nS * brian.nsiemens * (
            protocol.hyperpolarized_hold_mV * brian.mV - group.v_soma
        )

    clamp_operation = brian.NetworkOperation(
        apply_voltage_clamp if exact_preclamp else apply_finite_voltage_input,
        when="start",
    )
    voltage = brian.StateMonitor(group, "v_soma", record=True)
    spikes = brian.SpikeMonitor(group)
    network = brian.Network(group, clamp_operation, voltage, spikes)
    network.run(protocol.precondition_ms * brian.ms)
    pulse_start_ms = float(network.t / brian.ms)
    if exact_preclamp:
        clamp_operation.active = False
    pulse_pA = protocol.pulse_pA
    if hyperpolarized:
        pulse_pA += protocol.hyperpolarizing_bias_pA
    group.i_drive_soma = pulse_pA * brian.pA
    network.run(protocol.pulse_ms * brian.ms)

    times_ms = np.asarray(voltage.t / brian.ms)
    pulse_mask = times_ms >= pulse_start_ms
    spike_times_ms = np.asarray(spikes.t / brian.ms) - pulse_start_ms
    spike_times_ms = spike_times_ms[spike_times_ms >= 0]
    return IsolatedCellTrace(
        condition="hyperpolarized" if hyperpolarized else "depolarized",
        time_ms=times_ms[pulse_mask] - pulse_start_ms,
        soma_voltage_mV=np.asarray(voltage.v_soma[0] / brian.mV)[pulse_mask],
        spike_times_ms=spike_times_ms,
    )


def figure8_source_parameters(
    *,
    leak_density_mS_cm2: float,
    specific_capacitance_uF_cm2: float,
) -> dict[str, Any]:
    """Build the dedicated Ca_rebound.xml cell with two explicit missing defaults."""

    if leak_density_mS_cm2 <= 0 or specific_capacitance_uF_cm2 <= 0:
        raise ValueError("Figure 8 leak and capacitance candidates must be positive")
    return {
        "cell_spec": figure8_relay_spec(leak_density_mS_cm2=leak_density_mS_cm2),
        "cell_class": "thalamic_relay",
        "axial_convention": "kinness_serialized_edge",
        "leak_convention": "table3_reversal",
        "voltage_coordinate": "relative_to_table3_leak",
        "nak_rate_convention": "standard_traub_miles",
        "calcium_gate_convention": "modeldb_112923",
        "calcium_voltage_coordinate": "integrated_voltage",
        "gate_initialization_convention": "steady_state_at_initial_voltage",
        "membrane_initialization_convention": "physical_leak_voltage",
        "spike_event_coordinate": "absolute_physical",
        "spike_event_threshold_mV": 30.0,
        "spike_event_rule": "latched_peak_then_zero",
        "calcium_density_convention": "table3",
        "ahp_convention": "modeldb_112923",
        "specific_capacitance_uF_cm2": float(specific_capacitance_uF_cm2),
        "enable_ahp_ach": False,
        "method": "rk4",
        "e_na_mV": 50.0,
        "e_k_mV": -90.0,
        "e_ca_mV": 180.0,
    }


def run_figure8_source_candidate(
    *,
    leak_density_mS_cm2: float,
    specific_capacitance_uF_cm2: float,
    protocol: Figure8Protocol | None = None,
    brian=None,
) -> Figure8SourceCandidate:
    """Run both Figure 8 conditions for one labeled missing-default candidate."""

    protocol = protocol or Figure8Protocol(depolarized_hold_mV=-62.3)
    params = figure8_source_parameters(
        leak_density_mS_cm2=leak_density_mS_cm2,
        specific_capacitance_uF_cm2=specific_capacitance_uF_cm2,
    )
    tonic = run_figure8_condition(
        hyperpolarized=False,
        model_params=params,
        protocol=protocol,
        brian=brian,
    )
    burst = run_figure8_condition(
        hyperpolarized=True,
        model_params=params,
        protocol=protocol,
        brian=brian,
    )
    return Figure8SourceCandidate(
        leak_density_mS_cm2=float(leak_density_mS_cm2),
        specific_capacitance_uF_cm2=float(specific_capacitance_uF_cm2),
        tonic=tonic,
        burst=burst,
        assessment=assess_figure8(tonic, burst, pulse_ms=protocol.pulse_ms),
    )


def assess_figure8(
    tonic: IsolatedCellTrace,
    burst: IsolatedCellTrace,
    *,
    pulse_ms: float = 300.0,
) -> Figure8Assessment:
    """Score the qualitative Figure 8 signatures without fitting the traces."""

    tonic_spikes = tonic.spike_times_ms
    burst_spikes = burst.spike_times_ms
    tonic_sustained = len(tonic_spikes) >= 3 and tonic_spikes[-1] >= 0.6 * pulse_ms
    if len(tonic_spikes) >= 4:
        intervals = np.diff(tonic_spikes)
        tonic_regular = float(np.std(intervals) / np.mean(intervals)) < 0.5
    else:
        tonic_regular = False
    early_burst = int(np.sum(burst_spikes <= 80.0))
    late_burst = int(np.sum(burst_spikes > 120.0))
    burst_transient = 2 <= early_burst <= 10 and late_burst == 0
    notes: list[str] = []
    if not (tonic_sustained and tonic_regular):
        notes.append("depolarized condition is not a sustained regular tonic train")
    if not burst_transient:
        notes.append("hyperpolarized condition is not an early burst followed by silence")
    return Figure8Assessment(
        tonic_pass=tonic_sustained and tonic_regular,
        burst_pass=burst_transient,
        tonic_spike_count=len(tonic_spikes),
        burst_spike_count=len(burst_spikes),
        notes=tuple(notes),
    )


def run_figure19_kernel_condition(
    *,
    spike_count: int,
    acetylcholine: bool,
    protocol: Figure19Protocol | None = None,
    brian=None,
) -> IsolatedCellTrace:
    """Run the source-specific layer-5 AHP/ACh kernel after explicit events."""

    if brian is None:
        import brian2 as brian

    if spike_count < 0:
        raise ValueError("spike_count cannot be negative")
    protocol = protocol or Figure19Protocol()
    brian.start_scope()
    brian.defaultclock.dt = protocol.dt_ms * brian.ms
    cell = ahp_ach_layer5_spec(soma_axial_resistance_kohm_cm=protocol.soma_axial_resistance_kohm_cm)
    params = {
        "cell_spec": cell,
        "cell_class": "layer5_excitatory",
        "axial_convention": "kinness_2008",
        "leak_convention": "table3_reversal",
        "voltage_coordinate": "shifted_67_mV",
        "nak_rate_convention": "standard_traub_miles",
        "calcium_gate_convention": "modeldb_112923",
        "calcium_voltage_coordinate": "integrated_voltage",
        "gate_initialization_convention": "steady_state_at_initial_voltage",
        "membrane_initialization_convention": "physical_leak_voltage",
        "spike_event_coordinate": "relative_to_soma_leak",
        "spike_event_threshold_mV": 30.0,
        "spike_event_rule": "latched_peak_then_zero",
        "calcium_density_convention": "table3",
        "ahp_convention": protocol.ahp_convention,
        "specific_capacitance_uF_cm2": 1.0,
        "enable_ahp_ach": True,
        "ahp_max_conductance_nS": ahp_density_to_total_nS(protocol.ahp_density_mS_cm2, cell),
        "ahp_event_weight": protocol.ahp_event_weight,
        "e_ahp_mV": -90.0,
        "v_init_mV": -78.0,
        "method": "rk4",
    }
    population = create_compartmental_hh_population(
        name="figure19_layer5", size=1, params=params, brian=brian
    )
    group = population.group
    group.ahp_rise = spike_count
    group.ahp_fall = spike_count
    if acetylcholine:
        population.trigger_ach()
    voltage = brian.StateMonitor(group, "v_soma", record=True)
    network = brian.Network(group, voltage)
    network.run(protocol.duration_ms * brian.ms)
    return IsolatedCellTrace(
        condition=f"{spike_count}_spike" + ("_ach" if acetylcholine else ""),
        time_ms=np.asarray(voltage.t / brian.ms),
        soma_voltage_mV=np.asarray(voltage.v_soma[0] / brian.mV),
        spike_times_ms=np.asarray([], dtype=float),
    )


def assess_figure19_kernel(
    control: IsolatedCellTrace,
    one_spike: IsolatedCellTrace,
    two_spike: IsolatedCellTrace,
    two_spike_ach: IsolatedCellTrace,
    *,
    recovery_tolerance_mV: float = 2.0,
) -> Figure19Assessment:
    """Score AHP effects relative to a matched no-event control trace."""

    one_min = float(np.min(one_spike.soma_voltage_mV))
    two_min = float(np.min(two_spike.soma_voltage_mV))
    ach_min = float(np.min(two_spike_ach.soma_voltage_mV))
    one_effect = control.soma_voltage_mV - one_spike.soma_voltage_mV
    two_effect = control.soma_voltage_mV - two_spike.soma_voltage_mV
    ach_effect = control.soma_voltage_mV - two_spike_ach.soma_voltage_mV
    recovery_error = abs(float(two_effect[-1]))
    return Figure19Assessment(
        frequency_dependence_pass=float(np.max(two_effect)) > float(np.max(one_effect)),
        ach_suppression_pass=float(np.max(ach_effect)) < float(np.max(two_effect)),
        recovery_pass=recovery_error <= recovery_tolerance_mV,
        one_spike_minimum_mV=one_min,
        two_spike_minimum_mV=two_min,
        ach_minimum_mV=ach_min,
        two_spike_recovery_error_mV=recovery_error,
    )
