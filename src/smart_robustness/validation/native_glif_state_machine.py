"""Independent current-based GLIF reconstruction, not the SMART GIF backend.

Implements the pinned numerical/scheduling conventions for reference parity.
It is not yet a physiologically validated or network-promoted neuron model.
"""

from __future__ import annotations

import numpy as np


def simulate_glif(parameters, stimulus):
    p = parameters
    coeff = {
        "G": 1,
        "C": 1,
        "th_inf": 1,
        "a": 1,
        "b": 1,
        "asc_amp_array": np.ones(len(p["asc_tau_array"])),
        **p.get("coeffs", {}),
    }
    dt = p["dt"]
    cut = int(p["spike_cut_length"])
    decay_rates = 1 / np.asarray(p["asc_tau_array"], dtype=float)
    capacitance = p["C"] * coeff["C"]
    conductance = coeff["G"] / p["R_input"]
    baseline_threshold = p["th_inf"] * coeff["th_inf"]
    voltage = p["init_voltage"]
    threshold = p["init_threshold"]
    currents = np.asarray(p["init_AScurrents"], dtype=float).copy()
    size = len(stimulus)
    outputs = {
        "voltage": np.full(size, np.nan),
        "threshold": np.full(size, np.nan),
        "AScurrents": np.full((size, len(currents)), np.nan),
    }
    spikes, grid, interpolated, spike_voltages, spike_thresholds = [], [], [], [], []
    component_initialized = False
    spike_component = voltage_component = 0.0
    stopped_bad_reset = False
    index = 0
    while index < size:
        asc_name = p["AScurrent_dynamics_method"]["name"]
        if asc_name == "exp":
            next_currents = currents * np.exp(-decay_rates * dt)
        elif asc_name == "none":
            next_currents = np.zeros_like(currents)
        else:
            raise ValueError("unsupported ASC dynamics")
        if p["voltage_dynamics_method"]["name"] != "linear_forward_euler":
            raise ValueError("unsupported voltage dynamics")
        injection = stimulus[index] + np.sum(currents)
        next_voltage = voltage + (injection - conductance * (voltage - p["El"])) * dt / capacitance
        method = p["threshold_dynamics_method"]
        name, arguments = method["name"], method["params"]
        if name == "inf":
            next_threshold = baseline_threshold
        elif name in ("spike_component", "three_components_exact"):
            component_initialized = True
            spike_component *= np.exp(-arguments["b_spike"] * dt)
            if name == "three_components_exact":
                a = arguments["a_voltage"] * coeff["a"]
                b = arguments["b_voltage"] * coeff["b"]
                equilibrium = (injection + conductance * p["El"]) / conductance
                phi = a / (b - conductance / capacitance)
                e_voltage = np.exp(-conductance * dt / capacitance)
                e_threshold = np.exp(-b * dt)
                voltage_component = (
                    phi * (voltage - equilibrium) * e_voltage
                    + e_threshold
                    * (
                        voltage_component
                        - phi * (voltage - equilibrium)
                        - a / b * (equilibrium - p["El"])
                    )
                    + a / b * (equilibrium - p["El"])
                )
            else:
                voltage_component = 0.0
            next_threshold = voltage_component + spike_component + baseline_threshold
        else:
            raise ValueError("unsupported threshold dynamics")
        if not next_voltage > next_threshold:
            voltage, threshold, currents = next_voltage, next_threshold, next_currents
            outputs["voltage"][index] = voltage
            outputs["threshold"][index] = threshold
            outputs["AScurrents"][index] = currents
            index += 1
            continue
        # Preserve the native interpolation offset, including its legacy indexing.
        delta = (
            dt * (threshold - voltage) / ((next_voltage - voltage) - (next_threshold - threshold))
        )
        crossing = index * dt + delta
        offset = crossing - (index - 1) * dt
        spikes.append(index)
        grid.append(index * dt)
        interpolated.append(crossing)
        spike_voltages.append(voltage + (next_voltage - voltage) * offset / dt)
        spike_thresholds.append(threshold + (next_threshold - threshold) * offset / dt)
        asc_reset = p["AScurrent_reset_method"]
        if asc_reset["name"] == "sum":
            currents = np.asarray(p["asc_amp_array"]) * coeff[
                "asc_amp_array"
            ] + next_currents * asc_reset["params"]["r"] * np.exp(-decay_rates * dt * cut)
        elif asc_reset["name"] == "none":
            if np.sum(next_currents) != 0:
                raise ValueError("nonzero ASC currents under none reset")
            currents = np.zeros_like(currents)
        else:
            raise ValueError("unsupported ASC reset")
        reset_voltage = p["voltage_reset_method"]
        if reset_voltage["name"] == "zero":
            voltage = 0.0
        elif reset_voltage["name"] == "v_before":
            voltage = reset_voltage["params"]["a"] * next_voltage + reset_voltage["params"]["b"]
        else:
            raise ValueError("unsupported voltage reset")
        reset_threshold = p["threshold_reset_method"]
        if reset_threshold["name"] == "inf":
            threshold = baseline_threshold
        elif reset_threshold["name"] == "three_components":
            if not component_initialized:
                raise ValueError("adaptive reset before threshold component initialization")
            pars = reset_threshold["params"]
            # No additional decay for cut==0; the native component list retains
            # its just-integrated last value before adding the reset amplitude.
            spike_component *= np.exp(-pars["b_spike"] * cut * dt)
            spike_component += pars["a_spike"]
            threshold = spike_component + voltage_component + baseline_threshold
        else:
            raise ValueError("unsupported threshold reset")
        stopped_bad_reset = bool(voltage > threshold)
        if cut == 0 or index + cut < size:
            target = index + cut
            outputs["voltage"][target] = voltage
            outputs["threshold"][target] = threshold
            outputs["AScurrents"][target] = currents
        index += cut + 1
        if stopped_bad_reset:
            outputs["voltage"][index : index + 5] = voltage
            outputs["threshold"][index : index + 5] = threshold
            outputs["AScurrents"][index : index + 5] = currents
            break
    outputs.update(
        spike_time_steps=np.asarray(spikes),
        grid_spike_times=np.asarray(grid),
        interpolated_spike_times=np.asarray(interpolated),
        interpolated_spike_voltage=np.asarray(spike_voltages),
        interpolated_spike_threshold=np.asarray(spike_thresholds),
    )
    return outputs, {"bad_reset_stop": stopped_bad_reset, "native_code_called": False}
