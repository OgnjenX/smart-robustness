from __future__ import annotations

from typing import Any


def connect_conductance(
    pre: Any,
    post: Any,
    *,
    receptor: str,
    probability: float,
    weight_ns: float,
    delay_ms: float,
    depletion: float,
    recovery_ms: float,
    brian: Any,
    name: str,
) -> Any:
    """Conductance synapse with Grossberg-style recoverable transmitter resource.

    The post-synaptic receptor port is a normalized dual exponential. Resource
    dynamics implement dz/dt=(1-z)/tau and z <- z*(1-epsilon) on an event.
    """
    if receptor not in {"exc", "inh"}:
        raise ValueError("receptor must be 'exc' or 'inh'")
    targets = (
        ("g_exc_rise_post", "g_exc_decay_post")
        if receptor == "exc"
        else ("g_inh_rise_post", "g_inh_decay_post")
    )
    syn = brian.Synapses(
        pre,
        post,
        model="dz/dt = (1-z)/tau_rec : 1 (clock-driven)\n"
        "w : siemens (constant)\n"
        "epsilon : 1 (constant)\n"
        "tau_rec : second (constant)",
        on_pre=f"{targets[0]} += w*z\n{targets[1]} += w*z\nz *= (1-epsilon)",
        method="exact",
        name=name,
    )
    syn.connect(p=probability)
    syn.w = weight_ns * brian.nsiemens
    syn.delay = delay_ms * brian.ms
    syn.z = 1.0
    syn.epsilon = depletion
    syn.tau_rec = recovery_ms * brian.ms
    return syn
