from __future__ import annotations

import numpy as np

from smart_robustness.models.integration import sanndra_scalar_rk4


def test_sanndra_scalar_rk4_freezes_other_differential_variables() -> None:
    import brian2 as brian

    brian.start_scope()
    brian.defaultclock.dt = 0.1 * brian.ms
    equations = """
    dx/dt = y/ms : 1
    dy/dt = x/ms : 1
    """
    group = brian.NeuronGroup(1, equations, method=sanndra_scalar_rk4)
    group.x = 1
    group.y = 2
    network = brian.Network(group)
    network.run(0.1 * brian.ms)

    # With the other equation frozen, x'=y and y'=x each reduce to one Euler
    # increment even though the equation's own RK4 calculation is retained.
    assert float(group.x[0]) == 1.2
    assert float(group.y[0]) == 2.1


def test_sanndra_scalar_rk4_matches_rk4_for_one_linear_state() -> None:
    import brian2 as brian

    def run(method) -> float:
        brian.start_scope()
        brian.defaultclock.dt = 0.1 * brian.ms
        group = brian.NeuronGroup(1, "dx/dt = -x/(2*ms) : 1", method=method)
        group.x = 1
        brian.Network(group).run(1 * brian.ms)
        return float(group.x[0])

    assert np.isclose(run(sanndra_scalar_rk4), run("rk4"), rtol=0, atol=1e-14)


def test_sanndra_scalar_rk4_is_registered_by_model_package() -> None:
    import brian2 as brian

    brian.start_scope()
    group = brian.NeuronGroup(
        1,
        "dx/dt = -x/ms : 1",
        method="sanndra_scalar_rk4",
    )
    group.x = 1
    brian.Network(group).run(0.1 * brian.ms)
    assert 0 < float(group.x[0]) < 1


def test_registered_wrapper_generates_identical_code_to_executed_callable() -> None:
    import brian2 as brian

    from smart_robustness.models.integration import SanndraScalarRK4

    equations = brian.Equations(
        """
        dx/dt = y/ms : 1
        dy/dt = x/ms : 1
        """
    )
    assert SanndraScalarRK4()(equations) == sanndra_scalar_rk4(equations)
