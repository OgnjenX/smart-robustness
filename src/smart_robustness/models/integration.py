"""Numerical integration conventions used by the SMART reconstruction.

The standard Brian2 ``rk4`` updater advances all differential state variables
as one coupled vector.  Preserved SANNDRA 0.4.0 source instead exposes a
per-equation Runge--Kutta helper: the equation's own state variable is sampled
at the four RK stages while values supplied by other units/equations remain at
their previous-step values.  ``sanndra_scalar_rk4`` reproduces that narrowly
defined behavior as an explicit, opt-in comparator.  It is not claimed to be
bit-identical to the unavailable SMART-era SANNDRA 1.2.0 RC2 runtime.
"""

from __future__ import annotations

from brian2.parsing.sympytools import str_to_sympy, sympy_to_str
from brian2.stateupdaters.base import StateUpdateMethod, UnsupportedEquationsException
from brian2.utils.stringtools import get_identifiers
from sympy import Symbol


def sanndra_scalar_rk4(equations, variables=None, method_options=None) -> str:
    """Advance each deterministic equation with other state variables frozen.

    This implements the numerical meaning of SANNDRA 0.4.0's preserved
    ``Runge_Kutta(x, ext, coef, fun)`` helper.  For each differential variable
    ``x``, only occurrences of ``x`` are replaced by stage-intermediate values;
    all other differential variables and time-dependent inputs retain their
    values at the start of the step.
    """

    if equations.is_stochastic:
        raise UnsupportedEquationsException(
            "sanndra_scalar_rk4 does not support stochastic equations"
        )
    if method_options not in (None, {}):
        raise TypeError("sanndra_scalar_rk4 does not accept method options")

    substituted = equations.get_substituted_expressions(variables)
    statements: list[str] = []
    final_variables: list[str] = []
    reserved = set(equations.names)

    for variable, expression in substituted:
        rhs = str_to_sympy(expression.code)
        state = Symbol(variable, real=True)
        # Brian's parser may create a symbol with assumptions that differ from
        # the one above.  Select the expression's actual symbol by name.
        state = next(
            (symbol for symbol in rhs.free_symbols if symbol.name == variable),
            state,
        )
        k1_name = _temporary_name("sanndra_k1", variable, reserved)
        k2_name = _temporary_name("sanndra_k2", variable, reserved)
        k3_name = _temporary_name("sanndra_k3", variable, reserved)
        k4_name = _temporary_name("sanndra_k4", variable, reserved)
        new_name = _temporary_name("sanndra_new", variable, reserved)
        k1 = Symbol(k1_name, real=True)
        k2 = Symbol(k2_name, real=True)
        k3 = Symbol(k3_name, real=True)

        statements.extend(
            (
                f"{k1_name} = dt*({sympy_to_str(rhs)})",
                f"{k2_name} = dt*({sympy_to_str(rhs.subs(state, state + k1 / 2))})",
                f"{k3_name} = dt*({sympy_to_str(rhs.subs(state, state + k2 / 2))})",
                f"{k4_name} = dt*({sympy_to_str(rhs.subs(state, state + k3))})",
                (
                    f"{new_name} = {variable} + "
                    f"({k1_name} + 2*{k2_name} + 2*{k3_name} + {k4_name})/6"
                ),
            )
        )
        final_variables.append(variable)

    for variable in final_variables:
        statements.append(f"{variable} = _sanndra_new_{variable}")
    return "\n".join(statements)


class SanndraScalarRK4(StateUpdateMethod):
    """Brian2 state-updater wrapper for the preserved scalar RK4 semantics."""

    def __call__(self, equations, variables=None, method_options=None) -> str:
        return sanndra_scalar_rk4(equations, variables, method_options)


def register_sanndra_scalar_rk4() -> None:
    """Register the opt-in updater idempotently at model-package import."""

    name = "sanndra_scalar_rk4"
    if name not in StateUpdateMethod.stateupdaters:
        StateUpdateMethod.register(name, SanndraScalarRK4())


def _temporary_name(prefix: str, variable: str, reserved: set[str]) -> str:
    name = f"_{prefix}_{variable}"
    if name in reserved or get_identifiers(name) != {name}:
        raise ValueError(f"cannot construct integration temporary {name!r}")
    return name
