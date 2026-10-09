"""Independent analytic checks, not historical fit or biological validation."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from scipy.optimize import brentq


@pytest.mark.parametrize("power", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("decay", [0.004, 0.02, 0.05])
@pytest.mark.parametrize("ratio", [0.01, 0.1, 0.5, 0.9])
def test_normalized_component_peak_relation(power, decay, ratio):
    # Solve an analytic identity, never optimize a model against recordings.
    desired = ratio * power * decay

    def root(y):
        return (1.0 if y == 0 else math.log1p(y) / y) - ratio

    upper = 1.0
    while root(upper) > 0:
        upper *= 2
    y = brentq(root, 0.0, upper, xtol=1e-14, rtol=1e-14)
    tau = power * decay / y
    peak = tau * math.log1p(power * decay / tau)
    assert math.isclose(peak, desired, rel_tol=1e-12, abs_tol=1e-15)
    derivative = power / (tau * math.expm1(desired / tau)) - 1 / decay
    assert abs(derivative * decay) < 1e-11

    def shape(elapsed):
        return (-math.expm1(-elapsed / tau)) ** power * math.exp(-elapsed / decay)

    assert shape(desired) > shape(desired * 0.99)
    assert shape(desired) > shape(desired * 1.01)
    for amplitude in (-0.001, 0.001):
        assert amplitude * shape(desired) / shape(desired) == amplitude


def test_audit_keeps_historical_and_execution_boundaries():
    path = Path(__file__).resolve().parents[1] / "docs/validation-results/post2008-synphys-psp-semantics-audit-1136.yaml"
    audit = yaml.safe_load(path.read_text())
    for source in audit["primary_sources"].values():
        assert len(source["sha256"]) == 64
        assert all(character in "0123456789abcdef" for character in source["sha256"])
    assert audit["primary_sources"]["date_bounded_PSP_reference"]["selection_is_historical_dependency_proof"] is False
    assert audit["static_findings"]["dependency_boundary"]["historical_fitting_revision_recovered"] is False
    assert audit["static_findings"]["dependency_boundary"]["published_invalid_domain_values_repaired"] is False
    for key in ("source_code_executed", "experimental_values_newly_read", "parameter_fitting_performed",
                "cell_or_network_execution_performed", "frozen_baseline_modified"):
        assert audit[key] is False
