from dataclasses import replace

import numpy as np
import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.classic_sector import (
    FirstOrderRuntimeConventions,
    build_first_order_chemical_sector,
)
from smart_robustness.protocols import (
    BarOrientation,
    ClassicBarStimulus,
    apply_bar_stimulus,
    clear_bar_stimulus,
)


@pytest.mark.parametrize("declared", [False, True])
def test_input_precedence_replaces_projection_and_follows_image_lifecycle(declared):
    brian.start_scope()
    base = FirstOrderRuntimeConventions()
    conventions = replace(base, mixed_input_gate_convention="declared_external_input") if declared else base
    assert (conventions.fingerprint != base.fingerprint) == declared
    sector = build_first_order_chemical_sector(conventions=conventions, brian=brian)
    assert ("modeldb112923.projection.042" in sector.projections) == (not declared)
    population = sector.populations["thalamic_interneuron"]
    ports = population.compiled.external_input_ports
    assert bool(ports) == declared
    if not declared:
        return
    assert ports[0].sensitivities_mV == (0, 0.37, 0, 0)
    assert ports[0].conductance_density_mS_cm2 == 0.21
    group = population.group
    assert np.all(group.external_mixed_input_input_green[:] == 0)
    stimulus = ClassicBarStimulus(BarOrientation.VERTICAL)
    apply_bar_stimulus(sector, stimulus)
    expected = np.zeros(81)
    expected[list(stimulus.active_indices)] = 120
    np.testing.assert_array_equal(group.external_mixed_input_input_green[:], expected)
    # Input Equation 6: local leak plus sensitivity times image, not 0-mV
    # ligand reversal or a layer-4 event gate.
    group.v_proximal_dendrite = group.e_l_proximal_dendrite
    current = np.asarray(group.i_external_mixed_input[:] / brian.pA)
    conductance = float(group.g_external_mixed_input[0] / brian.nsiemens)
    np.testing.assert_allclose(current, conductance * 0.37 * expected)
    clear_bar_stimulus(sector, stimulus)
    np.testing.assert_array_equal(group.external_mixed_input_input_green[:], 0)
    brian.prefs.codegen.target = "numpy"
    sector.network.run(0.01 * brian.ms)
    assert np.all(np.isfinite(group.v_proximal_dendrite[:] / brian.mV))
