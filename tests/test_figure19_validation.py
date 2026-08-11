from __future__ import annotations

import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.validation.isolated_cells import (
    Figure19Protocol,
    assess_figure19_kernel,
    run_figure19_kernel_condition,
)


def test_source_specific_figure19_kernel_has_frequency_ach_and_recovery_signatures() -> None:
    control = run_figure19_kernel_condition(spike_count=0, acetylcholine=False, brian=brian)
    one = run_figure19_kernel_condition(spike_count=1, acetylcholine=False, brian=brian)
    two = run_figure19_kernel_condition(spike_count=2, acetylcholine=False, brian=brian)
    ach = run_figure19_kernel_condition(spike_count=2, acetylcholine=True, brian=brian)
    assessment = assess_figure19_kernel(control, one, two, ach)
    assert assessment.frequency_dependence_pass
    assert assessment.ach_suppression_pass
    assert assessment.recovery_pass


def test_modeldb_ahp_profile_preserves_known_500ms_recovery_mismatch() -> None:
    protocol = Figure19Protocol(ahp_event_weight=4.5, ahp_convention="modeldb_112923")
    control = run_figure19_kernel_condition(
        spike_count=0, acetylcholine=False, protocol=protocol, brian=brian
    )
    one = run_figure19_kernel_condition(
        spike_count=1, acetylcholine=False, protocol=protocol, brian=brian
    )
    two = run_figure19_kernel_condition(
        spike_count=2, acetylcholine=False, protocol=protocol, brian=brian
    )
    ach = run_figure19_kernel_condition(
        spike_count=2, acetylcholine=True, protocol=protocol, brian=brian
    )
    assessment = assess_figure19_kernel(control, one, two, ach)
    assert assessment.frequency_dependence_pass
    assert assessment.ach_suppression_pass
    assert not assessment.recovery_pass
