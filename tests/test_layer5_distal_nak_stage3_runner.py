from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

NAMESPACE = run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "scripts/run_layer5_distal_nak_stage3.py"
    )
)
figure7_gates = NAMESPACE["_figure7_gates"]
figure10_gates = NAMESPACE["_figure10_gates"]


def test_stage3_figure7_gate_conjunction_matches_frozen_targets() -> None:
    profile = {
        "figure7_gates": {
            "match_relay_active_indices": [38, 39, 40, 41, 42],
            "match_relay_events": 20,
            "mismatch_relay_allowed_indices": [40],
            "match_nonspecific_events": 4,
            "mismatch_nonspecific_events": 7,
        }
    }
    match = {
        "relay_active_indices": [38, 39, 40, 41, 42],
        "relay_events": 20,
        "trn_events": 700,
        "nonspecific_events": 4,
        "trn_to_nonspecific_gaba_integral_ms": 10.0,
    }
    mismatch = {
        "relay_active_indices": [40],
        "relay_events": 3,
        "trn_events": 500,
        "nonspecific_events": 7,
        "trn_to_nonspecific_gaba_integral_ms": 5.0,
    }

    gates = figure7_gates(match, mismatch, profile)
    assert all(gates.values())
    assert figure7_gates(
        match,
        mismatch | {"nonspecific_events": 6},
        profile,
    )["mismatch_nonspecific_events"] is False


def test_stage3_figure10_gate_conjunction_requires_causal_replacement() -> None:
    intact = SimpleNamespace(
        pre_layer4_events=10,
        pre_layer4_active_indices=(38, 39, 40, 41, 42),
        projection035_post_release_integral_pA_ms=((31, 2.0), (49, 2.0)),
        nonspecific_post_events=7,
        layer5_post_events=5,
        layer6i_post_events=4,
        winner_post_events=53,
    )
    control = SimpleNamespace(
        pre_layer4_events=10,
        pre_layer4_active_indices=(38, 39, 40, 41, 42),
        projection035_post_release_integral_pA_ms=((31, 2.0), (49, 2.0)),
        winner_post_events=63,
    )
    intact_summary = {
        "pre_release_alternative_events": 0,
        "replacement_winner_candidates": [31],
        "late_alternative_margin": 3,
    }
    control_summary = {
        "pre_release_alternative_events": 0,
        "replacement_winner_candidates": [],
        "late_alternative_margin": -2,
    }

    gates = figure10_gates(intact, control, intact_summary, control_summary)
    assert all(gates.values())
    failed = figure10_gates(
        intact,
        control,
        intact_summary | {"replacement_winner_candidates": []},
        control_summary,
    )
    assert failed["intact_replacement_winner"] is False
