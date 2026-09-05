from dataclasses import replace
from pathlib import Path

import yaml

from smart_robustness.validation.calibration import runtime_conventions_for_candidate


def test_declared_input_diagnostic_changes_only_registered_runtime_factor():
    root = Path(__file__).resolve().parents[1]
    profile = yaml.safe_load((root / "configs/calibration/figure6_declared_interneuron_input_v1.yaml").read_text())
    base = yaml.safe_load((root / profile["base_profile"]).read_text())
    control = yaml.safe_load((root / profile["baseline_profile"]).read_text())
    conventions = replace(runtime_conventions_for_candidate(base["candidate"]), **profile["runtime_overrides"])
    historical = replace(conventions, mixed_input_gate_convention="historical_nested_projection")
    assert historical.fingerprint == "0aa301be0a82a34b0a3337030eedb66dbc33e0ec0532ead902638088a434b4d9"
    assert conventions.fingerprint == "e016bc9f3d0557e2b23ad00e7d956509df9cf44de42db6f66c6a484a42d1f7a0"
    assert profile["trn_to_relay_gaba"] == control["trn_to_relay_gaba"]
    required = {"thalamic_relay", "thalamic_interneuron", "layer4_excitatory_v1",
                "layer23_excitatory_v1", "layer5_excitatory_v1",
                "layer6i_excitatory_v1", "layer6ii_excitatory_v1"}
    assert required <= set(profile["monitored_populations"])
