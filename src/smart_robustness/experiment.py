from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np

from .analysis import summarize_rate
from .circuit import build_minimal_benchmark
from .config import ExperimentConfig


def run_experiment(config: ExperimentConfig) -> tuple[Path, Path]:
    import brian2 as brian

    brian.start_scope()
    brian.prefs.codegen.target = "numpy"
    brian.seed(config.seed)
    np.random.seed(config.seed)
    brian.defaultclock.dt = config.dt_ms * brian.ms

    built = build_minimal_benchmark(config.raw, brian)
    built.network.run(config.duration_ms * brian.ms)

    analysis_cfg = config.raw["analysis"]
    rate_monitor = built.rates[analysis_cfg["population"]]
    rate_hz = np.asarray(rate_monitor.smooth_rate(window="gaussian", width=analysis_cfg["smooth_ms"] * brian.ms) / brian.Hz)
    if not np.all(np.isfinite(rate_hz)):
        raise RuntimeError("Simulation produced non-finite rates; no result artifacts were written")
    time_s = np.asarray(rate_monitor.t / brian.second)
    sample_rate_hz = 1000.0 / config.dt_ms
    metrics = summarize_rate(rate_hz, sample_rate_hz, analysis_cfg["bands"])
    beta = metrics["beta_power"]
    gamma = metrics["gamma_power"]
    metrics["predicted_band_power_dominates"] = (
        gamma > beta if config.condition == "match" else beta > gamma
    )
    target_band = analysis_cfg["bands"]["gamma_hz" if config.condition == "match" else "beta_hz"]
    metrics["dominant_frequency_in_target_band"] = (
        target_band[0] <= metrics["dominant_frequency_hz"] <= target_band[1]
    )

    output_dir = Path(config.raw.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{config.condition}-seed{config.seed}-{config.fingerprint}"
    data_path = output_dir / f"{stem}.npz"
    summary_path = output_dir / f"{stem}.json"
    np.savez_compressed(data_path, time_s=time_s, rate_hz=rate_hz)
    summary = {
        "condition": config.condition,
        "seed": config.seed,
        "config_fingerprint": config.fingerprint,
        "model": config.raw["model"]["name"],
        "milestone": "M0_reduced_benchmark",
        "metrics": metrics,
        "versions": {
            "python": platform.python_version(),
            "brian2": brian.__version__,
            "numpy": np.__version__,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return data_path, summary_path
