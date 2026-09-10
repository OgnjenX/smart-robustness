"""Recover the shared-clock replay with layer-4 inhibitory spikes monitored."""

from __future__ import annotations

from collections.abc import Callable

import run_figure15_shared_clock_event_replay as original_replay

from smart_robustness.validation import figure15


def _with_interneuron_recording(run_condition: Callable) -> Callable:
    def monitored_run(*args, **kwargs):
        kwargs["record_interneuron_spikes"] = True
        return run_condition(*args, **kwargs)

    return monitored_run


def main() -> None:
    original_run = figure15.run_figure7_condition
    figure15.run_figure7_condition = _with_interneuron_recording(original_run)
    try:
        original_replay.main()
    finally:
        figure15.run_figure7_condition = original_run


if __name__ == "__main__":
    main()
