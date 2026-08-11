from __future__ import annotations

import argparse

from .config import load_config
from .experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible SMART benchmark")
    parser.add_argument("config", help="YAML experiment configuration")
    args = parser.parse_args()
    data, summary = run_experiment(load_config(args.config))
    print(f"data: {data}")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
