"""Practical exact-equation execution for large Brian2 SMART networks."""

from __future__ import annotations

import os
import subprocess
import sysconfig
from pathlib import Path


def build_and_run_cpp_standalone(
    brian,
    directory: str | Path,
    *,
    jobs: int | None = None,
) -> None:
    """Generate, compile, run, and reload a queued Brian2 standalone model.

    Brian2's generated ``objects.cpp`` only loads static arrays and initializes
    storage. For dense SMART connectivity, optimizing that very large file at
    ``-O3`` takes many minutes without affecting simulation speed. Compile it
    at ``-O0`` while retaining Brian2's ``-O3`` flags for every numerical code
    object, then let the generated makefile link the complete executable.
    """

    project = Path(directory).resolve()
    if project == Path(project.anchor):
        raise ValueError("standalone directory must not be a filesystem root")
    project.mkdir(parents=True, exist_ok=True)
    brian.device.build(directory=str(project), compile=False, run=False)

    objects_source = project / "objects.cpp"
    makefile = project / "makefile"
    if not objects_source.is_file() or not makefile.is_file():
        raise RuntimeError("Brian2 did not generate the expected standalone project")

    compiler = os.environ.get("CXX", "c++")
    include = Path(sysconfig.get_paths()["include"])
    subprocess.run(
        [
            compiler,
            "-c",
            "-Wno-write-strings",
            f"-I{include}",
            "-w",
            "-O0",
            "-fno-finite-math-only",
            "-std=c++11",
            "-I.",
            "objects.cpp",
            "-o",
            "objects.o",
        ],
        cwd=project,
        check=True,
    )
    worker_count = jobs or min(8, os.cpu_count() or 1)
    subprocess.run(
        ["make", f"-j{worker_count}"],
        cwd=project,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    brian.device.run(directory=str(project), results_directory="results")
