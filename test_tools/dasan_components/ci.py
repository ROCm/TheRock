#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Install component-owned test dependencies, then execute a CI test group."""

import argparse
from pathlib import Path
import subprocess
import sys

GROUPS = {
    "runtime": [
        "dasan",
        "sanity",
        "rocrtst",
        "hip-smoke",
        "amdsmi-quick",
        "rdc-unit",
        "rocgdb-version",
    ],
    "communications": [
        "rccl-unit",
        "rccl-allreduce",
        "rocshmem-smoke",
        "rocshmem-info",
    ],
    "libraries": [
        "hipfile-unit",
        "rpp-quick",
        "hipdnn",
        "hipdnn-install",
        "rocdecode-correctness",
        "rocjpeg-correctness",
    ],
    "profiler-sdk": ["profiler-sdk", "aqlprofile", "profiler-hub-install"],
    "profiler-systems": ["rocprofiler-systems-quick"],
    "profiler-compute": ["rocprofiler-compute-gpu"],
}
REQUIREMENTS = {
    "profiler-sdk": [
        "share/rocprofiler-sdk/tests/requirements.txt",
        "share/rocprofiler-sdk/requirements.txt",
    ],
    "profiler-systems": ["share/rocprofiler-systems/tests/requirements.txt"],
    "profiler-compute": [
        "libexec/rocprofiler-compute/requirements.txt",
        "libexec/rocprofiler-compute/requirements-test.txt",
    ],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", choices=GROUPS, required=True)
    p.add_argument("--prefix", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parent.parent
    requirements = REQUIREMENTS.get(a.group, [])
    if requirements:
        command = [sys.executable, "-m", "pip", "install", "perfetto==0.7.0"]
        for path in requirements:
            path = a.prefix.resolve() / path
            if not path.is_file():
                raise FileNotFoundError(path)
            command += ["-r", str(path)]
        subprocess.run(command, check=True)
    # Complete dependency installation before any installed CMake configure.
    command = [
        sys.executable,
        str(here / "run_validation.py"),
        "--manifest",
        str(here / "suites.json"),
        "--prefix",
        str(a.prefix),
        "--repo",
        str(repo),
        "--bundle",
        str(a.bundle),
        "--output",
        str(a.output),
    ]
    for suite in GROUPS[a.group]:
        command += ["--suite", suite]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
