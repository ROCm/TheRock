# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Configure installed media tests without enabling unavailable FFmpeg tests."""

import argparse
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--prefix", type=Path, required=True)
p.add_argument("--build", type=Path, required=True)
p.add_argument("--component", choices=["rocdecode", "rocjpeg"], required=True)
p.add_argument("--junit", required=True)
a = p.parse_args()
command = [
    "cmake",
    "-S",
    str(a.prefix / "share" / a.component / "test"),
    "-B",
    str(a.build),
    "-G",
    "Ninja",
    f"-DCMAKE_PREFIX_PATH={a.prefix};{a.prefix}/lib/rocm_sysdeps",
]
if a.component == "rocdecode":
    command.append("-DENABLE_EXTENDED_TESTS=OFF")
test_command = [
    "ctest",
    "--test-dir",
    str(a.build),
    "--no-tests=error",
    "--output-on-failure",
    "--timeout",
    "600",
    "--output-junit",
    a.junit,
]
if a.component == "rocjpeg":
    test_command += ["-E", "jpeg-decode-perf"]
for command in [command, test_command]:
    print(command, flush=True)
    subprocess.run(command, check=True)
