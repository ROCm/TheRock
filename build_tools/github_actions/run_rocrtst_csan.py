#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Separate the exhaustive reference-count case from the CSAN category suite."""

import argparse
import fnmatch
import json
import os
from pathlib import Path
import subprocess
import sys

MAX_CASE = "rocrtstFunc.Max_Reference_Count"


def split_command(command, mode):
    """Keep category exclusions and flags; partition only the exhaustive case."""
    filters = [arg for arg in command if arg.startswith("--gtest_filter=")]
    if len(filters) != 1:
        raise ValueError("Expected one explicit category --gtest_filter")
    original = filters[0].split("=", 1)[1]
    positive, _, negative = original.partition("-")
    contains_max = any(fnmatch.fnmatchcase(MAX_CASE, p) for p in positive.split(":"))
    excludes_max = any(fnmatch.fnmatchcase(MAX_CASE, p) for p in negative.split(":"))
    if not contains_max or excludes_max:
        raise ValueError("Selected category does not contain Max_Reference_Count")
    if mode == "max":
        replacement = MAX_CASE
    else:
        remaining = ":".join(p for p in positive.split(":") if p != MAX_CASE)
        if not remaining:
            raise ValueError("No regular category cases remain")
        excluded = ":".join(p for p in (negative, MAX_CASE) if p)
        replacement = remaining + "-" + excluded
    return [
        "--gtest_filter=" + replacement if arg == filters[0] else arg for arg in command
    ]


def calibrate(artifacts, diagnostics):
    compiler = artifacts / "bin" / "amdclang++"
    source = Path(__file__).with_name("rocrtst_refcount_probe.cpp")
    binary = diagnostics / "rocrtst-refcount-probe"
    subprocess.run(
        [
            str(compiler),
            "-O2",
            "-fsanitize=concurrency",
            "-fno-omit-frame-pointer",
            str(source),
            "-I" + str(artifacts / "include"),
            "-L" + str(artifacts / "lib"),
            "-Wl,-rpath," + str(artifacts / "lib"),
            "-lhsa-runtime64",
            "-o",
            str(binary),
        ],
        check=True,
        timeout=120,
    )
    print(
        "Calibrating 100000 real HSA references with the prepared CSAN preload",
        flush=True,
    )
    try:
        result = subprocess.run(
            [str(binary)], capture_output=True, text=True, timeout=180, check=False
        )
    except subprocess.TimeoutExpired as error:
        output = error.stdout or b""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        (diagnostics / "calibration.log").write_text(
            output + "\nCalibration timed out\n"
        )
        raise
    output = result.stdout + result.stderr
    (diagnostics / "calibration.log").write_text(output)
    print(output, flush=True)
    result.check_returncode()


def run_suite(artifacts, diagnostics, category, mode):
    artifacts = artifacts.resolve()
    diagnostics = diagnostics.resolve()
    diagnostics.mkdir(parents=True, exist_ok=True)
    test_dir = artifacts / "bin" / "rocrtst"
    result = subprocess.run(
        [
            "ctest",
            "--show-only=json-v1",
            "--test-dir",
            str(test_dir),
            "-L",
            "^" + category + "$",
            "-LE",
            "ex_gpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    tests = json.loads(result.stdout)["tests"]
    if len(tests) != 1:
        raise ValueError(f"Expected one rocrtst category command, got {len(tests)}")
    test = tests[0]
    command = split_command(test["command"], mode)
    properties = {p["name"]: p["value"] for p in test.get("properties", [])}
    env = os.environ.copy()
    for setting in properties.get("ENVIRONMENT", []):
        key, value = setting.split("=", 1)
        env[key] = value
    if mode == "max":
        calibrate(artifacts, diagnostics)
    (diagnostics / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    print(
        f"Running original {category} command, CSAN partition={mode}: {command}",
        flush=True,
    )
    result = subprocess.run(
        command,
        cwd=properties.get("WORKING_DIRECTORY", str(test_dir)),
        env=env,
        check=False,
    )
    return result.returncode if result.returncode >= 0 else 128 - result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("regular", "max"), required=True)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(os.getenv("OUTPUT_ARTIFACTS_DIR", "build")),
    )
    parser.add_argument(
        "--diagnostics-dir", type=Path, default=Path("test-diagnostics/rocrtst")
    )
    parser.add_argument(
        "--category",
        choices=("comprehensive", "full"),
        default=os.getenv("TEST_TYPE", "comprehensive"),
    )
    args = parser.parse_args(argv)
    try:
        return run_suite(
            args.artifacts_dir, args.diagnostics_dir, args.category, args.mode
        )
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"CSAN rocrtst partition failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
