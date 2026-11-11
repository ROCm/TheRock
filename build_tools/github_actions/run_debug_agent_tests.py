#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run the installed debug-agent suite once and retain its child diagnostics."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def run_tests(artifacts_dir: Path, diagnostics_dir: Path) -> int:
    artifacts_dir = artifacts_dir.resolve()
    test_dir = artifacts_dir / "tests" / "rocm-debug-agent"
    script = test_dir / "run-test.py"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    if not script.is_file():
        print(f"Debug-agent test runner is missing: {script}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    for variable, directory in (
        ("PATH", artifacts_dir / "bin"),
        ("LD_LIBRARY_PATH", artifacts_dir / "lib"),
    ):
        env[variable] = str(directory) + (
            os.pathsep + env[variable] if env.get(variable) else ""
        )

    # Remove a previous diagnostic file so a failed launch cannot report stale
    # evidence. The installed harness opens this file in write mode itself.
    log_path = test_dir / "run-test.log"
    log_path.unlink(missing_ok=True)
    try:
        result = subprocess.run(
            [sys.executable, str(script), str(test_dir)],
            cwd=test_dir,
            env=env,
            check=False,
        )
        exit_code = result.returncode
    except OSError as error:
        print(f"Cannot launch debug-agent tests: {error}", file=sys.stderr)
        exit_code = 1

    if log_path.is_file():
        destination = diagnostics_dir / "run-test.log"
        shutil.copyfile(log_path, destination)
        print(f"Debug-agent diagnostics preserved: {destination}", flush=True)
        if exit_code:
            print("--- debug-agent child diagnostics ---", flush=True)
            print(log_path.read_text(encoding="utf-8", errors="replace"), flush=True)
    else:
        print(
            f"Debug-agent runner produced no diagnostic log: {log_path}",
            file=sys.stderr,
        )

    # Convert a signal termination to the conventional shell exit status.
    return exit_code if exit_code >= 0 else 128 - exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_ARTIFACTS_DIR", "build")),
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=Path("test-diagnostics/rocr-debug-agent"),
    )
    args = parser.parse_args(argv)
    return run_tests(args.artifacts_dir, args.diagnostics_dir)


if __name__ == "__main__":
    sys.exit(main())
