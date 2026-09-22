#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Installs what the JAX test suite needs into the active environment.

The requirements come from the ROCm/jax checkout, pinned by its per-Python lock
file, so the versions match the ones the release was tested with.
pytest-json-report is ours: the fresh-process retry in run_jax_tests.py reads the
reports it writes.

Installs are retried via build_tools/setup_venv.py, since one index timeout has
cost a job a whole run.

Example usage:

    python install_jax_test_requirements.py --jax-dir jax --python-version 3.12
"""

import argparse
import os
import pathlib
import sys

THIS_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
THEROCK_DIR = THIS_SCRIPT_DIR.parent.parent

sys.path.insert(0, os.fspath(THEROCK_DIR / "build_tools"))

from setup_venv import run_command_with_retries

# uv resolves the JAX requirements files considerably faster than pip.
UV_REQUIREMENT = "uv~=0.11.2"

# Written by pytest-json-report, read by the fresh-process retry.
REPORT_REQUIREMENT = "pytest-json-report"

RETRY_TIMEOUT_SECONDS = 180
RETRY_WAIT_BETWEEN_SECONDS = 15


def lock_file(jax_dir: pathlib.Path, python_version: str) -> pathlib.Path:
    """The JAX requirements lock file for one Python version."""
    return (
        jax_dir / "build" / f"requirements_lock_{python_version.replace('.', '_')}.txt"
    )


def install_commands(jax_dir: pathlib.Path, python_version: str) -> list[list[str]]:
    """The install commands, in the order they have to run."""
    python = sys.executable
    return [
        [python, "-m", "pip", "install", UV_REQUIREMENT],
        [
            python,
            "-m",
            "uv",
            "pip",
            "install",
            "-r",
            os.fspath(jax_dir / "build" / "test-requirements.txt"),
        ],
        # The suite is sensitive to these two versions, so pin them to the lock
        # file rather than whatever the requirements resolve to.
        [
            python,
            "-m",
            "uv",
            "pip",
            "install",
            "--upgrade",
            "-c",
            os.fspath(lock_file(jax_dir, python_version)),
            "scipy",
            "pytest",
        ],
        [python, "-m", "pip", "install", REPORT_REQUIREMENT],
    ]


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="install_jax_test_requirements.py")
    p.add_argument(
        "--jax-dir",
        type=pathlib.Path,
        default=pathlib.Path(os.getenv("JAX_DIR", "jax")),
        help="ROCm/jax checkout holding the requirements files (default: jax)",
    )
    p.add_argument(
        "--python-version",
        default=os.getenv("PYTHON_VERSION", ""),
        help="Python version selecting the lock file (e.g. 3.12)",
    )
    p.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the commands without running them",
    )
    args = p.parse_args(argv)

    if not args.python_version:
        p.error("--python-version is required to select the lock file")

    jax_dir = args.jax_dir.resolve()
    lock = lock_file(jax_dir, args.python_version)
    if not lock.exists() and not args.dry_run:
        print(f"::error::{lock} not found. Is --jax-dir a ROCm/jax checkout?")
        return 1

    for command in install_commands(jax_dir, args.python_version):
        if args.dry_run:
            print(f"++ Would exec $ {' '.join(command)}")
            continue
        run_command_with_retries(
            command,
            retry_timeout_seconds=RETRY_TIMEOUT_SECONDS,
            retry_wait_between_seconds=RETRY_WAIT_BETWEEN_SECONDS,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
