#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
hipDNN Python bindings wheel test.

This test validates the hipDNN Python wheel packaging pipeline:
1. Builds a wheel from the staged Python bindings in the test artifact
2. Installs the wheel into a temporary virtual environment
3. Runs the upstream pytest suite against the installed bindings

Requires a GPU and OUTPUT_ARTIFACTS_DIR pointing at the merged artifact tree.
"""

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from libhipcxx_utils import build_rocm_loader_env

OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
PACK_WHEEL_SCRIPT = SCRIPT_DIR / "hipdnn" / "pack_frontend_wheel.py"

# build_tools/ on sys.path so we can reuse the shared venv helpers instead of
# re-implementing platform-specific python-exe discovery here.
_BUILD_TOOLS_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))
from setup_venv import create_venv, find_venv_python_exe  # noqa: E402

_HIPDNN_SHARE_RELPATH = Path("share/hipdnn")
_HIPDNN_TESTS_ARTIFACT_RELPATH = _HIPDNN_SHARE_RELPATH / "tests" / "python"
_HIPDNN_PKG_ARTIFACT_RELPATH = _HIPDNN_SHARE_RELPATH / "python" / "hipdnn_frontend"

# Per-step timeouts (seconds). Bounded so a hung GPU / deadlocked pytest fails
# the step instead of consuming the full CI matrix budget.
_TIMEOUT_WHEEL_BUILD = 5 * 60
_TIMEOUT_PIP_INSTALL = 5 * 60
_TIMEOUT_PYTEST = 20 * 60


logging.basicConfig(level=logging.INFO)


def _require_artifact_dir(
    artifacts_path: Path, relpath: Path, label: str, hint: str
) -> Path:
    """Resolve an artifact subdirectory and fail loudly if it is missing.

    Failing upfront keeps a missing artifact from silently masking itself as
    a green run later in the pipeline.
    """
    candidate = (artifacts_path / relpath).resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(f"{label} not found at: {candidate}\n{hint}")
    return candidate


def build_wheel(pkg_dir: Path, wheel_dir: Path) -> Path:
    """Build a wheel from the staged package directory."""
    subprocess.run(
        [
            sys.executable,
            str(PACK_WHEEL_SCRIPT),
            "--pkg-dir",
            str(pkg_dir),
            "--wheel-dir",
            str(wheel_dir),
        ],
        check=True,
        timeout=_TIMEOUT_WHEEL_BUILD,
    )
    wheels = list(wheel_dir.glob("hipdnn_frontend-*.whl"))
    if not wheels:
        raise RuntimeError(f"No wheel produced in {wheel_dir}")
    return wheels[0]


def install_wheel(python: Path, wheel_path: Path) -> None:
    """Install the wheel and test dependencies into the virtual environment.

    Two calls: `--no-deps` is per-invocation, not per-requirement, so the
    wheel install must be isolated from the test-dependency resolution.
    numpy is required by the pytest suite for array construction.
    """
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", str(wheel_path)],
        check=True,
        timeout=_TIMEOUT_PIP_INSTALL,
    )
    subprocess.run(
        [str(python), "-m", "pip", "install", "pytest>=7,<9", "numpy"],
        check=True,
        timeout=_TIMEOUT_PIP_INSTALL,
    )


def run_pytests(python: Path, tests_dir: Path, env: dict) -> None:
    """Run the upstream hipDNN Python test suite."""
    # Pin cwd so pytest discovery cannot pick up a sibling conftest.py.
    subprocess.run(
        [str(python), "-m", "pytest", "-v", str(tests_dir)],
        env=env,
        cwd=str(tests_dir),
        check=True,
        timeout=_TIMEOUT_PYTEST,
    )


if __name__ == "__main__":
    if not OUTPUT_ARTIFACTS_DIR:
        raise RuntimeError("OUTPUT_ARTIFACTS_DIR environment variable not set")

    artifacts_path = Path(OUTPUT_ARTIFACTS_DIR).resolve()
    logging.info(f"Using OUTPUT_ARTIFACTS_DIR: {artifacts_path}")

    tests_dir = _require_artifact_dir(
        artifacts_path,
        _HIPDNN_TESTS_ARTIFACT_RELPATH,
        "hipDNN upstream pytest directory",
        "Ensure the hipDNN test artifact includes share/hipdnn/tests/python.",
    )
    logging.info(f"Using hipDNN pytest dir: {tests_dir}")

    pkg_dir = _require_artifact_dir(
        artifacts_path,
        _HIPDNN_PKG_ARTIFACT_RELPATH,
        "hipdnn_frontend package",
        "Ensure hipDNN was built with HIPDNN_BUILD_PYTHON_BINDINGS=ON.",
    )
    logging.info(f"Found hipdnn_frontend at: {pkg_dir}")

    env = build_rocm_loader_env(artifacts_path)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        wheel_dir = tmp_path / "wheels"
        wheel_dir.mkdir()
        venv_dir = tmp_path / "venv"

        wheel_path = build_wheel(pkg_dir, wheel_dir)
        logging.info(f"Built wheel: {wheel_path.name}")

        create_venv(venv_dir)
        python = find_venv_python_exe(venv_dir)
        if python is None:
            raise RuntimeError(f"venv created but no python executable in {venv_dir}")
        logging.info(f"Created venv: {venv_dir}")

        install_wheel(python, wheel_path)
        logging.info("Wheel installed successfully")

        run_pytests(python, tests_dir, env)

    logging.info("All hipDNN Python bindings tests passed!")
