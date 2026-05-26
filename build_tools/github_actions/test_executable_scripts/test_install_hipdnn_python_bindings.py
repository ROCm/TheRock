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

import argparse
import logging
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
PACK_WHEEL_SCRIPT = SCRIPT_DIR / "pack_python_wheel.py"
HIPDNN_PYTHON_TESTS_DIR = (
    THEROCK_DIR
    / "external-sources"
    / "rocm-libraries"
    / "projects"
    / "hipdnn"
    / "python"
    / "tests"
)

logging.basicConfig(level=logging.INFO)


def find_pkg_dir(artifacts_path: Path) -> Path:
    """Locate the hipdnn_frontend package directory in the test artifact."""
    candidate = artifacts_path / "share" / "hipdnn" / "python" / "hipdnn_frontend"
    if not candidate.is_dir():
        raise FileNotFoundError(
            f"hipdnn_frontend package not found at: {candidate}\n"
            "Ensure hipDNN was built with HIPDNN_BUILD_PYTHON_BINDINGS=ON"
        )
    return candidate


def build_wheel(pkg_dir: Path, wheel_dir: Path) -> Path:
    """Build a wheel from the staged package directory."""
    subprocess.check_call(
        [
            sys.executable,
            str(PACK_WHEEL_SCRIPT),
            "--pkg-dir",
            str(pkg_dir),
            "--wheel-dir",
            str(wheel_dir),
        ]
    )
    wheels = list(wheel_dir.glob("hipdnn_frontend-*.whl"))
    if not wheels:
        raise RuntimeError(f"No wheel produced in {wheel_dir}")
    return wheels[0]


def create_venv(venv_dir: Path) -> Path:
    """Create a virtual environment and return the python executable path."""
    subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
    if platform.system() == "Windows":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def install_wheel(python: Path, wheel_path: Path) -> None:
    """Install the wheel and pytest into the virtual environment."""
    subprocess.check_call(
        [str(python), "-m", "pip", "install", "--no-deps", str(wheel_path)]
    )
    subprocess.check_call([str(python), "-m", "pip", "install", "pytest>=7,<9"])


def validate_import(python: Path) -> None:
    """Verify the installed package can be imported."""
    subprocess.check_call(
        [str(python), "-c", "import hipdnn_frontend; print(hipdnn_frontend.__file__)"]
    )


def run_pytests(python: Path, artifacts_path: Path) -> bool:
    """Run the upstream hipDNN Python test suite. Returns True if tests ran."""
    if not HIPDNN_PYTHON_TESTS_DIR.is_dir():
        logging.warning(
            f"Skipping pytests: {HIPDNN_PYTHON_TESTS_DIR} not found "
            "(rocm-libraries submodule may not be initialized)"
        )
        return False

    env = os.environ.copy()
    is_windows = platform.system() == "Windows"
    if is_windows:
        rocm_lib = str(artifacts_path)
        env["PATH"] = f"{rocm_lib};{env.get('PATH', '')}"
    else:
        rocm_lib = str(artifacts_path / "lib")
        env["LD_LIBRARY_PATH"] = f"{rocm_lib}:{env.get('LD_LIBRARY_PATH', '')}"

    subprocess.check_call(
        [str(python), "-m", "pytest", "-v", str(HIPDNN_PYTHON_TESTS_DIR)],
        env=env,
    )
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test hipDNN Python bindings wheel packaging and installation"
    )
    parser.parse_args()

    if not OUTPUT_ARTIFACTS_DIR:
        raise RuntimeError("OUTPUT_ARTIFACTS_DIR environment variable not set")

    artifacts_path = Path(OUTPUT_ARTIFACTS_DIR).resolve()
    logging.info(f"Using OUTPUT_ARTIFACTS_DIR: {artifacts_path}")

    pkg_dir = find_pkg_dir(artifacts_path)
    logging.info(f"Found hipdnn_frontend at: {pkg_dir}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        wheel_dir = tmp_path / "wheels"
        wheel_dir.mkdir()
        venv_dir = tmp_path / "venv"

        wheel_path = build_wheel(pkg_dir, wheel_dir)
        logging.info(f"Built wheel: {wheel_path.name}")

        python = create_venv(venv_dir)
        logging.info(f"Created venv: {venv_dir}")

        install_wheel(python, wheel_path)
        logging.info("Wheel installed successfully")

        validate_import(python)
        logging.info("Import validation passed")

        tests_ran = run_pytests(python, artifacts_path)

    if not tests_ran:
        logging.error("Python test suite was skipped — test directory not found")
        sys.exit(2)

    logging.info("All hipDNN Python bindings tests passed!")
