#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
hipDNN Python bindings wheel install and smoke test.

This test verifies that the hipdnn-frontend wheel built by TheRock can be
installed into a fresh venv and successfully imported.

Environment variables:
    OUTPUT_ARTIFACTS_DIR: Path to the TheRock dist/rocm output directory
                         that contains share/hipdnn/wheels/*.whl
"""

import argparse
import logging
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = Path(
    os.environ.get("THEROCK_DIR") or SCRIPT_DIR.parent.parent.parent
).resolve()

logging.basicConfig(level=logging.INFO)


def find_wheel(artifacts_path: Path) -> Path:
    """Locate the hipdnn-frontend wheel under the artifacts directory."""
    wheel_dir = artifacts_path / "share" / "hipdnn" / "wheels"
    wheels = sorted(wheel_dir.glob("hipdnn_frontend-*.whl"))
    if not wheels:
        raise FileNotFoundError(
            f"No hipdnn-frontend wheel found in {wheel_dir}. "
            "Ensure the build was configured with -DHIPDNN_BUILD_PYTHON_BINDINGS=ON."
        )
    logging.info(f"Found wheel: {wheels[-1]}")
    return wheels[-1]


def create_venv(venv_dir: Path) -> Path:
    """Create a virtual environment and return the python executable path."""
    logging.info(f"Creating virtual environment in {venv_dir}")
    venv.create(venv_dir, with_pip=True)

    if platform.system() == "Windows":
        python = venv_dir / "Scripts" / "python.exe"
    else:
        python = venv_dir / "bin" / "python"

    if not python.exists():
        raise RuntimeError(f"venv python not found at {python}")
    return python


def install_wheel(python: Path, wheel: Path, artifacts_path: Path) -> None:
    """Install the wheel into the venv."""
    env = os.environ.copy()

    if platform.system() == "Windows":
        lib_path = str(artifacts_path)
        env["PATH"] = f"{lib_path};{env.get('PATH', '')}"
    else:
        lib_path = str(artifacts_path / "lib")
        env["LD_LIBRARY_PATH"] = f"{lib_path}:{env.get('LD_LIBRARY_PATH', '')}"

    cmd = [str(python), "-m", "pip", "install", str(wheel)]
    logging.info(f"++ {shlex.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def run_smoke_tests(python: Path, artifacts_path: Path) -> None:
    """Run inline Python smoke tests inside the venv."""
    env = os.environ.copy()

    if platform.system() == "Windows":
        lib_path = str(artifacts_path)
        env["PATH"] = f"{lib_path};{env.get('PATH', '')}"
    else:
        lib_path = str(artifacts_path / "lib")
        env["LD_LIBRARY_PATH"] = f"{lib_path}:{env.get('LD_LIBRARY_PATH', '')}"

    test_script = r"""
import hipdnn_frontend as fe
print(f"OK import hipdnn_frontend (version {fe.__version__})")
"""

    cmd = [str(python), "-c", test_script]
    logging.info("Running smoke tests...")
    subprocess.run(cmd, check=True, env=env)


def run_tests(artifacts_path: Path, venv_dir: Path) -> None:
    """Find wheel, create venv, install, and run tests."""
    wheel = find_wheel(artifacts_path)
    python = create_venv(venv_dir)
    install_wheel(python, wheel, artifacts_path)
    run_smoke_tests(python, artifacts_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Install and test hipDNN Python bindings wheel"
    )
    parser.add_argument(
        "--venv-dir",
        type=Path,
        help="Directory for the test virtual environment. "
        "If not specified, uses a temporary directory that is auto-deleted.",
    )
    args = parser.parse_args()

    if not OUTPUT_ARTIFACTS_DIR:
        raise RuntimeError("OUTPUT_ARTIFACTS_DIR environment variable not set")

    artifacts_path = Path(OUTPUT_ARTIFACTS_DIR).resolve()
    logging.info(f"Using OUTPUT_ARTIFACTS_DIR: {artifacts_path}")

    if args.venv_dir:
        venv_dir = args.venv_dir.resolve()
        venv_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Using persistent venv directory: {venv_dir}")
        run_tests(artifacts_path, venv_dir)
        logging.info(f"Venv retained in: {venv_dir}")
    else:
        logging.info("Using temporary venv directory (auto-cleanup)")
        with tempfile.TemporaryDirectory() as temp_dir:
            run_tests(artifacts_path, Path(temp_dir) / "venv")

    logging.info("All hipDNN Python binding tests passed!")
