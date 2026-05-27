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


def build_runtime_env(artifacts_path: Path) -> dict:
    """Construct env exposing the ROCm loader path for the native extension.

    The `hipdnn_frontend_python` extension links against `libhipdnn_backend`
    (and transitively HIP/ROCm libs). Without LD_LIBRARY_PATH (Linux) or PATH
    (Windows) pointing at the artifact tree, `import hipdnn_frontend` fails in
    a clean venv. Shared by `validate_import` and `run_pytests` so both run
    against the same loader configuration.
    """
    env = os.environ.copy()
    if platform.system() == "Windows":
        # Windows ROCm DLLs live in bin/ (Linux uses lib/), so PATH must
        # include artifacts_path / "bin" for the loader to find them.
        rocm_lib = str(artifacts_path / "bin")
        env["PATH"] = f"{rocm_lib};{env.get('PATH', '')}"
    else:
        rocm_lib = str(artifacts_path / "lib")
        env["LD_LIBRARY_PATH"] = f"{rocm_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


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
    """Install the wheel and pytest into the virtual environment.

    Two calls: `--no-deps` is per-invocation, not per-requirement, so the
    wheel install must be isolated from pytest's dependency resolution.
    """
    subprocess.check_call(
        [str(python), "-m", "pip", "install", "--no-deps", str(wheel_path)]
    )
    subprocess.check_call([str(python), "-m", "pip", "install", "pytest>=7,<9"])


def validate_import(python: Path, cwd: Path, env: dict) -> None:
    """Verify the installed package can be imported.

    Runs from `cwd` (a neutral directory) so that `import hipdnn_frontend`
    cannot accidentally resolve to a sibling staged package directory. Uses
    the same loader env as `run_pytests` so the native extension can resolve
    `libhipdnn_backend` and its transitive ROCm deps.
    """
    subprocess.check_call(
        [str(python), "-c", "import hipdnn_frontend; print(hipdnn_frontend.__file__)"],
        cwd=cwd,
        env=env,
    )


def run_pytests(python: Path, env: dict) -> None:
    """Run the upstream hipDNN Python test suite."""
    # Pin cwd so pytest discovery cannot pick up a sibling conftest.py.
    subprocess.check_call(
        [str(python), "-m", "pytest", "-v", str(HIPDNN_PYTHON_TESTS_DIR)],
        env=env,
        cwd=str(HIPDNN_PYTHON_TESTS_DIR),
    )


if __name__ == "__main__":
    if not OUTPUT_ARTIFACTS_DIR:
        raise RuntimeError("OUTPUT_ARTIFACTS_DIR environment variable not set")

    artifacts_path = Path(OUTPUT_ARTIFACTS_DIR).resolve()
    logging.info(f"Using OUTPUT_ARTIFACTS_DIR: {artifacts_path}")

    # Fail upfront on missing test infra so a missing submodule cannot mask
    # itself as a green run.
    if not HIPDNN_PYTHON_TESTS_DIR.is_dir():
        raise FileNotFoundError(
            f"hipDNN upstream pytest directory not found: {HIPDNN_PYTHON_TESTS_DIR}. "
            "Initialize the rocm-libraries submodule."
        )

    pkg_dir = find_pkg_dir(artifacts_path)
    logging.info(f"Found hipdnn_frontend at: {pkg_dir}")

    env = build_runtime_env(artifacts_path)

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

        validate_import(python, tmp_path, env)
        logging.info("Import validation passed")

        run_pytests(python, env)

    logging.info("All hipDNN Python bindings tests passed!")
