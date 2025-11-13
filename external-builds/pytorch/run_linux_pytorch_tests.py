#!/usr/bin/env python3
"""PyTorch ROCm Pytest Runner with additional test exclusion capabilities.

This script runs PyTorch unit tests using pytest with additional test exclusion
capabilities tailored for AMD ROCm GPUs.

Test Exclusion Criteria
------------------------
Tests may be skipped based on:
- AMDGPU family compatibility (e.g., gfx942, gfx1151)
- PyTorch version-specific issues
- Known failures not yet upstreamed to PyTorch

Environment Variables
---------------------
THEROCK_ROOT_DIR : str, optional
                   Root directory of TheRock project.
                   If not set, auto-detected from script location.
AMDGPU_FAMILY :     str, optional
                    Target AMDGPU family for testing (e.g., "gfx942", "gfx94X").
                    Names should match those in "TheRock/cmake/therock_amdgpu_targets.cmake".
                    Supports wildcards (e.g., "gfx94X" matches any gfx94* architecture).
                    If not set, auto-detects from available hardware using amdgpu-arch.
PYTORCH_VERSION :   str, optional
                    PyTorch version for version-specific test filtering (e.g., "2.10").
                    Format: "major.minor" as string.
                    If not set, auto-detects from installed PyTorch package.

Usage Examples
--------------
Basic usage (auto-detect everything):
    $ python run_linux_pytorch_tests.py

Debug mode (run only skipped tests):
    $ python run_linux_pytorch_tests.py --debug

Custom test selection with pytest -k:
    $ python run_linux_pytorch_tests.py -k "test_nn and not test_dropout"

Disable pytest cache (useful in containers):
    $ python run_linux_pytorch_tests.py --no-cache

Exit Codes
----------
0 : All tests passed
1 : Test failures or collection errors
Other : Pytest-specific error codes

Side-effects
-----
- This script modifies PYTHONPATH and sys.path to include PyTorch test directory
- Creates a temporary MIOpen cache directory for each run
- Runs tests sequentially (--numprocesses=0) by default
"""

import argparse
import os
import subprocess
import sys
import tempfile

from skip_tests.create_skip_tests import *
from importlib.metadata import version
from pathlib import Path
from typing import Optional

import pytest


def setup_env(pytorch_dir: str) -> None:
    """Set up environment variables required for PyTorch testing with ROCm.

    Args:
        pytorch_dir: Path to the PyTorch directory containing test files.

    Side effects:
        - Sets multiple environment variables for PyTorch testing
        - Creates a temporary directory for MIOpen cache
        - Modifies sys.path to include the test directory
    """
    os.environ["PYTORCH_PRINT_REPRO_ON_FAILURE"] = "0"
    os.environ["PYTORCH_TEST_WITH_ROCM"] = "1"
    os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = tempfile.mkdtemp()
    os.environ["PYTORCH_TESTING_DEVICE_ONLY_FOR"] = "cuda"

    old_pythonpath = os.getenv("PYTHONPATH", "")
    test_dir = f"{pytorch_dir}/test"

    if old_pythonpath:
        os.environ["PYTHONPATH"] = f"{test_dir}:{old_pythonpath}"
    else:
        os.environ["PYTHONPATH"] = test_dir

    # Force update the PYTHONPATH to be part of the sys path
    # Otherwise our current python process that will run pytest will NOT
    # find it and pytest will crash!
    if test_dir not in sys.path:
        sys.path.insert(0, test_dir)


def cmd_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""
Runs PyTorch pytest for AMD GPUs. Skips additional tests compared to upstream.
Additional tests to be skipped can be tuned by PyTorch version and amdgpu family.
"""
    )

    amdgpu_family = os.getenv("AMDGPU_FAMILY")
    parser.add_argument(
        "--amdgpu-family",
        type=str,
        default=amdgpu_family if amdgpu_family is not None else "",
        required=False,
        help="""Amdgpu family (e.g. "gfx942").
Select (potentially) additional tests to be skipped based on the amdgpu family""",
    )

    pytorch_version = os.getenv("PYTORCH_VERSION")
    parser.add_argument(
        "--pytorch-version",
        type=str,
        default=pytorch_version if pytorch_version is not None else "",
        required=False,
        help="""Pytorch version (e.g. "2.7" or "all").
Select (potentially) additional tests to be skipped based on the Pytorch version.
'All' is also possible. Then all skip tests for all pytorch versions are included.
If no PyTorch version is given, it is auto-determined by the PyTorch used to run pytest.""",
    )

    env_root_dir = os.getenv("THEROCK_ROOT_DIR")
    parser.add_argument(
        "--the-rock-root-dir",
        default=env_root_dir if env_root_dir is not None else "",
        required=False,
        help="""Overwrites the root directory of TheRock.
By default TheRock root dir is determined based on this script's location.""",
    )

    parser.add_argument(
        "--debug",
        default=False,
        required=False,
        action=argparse.BooleanOptionalAction,
        help="""Inverts the selection. Only runs skipped tests.""",
    )

    parser.add_argument(
        "-k",
        default="",
        required=False,
        help="""Overwrites the pytest -k option that decides which tests should be run or skipped""",
    )

    parser.add_argument(
        "--no-cache",
        default=False,
        required=False,
        action=argparse.BooleanOptionalAction,
        help="""Disable pytest caching. Useful when only having read-only access to pytorch directory""",
    )

    args = parser.parse_args(argv)
    return args


def detect_amdgpu_family(amdgpu_family: str = "") -> Optional[str]:
    """Auto-detect or validate the AMDGPU family using the amdgpu-arch command.

    This function supports three modes of operation:
    1. If amdgpu_family is a specific GPU arch (e.g., "gfx1151"), returns it as-is
    2. If amdgpu_family is a family with wildcard (e.g., "gfx94X"), returns first match of amdgpu-arch output
    3. If amdgpu_family is empty, auto-detects from the first available GPU


    Returns:
        The detected or validated AMDGPU family string, or None if detection fails.
    """
    # Check if amdgpu_family is already set.
    # If it is a family with wildcard, extract prefix for partial matching
    partial_match = ""
    if amdgpu_family:
        family_part = amdgpu_family.split("-")[0]
        if family_part.upper().endswith("X"):
            # Extract prefix for partial matching (e.g., "gfx94X" -> "gfx94")
            partial_match = family_part[:-1]
        else:
            # Already a specific GPU arch, return as-is
            return amdgpu_family

    try:
        # Find executable in current python env
        print(f"Searching for amdgpu-arch in subdirectories of {Path(sys.executable).parent.parent}")
        proc = subprocess.run(["find", Path(sys.executable).parent.parent, "-name", "amdgpu-arch"], capture_output=True, text=True, check=False)
        amdgpu_arch_cmd = proc.stdout.strip()
        proc = subprocess.run(
            [amdgpu_arch_cmd], capture_output=True, text=True, check=False
        )

        if proc.returncode != 0 or proc.stderr:
            print(f"[ERROR] AMDGPU arch auto-detection FAILED: {proc.stderr}")
            return None

        available_gpus = [
            line.strip() for line in proc.stdout.split("\n") if line.strip()
        ]

        if not available_gpus:
            print("[ERROR] No AMD GPUs detected by amdgpu-arch")
            return None

        if amdgpu_family == "":
            # Auto-detect: use first available GPU
            detected_gpu = available_gpus[0]
            print(
                f"AMDGPU Arch auto-detected (based on GPU at index 0): {detected_gpu}"
            )
            return detected_gpu
        else:
            # Wildcard match: find first GPU matching the partial pattern
            for gpu in available_gpus:
                if partial_match in gpu:
                    print(
                        f"AMDGPU Arch auto-detected (based on partial match '{partial_match}'): {gpu}"
                    )
                    return gpu

            print(
                f"[ERROR] No GPU found matching pattern '{partial_match}'. GPUs found: {available_gpus}"
            )
            sys.exit(1)

    except FileNotFoundError:
        print("[ERROR] amdgpu-arch command not found. Is ROCm installed?")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error during AMDGPU detection: {e}")
        sys.exit(1)


def detect_pytorch_version() -> str:
    """Auto-detect the PyTorch version from the installed package.

    Returns:
        The detected PyTorch version as major.minor (e.g., "2.7").
    """
    # Get version, remove build suffix (+rocm, +cpu, etc.) and patch version
    return version("torch").rsplit("+", 1)[0].rsplit(".", 1)[0]


def determine_root_dir(provided_root: str) -> Path:
    """Determine the TheRock root directory.

    Args:
        provided_root: User-provided root directory path, or empty string.

    Returns:
        Path object representing the TheRock root directory.
    """
    if provided_root:
        return Path(provided_root)

    # Autodetect root dir via path of the script
    # We are in <TheRock Root Dir>/external-builds/pytorch
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent.parent


def main() -> int:
    """Main entry point for the PyTorch test runner.

    Returns:
        Exit code from pytest (0 for success, non-zero for failures).
    """
    args = cmd_arguments(sys.argv[1:])

    # Determine root directory
    root_dir = determine_root_dir(args.the_rock_root_dir)

    # Determine AMDGPU family
    amdgpu_family = detect_amdgpu_family(args.amdgpu_family)
    print(f"Using AMDGPU family: {amdgpu_family}")

    # Determine PyTorch version
    pytorch_version = args.pytorch_version
    if not pytorch_version:
        pytorch_version = detect_pytorch_version()
    print(f"Using PyTorch version: {pytorch_version}")

    # Get tests to skip
    tests_to_skip = get_tests(amdgpu_family, pytorch_version, not args.debug)

    # Allow manual override of test selection
    if args.k:
        tests_to_skip = args.k

    pytorch_dir = f"{root_dir}/external-builds/pytorch/pytorch"
    setup_env(pytorch_dir)

    pytorch_args = [
        f"{pytorch_dir}/test/test_nn.py",
        f"{pytorch_dir}/test/test_torch.py",
        f"{pytorch_dir}/test/test_cuda.py",
        f"{pytorch_dir}/test/test_unary_ufuncs.py",
        f"{pytorch_dir}/test/test_binary_ufuncs.py",
        f"{pytorch_dir}/test/test_autograd.py",
        "--continue-on-collection-errors",
        "--import-mode=importlib",
        f"-k={tests_to_skip}",
        # "-n 0",  # TODO does this need rework? why should we not run this multithreaded? this does not seem to exist?
        "-v",
        # -n numprocesses, --numprocesses=numprocesses
        #         Shortcut for '--dist=load --tx=NUM*popen'.
        #         With 'logical', attempt to detect logical CPU count (requires psutil, falls back to 'auto').
        #         With 'auto', attempt to detect physical CPU count. If physical CPU count cannot be determined, falls back to 1.
        #         Forced to 0 (disabled) when used with --pdb.
    ]

    if args.no_cache:
        pytorch_args += [
            "-p",
            "no:cacheprovider",  # Disable caching: useful when running in a container
        ]

    retcode = pytest.main(pytorch_args)
    print(f"Pytest finished with return code: {retcode}")
    return retcode


if __name__ == "__main__":
    # Lets make this script return pytest exit code (success or failure)
    sys.exit(main())
