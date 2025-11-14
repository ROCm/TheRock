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
- Sets HIP_VISIBLE_DEVICES environment variable to select specific GPU(s) for testing
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


def detect_amdgpu_family(amdgpu_family: str = "") -> list[str]:
    """Detect and configure AMDGPU family using amdgpu-arch command.

    This function always queries amdgpu-arch to get available GPUs and sets
    HIP_VISIBLE_DEVICES to select the appropriate GPU(s) for testing.

    Args:
        amdgpu_family: AMDGPU family string. Can be:
            - Empty string (default): Auto-detect and use first GPU (index 0)
            - Specific arch (e.g., "gfx1151"): Find and use matching GPU
            - Wildcard family (e.g., "gfx94X"): Find all matching GPUs

    Returns:
        List of detected AMDGPU family strings. Exits on failure.

    Side effects:
        Sets HIP_VISIBLE_DEVICES environment variable to comma-separated GPU indices.
    """
    try:
        # Find amdgpu-arch executable in current python environment
        print(
            f"Searching for amdgpu-arch in subdirectories of {Path(sys.executable).parent.parent}"
        )
        proc = subprocess.run(
            ["find", Path(sys.executable).parent.parent, "-name", "amdgpu-arch"],
            capture_output=True,
            text=True,
            check=False,
        )
        # There might be 2 matches: rocm_sdk_core and rocm_sdk_devel, so just take the first one
        amdgpu_arch_cmd = proc.stdout.split("\n")[0].strip()

        if not amdgpu_arch_cmd:
            print("[ERROR] amdgpu-arch command not found in Python environment")
            sys.exit(1)

        # Query available GPUs
        proc = subprocess.run(
            [amdgpu_arch_cmd], capture_output=True, text=True, check=False
        )

        if proc.returncode != 0 or proc.stderr:  # or proc.stdout == "\n":
            print(f"[ERROR] AMDGPU arch detection FAILED: {proc.stderr}")
            sys.exit(1)

        available_gpus = [
            line.strip() for line in proc.stdout.split("\n") if line.strip()
        ]

        if not available_gpus:
            print("[ERROR] No AMD GPUs detected by amdgpu-arch")
            sys.exit(1)

        print(f"Available AMD GPUs: {available_gpus}")

        # Determine which GPU to use based on input
        selected_gpu_indices = []
        selected_gpu_archs = []

        if not amdgpu_family:
            # Mode 1: Auto-detect - use first available GPU
            selected_gpu_indices = [0]
            selected_gpu_archs = [available_gpus[0]]
            print(
                f"AMDGPU Arch auto-detected (using GPU at index 0): {selected_gpu_archs}"
            )
        elif amdgpu_family.split("-")[0].upper().endswith("X"):
            # Mode 2: Wildcard match (e.g., "gfx94X" matches "gfx942", "gfx940", etc.)
            family_part = amdgpu_family.split("-")[0]
            partial_match = family_part[:-1]  # Remove the 'X'

            for idx, gpu in enumerate(available_gpus):
                if partial_match in gpu:
                    selected_gpu_indices += [idx]
                    selected_gpu_archs += [gpu]
            print(
                f"AMDGPU Arch detected via wildcard match '{partial_match}': "
                f"{selected_gpu_archs} (GPU indices {selected_gpu_indices})"
            )

            if len(selected_gpu_archs) == 0:
                print(
                    f"[ERROR] No GPU found matching wildcard pattern '{amdgpu_family}'. "
                    f"Available GPUs: {available_gpus}"
                )
                sys.exit(1)
        else:
            # Mode 3: Specific GPU arch - validate it exists in available GPUs
            for idx, gpu in enumerate(available_gpus):
                if gpu == amdgpu_family or amdgpu_family in gpu:
                    selected_gpu_indices += [idx]
                    selected_gpu_archs += [gpu]
                    print(
                        f"AMDGPU Arch validated: {selected_gpu_archs} "
                        f"(GPU indices {selected_gpu_indices})"
                    )
                    break

            if selected_gpu_archs is None:
                print(
                    f"[ERROR] Requested GPU '{amdgpu_family}' not found in available GPUs. "
                    f"Available GPUs: {available_gpus}"
                )
                sys.exit(1)

        # Set HIP_VISIBLE_DEVICES to select the specific GPU
        str_indices = ",".join(str(idx) for idx in selected_gpu_indices)
        os.environ["HIP_VISIBLE_DEVICES"] = str_indices
        print(f"Set HIP_VISIBLE_DEVICES={str_indices}")

        return selected_gpu_archs

    except FileNotFoundError as e:
        print(f"[ERROR] Command not found: {e}")
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
