#!/usr/bin/env python3
"""PyTorch ROCm Pytest Runner with additional test exclusion capabilities.

This script runs PyTorch unit tests using pytest with additional test exclusion
capabilities tailored for AMD ROCm GPUs.

Test Exclusion Criteria
------------------------
Tests may be skipped based on:
- AMDGPU family compatibility (e.g., gfx942, gfx1151)
- PyTorch version-specific issues
- Platform (Linux, Windows)
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
                    If not set, auto-detects from available hardware using PyTorch.
PYTORCH_VERSION :   str, optional
                    PyTorch version for version-specific test filtering (e.g., "2.10").
                    Format: "major.minor" as string.
                    If not set, auto-detects from installed PyTorch package.
HIP_VISIBLE_DEVICES : str, optional (read/write)
                      If already set, the script respects this constraint and only selects
                      from the GPUs visible within this limitation (e.g., in containers).
                      The script will further filter and update this variable based on
                      the AMDGPU_FAMILY selection or auto-detection.

Usage Examples
--------------
Basic usage (auto-detect everything):
    $ python run_pytorch_tests.py

Debug mode (run only skipped tests):
    $ python run_pytorch_tests.py --debug

Custom test selection with pytest -k:
    $ python run_pytorch_tests.py -k "test_nn and not test_dropout"

Disable pytest cache (useful in containers):
    $ python run_pytorch_tests.py --no-cache

Exit Codes
----------
0 : All tests passed
1 : Test failures or collection errors
15: SIGTERM for Windows (see notes below)
Other : Pytest-specific error codes

Side-effects
------------
- This script modifies PYTHONPATH and sys.path to include PyTorch test directory
- Creates a temporary MIOpen cache directory for each run
- Sets HIP_VISIBLE_DEVICES environment variable to select specific GPU(s) for testing
- Runs tests sequentially (--numprocesses=0) by default

Windows special notes
---------------------
To work around https://github.com/ROCm/TheRock/issues/999, this script
writes 'exit_code.txt' to the current directory and then kills the process.
"""

import argparse
import os
import platform
import subprocess
import sys
import tempfile

from skip_tests.create_skip_tests import *
from importlib.metadata import version
from pathlib import Path

import pytest

THIS_SCRIPT_DIR = Path(__file__).resolve().parent


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

    default_pytorch_dir = THIS_SCRIPT_DIR / "pytorch"
    parser.add_argument(
        "--pytorch-dir",
        type=Path,
        default=default_pytorch_dir,
        help="Path for the pytorch repository, where tests will be sourced from",
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

    if not args.pytorch_dir.exists():
        parser.error(
            f"Directory at '{args.pytorch_dir}' does not exist, checkout pytorch and then set the path via --pytorch-dir"
        )

    return args


def get_visible_gpus() -> list[str]:
    """Get a list of GPUs that are visible for torch.

    Note that the current torch build does not necessarily have
    support for all of the GPUs that are visible.
    The list of GPUs that are supported by the current torch build
    can be queried with method torch.cuda.get_arch_list().

    This function runs in a subprocess to avoid initializing CUDA
    in the main process before HIP_VISIBLE_DEVICES is set.

    Important: If HIP_VISIBLE_DEVICES is already set before calling this script,
    this function will only see GPUs within that constraint. This allows the
    script to work within pre-configured limitations (e.g., in containers).

    Returns:
        List of AMDGPU family strings visible (e.g., ["gfx942", "gfx1100"]).
        Exits on failure.
    """
    query_script = """
import sys
try:
    import torch
    visible_gpus = []
    if not torch.cuda.is_available():
        print("ERROR:ROCm is not available", file=sys.stderr)
        sys.exit(1)

    gpu_count = torch.cuda.device_count()
    print(f"GPU count visible for PyTorch: {gpu_count}", file=sys.stderr)

    for device_idx in range(gpu_count):
        device_id = f"cuda:{device_idx}"
        device = torch.cuda.device(device_id)
        if device:
            device_properties = torch.cuda.get_device_properties(device)
            if device_properties and hasattr(device_properties, 'gcnArchName'):
                # AMD GPUs have gcnArchName
                visible_gpus.append(device_properties.gcnArchName)

    if len(visible_gpus) == 0:
        print("No AMD GPUs with gcnArchName detected", file=sys.stderr)
        sys.exit(1)

    # Print one GPU per line for easy parsing
    for gpu in visible_gpus:
        print(gpu)
except Exception as e:
    print(f"{e}", file=sys.stderr)
    sys.exit(1)
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", query_script],
            capture_output=True,
            text=True,
            check=True,
        )
        visible_gpus = result.stdout.strip().split("\n")
        return visible_gpus
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Failed to retrieve visible GPUs: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error retrieving visible GPUs: {e}")
        sys.exit(1)


def get_supported_gpus() -> list[str]:
    """Get a list of AMD GPUs that are supported by the current PyTorch build.

    Returns:
        List of PyTorch supported GPU architecture strings (e.g., ["gfx942", "gfx1100"]).
        Exits on failure.
    """
    query_script = """
import sys
try:
    import torch
    if not torch.cuda.is_available():
        print("ROCm is not available", file=sys.stderr)
        sys.exit(1)
    gpus = torch.cuda.get_arch_list()
    if len(gpus) == 0:
        print("No AMD GPUs detected", file=sys.stderr)
        sys.exit(1)
    # Print one GPU per line for easy parsing
    for gpu in gpus:
        print(gpu)
except Exception as e:
    print(f"ERROR:{e}", file=sys.stderr)
    sys.exit(1)
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", query_script],
            capture_output=True,
            text=True,
            check=True,
        )
        available_gpus = result.stdout.strip().split("\n")
        return available_gpus
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Failed to retrieve available GPUs: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error retrieving available GPUs: {e}")
        sys.exit(1)


def detect_amdgpu_family(amdgpu_family: str = "") -> list[str]:
    """Detect and configure AMDGPU family for testing.

    This function queries available GPUs and sets HIP_VISIBLE_DEVICES BEFORE
    PyTorch/CUDA is initialized in the main process via pytest.

    Args:
        amdgpu_family: AMDGPU family string. Can be:
            - Empty string (default): Auto-detect first visible GPU supported by PyTorch
            - Specific arch (e.g., "gfx1151"): Find and use matching GPU
            - Wildcard family (e.g., "gfx94X"): Find all matching GPUs

    Returns:
        List of detected AMDGPU family strings. Exits on failure.

    Side effects:
        - Reads HIP_VISIBLE_DEVICES if already set (respects pre-configured constraints)
        - Updates HIP_VISIBLE_DEVICES to further filter GPU selection
        - This MUST be called before importing torch in the main process via pytest
    """

    # Get the current HIP_VISIBLE_DEVICES to properly map indices
    # If already set (e.g., "2,3,4"), visible GPU indices are remapped (0,1,2)
    # We need to track the original system indices for correct remapping
    current_hip_visible = os.environ.get("HIP_VISIBLE_DEVICES", "")
    if current_hip_visible:
        # Parse existing HIP_VISIBLE_DEVICES to get original system GPU indices
        original_system_indices = [
            int(idx.strip()) for idx in current_hip_visible.split(",")
        ]
        print(f"HIP_VISIBLE_DEVICES already set to: {current_hip_visible}")
    else:
        # HIP_VISIBLE_DEVICES not set, no remapping needed
        original_system_indices = None

    # Query available GPUs using subprocess (doesn't initialize CUDA in main process)
    # TODO: Combine those 2 functions to only import torch once to make it faster
    print("Getting GPUs supported by the current PyTorch build...", end="")
    supported_gpus = get_supported_gpus()
    print("done")
    print("Getting visible GPUs...", end="")
    raw_visible_gpus = get_visible_gpus()
    print("done")

    # Normalize gpu names
    # get_visible_gpus() (via device_properties.gcnArchName):
    # Often returns detailed arch names like "gfx942:sramecc+:xnack-" or "gfx1100:xnack-"
    visible_gpus = [gpu.split(":")[0] for gpu in raw_visible_gpus]

    print(f"Supported AMD GPUs: {supported_gpus}")
    print(f"Visible AMD GPUs: {visible_gpus}")

    selected_gpu_indices = []
    selected_gpu_archs = []

    if not amdgpu_family:
        # Mode 1: Auto-detect - use first supported GPU
        for idx, gpu in enumerate(visible_gpus):
            if gpu in supported_gpus:
                selected_gpu_indices = [idx]
                selected_gpu_archs = [gpu]
                break
        if len(selected_gpu_archs) == 0:
            print(f"[ERROR] No GPU found in visible GPUs that is supported by PyTorch")
            sys.exit(1)
        print(
            f"AMDGPU Arch auto-detected (using GPU at logical index {selected_gpu_indices[0]}): {selected_gpu_archs[0]}"
        )
    elif amdgpu_family.split("-")[0].upper().endswith("X"):
        # Mode 2: Wildcard match (e.g., "gfx94X" matches "gfx942", "gfx940", etc.)
        family_part = amdgpu_family.split("-")[0]
        partial_match = family_part[:-1]  # Remove the 'X'

        for idx, gpu in enumerate(visible_gpus):
            if partial_match in gpu and gpu in supported_gpus:
                selected_gpu_indices += [idx]
                selected_gpu_archs += [gpu]

        if len(selected_gpu_archs) == 0:
            print(f"[ERROR] No GPU found matching wildcard pattern '{amdgpu_family}'.")
            sys.exit(1)

        print(
            f"AMDGPU Arch detected via wildcard match '{partial_match}': "
            f"{selected_gpu_archs} (logical indices {selected_gpu_indices})"
        )
    else:
        # Mode 3: Specific GPU arch - validate it is visible and supported by the current PyTorch build.
        for idx, gpu in enumerate(visible_gpus):
            if gpu in supported_gpus:
                if gpu == amdgpu_family or amdgpu_family in gpu:
                    selected_gpu_indices += [idx]
                    selected_gpu_archs += [gpu]

        if len(selected_gpu_archs) == 0:
            print(
                f"[ERROR] Requested GPU '{amdgpu_family}' not found in visible GPUs that are supported by PyTorch"
            )
            sys.exit(1)

        print(
            f"AMDGPU Arch validated: {selected_gpu_archs} (logical indices {selected_gpu_indices})"
        )

    # Set HIP_VISIBLE_DEVICES to select the specific GPU(s)
    # This MUST be done before torch is imported in the main process via pytest.

    # Map logical indices back to system indices if HIP_VISIBLE_DEVICES was already set
    if original_system_indices is not None:
        # Map: logical index -> original system index
        # e.g., if HIP_VISIBLE_DEVICES="2,3,4" and we selected logical index 0,
        # we need to set HIP_VISIBLE_DEVICES="2" (the original system index)
        system_gpu_indices = [
            original_system_indices[idx] for idx in selected_gpu_indices
        ]
    else:
        # HIP_VISIBLE_DEVICES not set, no remapping needed
        system_gpu_indices = selected_gpu_indices

    str_indices = ",".join(str(idx) for idx in system_gpu_indices)
    os.environ["HIP_VISIBLE_DEVICES"] = str_indices
    print(f"Set HIP_VISIBLE_DEVICES={str_indices}")

    return selected_gpu_archs


def detect_pytorch_version() -> str:
    """Auto-detect the PyTorch version from the installed package.

    Returns:
        The detected PyTorch version as major.minor (e.g., "2.7").
    """
    # Get version, remove build suffix (+rocm, +cpu, etc.) and patch version
    return version("torch").rsplit("+", 1)[0].rsplit(".", 1)[0]


def main() -> int:
    """Main entry point for the PyTorch test runner.

    Returns:
        Exit code from pytest (0 for success, non-zero for failures).
    """
    args = cmd_arguments(sys.argv[1:])

    pytorch_dir = args.pytorch_dir

    # CRITICAL: Determine AMDGPU family and set HIP_VISIBLE_DEVICES
    # BEFORE importing torch/running pytest. Once torch.cuda is initialized,
    # changing HIP_VISIBLE_DEVICES has no effect.
    amdgpu_family = detect_amdgpu_family(args.amdgpu_family)
    print(f"Using AMDGPU family: {amdgpu_family}")

    # Determine PyTorch version
    pytorch_version = args.pytorch_version
    if not pytorch_version:
        pytorch_version = detect_pytorch_version()
    print(f"Using PyTorch version: {pytorch_version}")

    # Get tests to skip
    tests_to_skip = get_tests(
        amdgpu_family=amdgpu_family,
        pytorch_version=pytorch_version,
        platform=platform.system(),
        create_skip_list=not args.debug,
    )

    # Allow manual override of test selection
    if args.k:
        tests_to_skip = args.k

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


def force_exit_with_code(retcode):
    """Forces termination to work around https://github.com/ROCm/TheRock/issues/999."""
    import signal

    retcode_file = Path("exit_code.txt")
    print(f"Writing retcode {retcode} to '{retcode_file}'")
    with open(retcode_file, "w") as f:
        f.write(str(retcode))

    print("Forcefully terminating to avoid https://github.com/ROCm/TheRock/issues/999")

    # Flush output before we force exit so no logs get missed.
    sys.stdout.flush()

    # In order from "asking nicely" to "tear down immediately":
    #   1. `sys.exit(retcode)`
    #   2. `os._exit(retcode)`
    #   3. `os.kill(os.getpid(), signal.SIGTERM)`
    #   4. `subprocess.Popen(f'taskkill /F /PID {os.getpid()}', shell=True)`
    # As options (1) and (2) are not sufficient, we use option (3) here.
    os.kill(os.getpid(), signal.SIGTERM)


if __name__ == "__main__":
    retcode = main()
    if platform.system() == "Windows":
        force_exit_with_code(retcode)
    else:
        sys.exit(retcode)
