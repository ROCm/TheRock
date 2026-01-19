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

Pass additional pytest arguments after "--":
    $ python run_pytorch_tests.py -- -m "slow"
    $ python run_pytorch_tests.py -- --tb=short -x

Exit Codes
----------
0 : All tests passed
1 : Test failures or collection errors
? : Other exit codes from pytest
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
writes 'run_pytorch_tests_exit_code.txt' to the current directory and then kills the process.
"""

import argparse
import os
import platform
import sys
import tempfile

from skip_tests.create_skip_tests import *
from pathlib import Path

import pytest

from pytorch_utils import (
    set_gpu_execution_policy,
    detect_pytorch_version,
)

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


def cmd_arguments(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments.

    Args:
        argv: Command line arguments (without program name).

    Returns:
        Tuple of (parsed args, passthrough pytest args passed after "--").
    """
    # Extract passthrough pytest args after "--"
    try:
        rest_pos = argv.index("--")
    except ValueError:
        passthrough_pytest_args = []
    else:
        passthrough_pytest_args = argv[rest_pos + 1 :]
        argv = argv[:rest_pos]

    parser = argparse.ArgumentParser(
        description="""
Runs PyTorch pytest for AMD GPUs. Skips additional tests compared to upstream.
Additional tests to be skipped can be tuned by PyTorch version and amdgpu family.
All arguments after "--" are passed directly to pytest.
"""
    )

    parser.add_argument(
        "--amdgpu-family",
        type=str,
        default=os.getenv("AMDGPU_FAMILY", ""),
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
        help="""Path for the pytorch repository, where tests will be sourced from
By default the pytorch directory is determined based on this script's location
""",
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
            f"Directory at '{args.pytorch_dir}' does not exist, checkout pytorch and then set the path via --pytorch-dir or check it out in TheRock/external-build/pytorch/<your pytorch directory>"
        )

    return args, passthrough_pytest_args


def main() -> int:
    """Main entry point for the PyTorch test runner.

    Returns:
        Exit code from pytest (0 for success, non-zero for failures).
    """
    try:
        args, passthrough_pytest_args = cmd_arguments(sys.argv[1:])

        pytorch_dir = args.pytorch_dir

        # CRITICAL: Determine AMDGPU family and set HIP_VISIBLE_DEVICES
        # BEFORE importing torch/running pytest. Once torch.cuda is initialized,
        # changing HIP_VISIBLE_DEVICES has no effect.
        # For unit tests, run only on the first supported device (policy="single")
        ((first_arch, _),) = set_gpu_execution_policy(
            args.amdgpu_family, policy="single"
        )
        print(f"Using AMDGPU family: {first_arch}")

        # Determine PyTorch version
        pytorch_version = args.pytorch_version
        if not pytorch_version:
            pytorch_version = detect_pytorch_version()
        print(f"Using PyTorch version: {pytorch_version}")

        # Get tests to skip
        tests_to_skip = get_tests(
            amdgpu_family=first_arch,
            pytorch_version=pytorch_version,
            platform=platform.system(),
            create_skip_list=not args.debug,
        )

        # Allow manual override of test selection
        if args.k:
            tests_to_skip = args.k

        setup_env(pytorch_dir)

        pytest_args = [
            f"{pytorch_dir}/test/test_nn.py",
            f"{pytorch_dir}/test/test_torch.py",
            f"{pytorch_dir}/test/test_cuda.py",
            f"{pytorch_dir}/test/test_unary_ufuncs.py",
            f"{pytorch_dir}/test/test_binary_ufuncs.py",
            f"{pytorch_dir}/test/test_autograd.py",
            f"-k={tests_to_skip}",
            # "-n 0",  # TODO does this need rework? why should we not run this multithreaded? this does not seem to exist?
            # -n numprocesses, --numprocesses=numprocesses
            #         Shortcut for '--dist=load --tx=NUM*popen'.
            #         With 'logical', attempt to detect logical CPU count (requires psutil, falls back to 'auto').
            #         With 'auto', attempt to detect physical CPU count. If physical CPU count cannot be determined, falls back to 1.
            #         Forced to 0 (disabled) when used with --pdb.
        ]

        if args.no_cache:
            pytest_args += [
                "-p",
                "no:cacheprovider",  # Disable caching: useful when running in a container
            ]
        # Append any passthrough pytest args passed after "--"
        pytest_args.extend(passthrough_pytest_args)

        retcode = pytest.main(pytest_args)
        print(f"Pytest finished with return code: {retcode}")
        return retcode
    except (ValueError, IndexError) as e:
        print(f"[ERROR] Exception in PyTorch unit-tests runner: {e}")
        sys.exit(1)


def force_exit_with_code(retcode):
    """Forces termination to work around https://github.com/ROCm/TheRock/issues/999."""
    import signal

    # We're going to kill the current process with SIGTERM below, which will
    # return exit code 15. This preserves the original exit code in a file.
    # Note: this path is relative to CWD, *not the script directory*.
    # TODO(#2258): output a test report file that can be inspected on both
    #              Linux and Windows then remove this special file
    retcode_file = Path("run_pytorch_tests_exit_code.txt")
    # Convert to int in case retcode is a pytest.ExitCode enum
    retcode_int = int(retcode)
    print(f"Writing retcode {retcode_int} to '{retcode_file}'")
    with open(retcode_file, "w") as f:
        f.write(str(retcode_int))

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
