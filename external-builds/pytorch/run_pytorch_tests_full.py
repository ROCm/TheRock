#!/usr/bin/env python3
"""Runs the full PyTorch test suite on AMD GPUs via PyTorch's run_test.py,
with TheRock ROCm-specific skip-test integration and sharding support.

Mirrors how PyTorch CI's test.sh invokes test_python_shard():
    python test/run_test.py \\
        --exclude-jit-executor --exclude-distributed-tests \\
        --exclude-quantization-tests --shard N M --verbose

Usage examples:

    # Run all tests (no sharding):
    python run_pytorch_tests_full.py

    # Run shard 2 of 4 with the "default" config:
    python run_pytorch_tests_full.py --shard 2 --num-shards 4

    # Run only the test_nn test file:
    python run_pytorch_tests_full.py --include test_nn

    # Run a few specific test files:
    python run_pytorch_tests_full.py --include test_nn test_torch test_cuda

    # Run with the "distributed" config on a multi-GPU runner:
    python run_pytorch_tests_full.py --test-config distributed

    # Pass extra pytest arguments after "--":
    python run_pytorch_tests_full.py -- --continue-on-collection-errors

    # Dry run to list tests without executing them:
    python run_pytorch_tests_full.py --dry-run

    # Disable pytest caching (useful with read-only pytorch directory):
    python run_pytorch_tests_full.py --no-cache

Environment variables (all overridable via CLI flags or workflow YAML):
    AMDGPU_FAMILY, TEST_CONFIG, SHARD_NUMBER, NUM_TEST_SHARDS,
    TESTS_TO_INCLUDE, PYTORCH_VERSION
"""

import argparse
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

from skip_tests.create_skip_tests import get_tests

from pytorch_utils import (
    detect_pytorch_version,
    set_gpu_execution_policy,
)

THIS_SCRIPT_DIR = Path(__file__).resolve().parent

# Maps AMDGPU_FAMILY to BUILD_ENVIRONMENT, which run_test.py uses to look up
# historical test durations in test-times.json for balanced shard splitting.
# See: https://raw.githubusercontent.com/pytorch/test-infra/generated-stats/stats/test-times.json
#
# The values must match keys present in that JSON file exactly, which is why
# they include linux-specific OS and Python versions (e.g. "linux-noble-rocm-
# py3.12-mi300").  This list is intentionally non-exhaustive: GPU families not
# listed here (e.g. gfx1151, Windows targets) fall back to
# ROCM_BUILD_ENVIRONMENT_DEFAULT.  Falling back to mi300 timings still gives
# reasonably balanced shards since relative test durations are similar across
# GPU types.  Once pytorch/pytorch#176445 lands, we can match on the GPU SKU
# suffix for a more robust lookup.
AMDGPU_FAMILY_TO_BUILD_ENV = {
    "gfx90X-dcgpu": "linux-jammy-rocm-py3.10-mi200",
    "gfx94X-dcgpu": "linux-noble-rocm-py3.12-mi300",
    "gfx950-dcgpu": "linux-jammy-rocm-py3.10-mi355",
    "gfx110X-all": "linux-jammy-rocm-py3.10-navi31",
}
ROCM_BUILD_ENVIRONMENT_DEFAULT = "linux-noble-rocm-py3.12-mi300"

THEROCK_ENV_VARS = [
    "CI",
    "BUILD_ENVIRONMENT",
    "PYTORCH_TEST_WITH_ROCM",
    "PYTORCH_TESTING_DEVICE_ONLY_FOR",
    "PYTORCH_PRINT_REPRO_ON_FAILURE",
    "PYTORCH_TEST_RUN_EVERYTHING_IN_SERIAL",
    "MIOPEN_CUSTOM_CACHE_DIR",
    "TEST_CONFIG",
    "PYTHONPATH",
    "HIP_VISIBLE_DEVICES",
    "SHARD_NUMBER",
    "NUM_TEST_SHARDS",
    "TESTS_TO_INCLUDE",
]


def setup_env(pytorch_dir: Path, test_config: str, amdgpu_family: str = "") -> None:
    os.environ.setdefault("CI", "1")
    build_env = AMDGPU_FAMILY_TO_BUILD_ENV.get(
        amdgpu_family, ROCM_BUILD_ENVIRONMENT_DEFAULT
    )
    os.environ.setdefault("BUILD_ENVIRONMENT", build_env)
    os.environ.setdefault("PYTORCH_TEST_WITH_ROCM", "1")
    os.environ.setdefault("PYTORCH_TESTING_DEVICE_ONLY_FOR", "cuda")
    os.environ.setdefault("PYTORCH_PRINT_REPRO_ON_FAILURE", "0")
    os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = tempfile.mkdtemp()

    if test_config:
        os.environ.setdefault("TEST_CONFIG", test_config)

    # On 1-GPU runners, rocminfo reports all physical GPUs (e.g. 3) but only one
    # is visible via HIP_VISIBLE_DEVICES.  This causes NUM_PROCS=3 inside
    # run_test.py, spawning 3 parallel workers that all contend for the same GPU.
    # Force serial execution for non-distributed configs to avoid contention and
    # ensure even shard distribution by wall-clock time.
    if test_config != "distributed":
        os.environ["PYTORCH_TEST_RUN_EVERYTHING_IN_SERIAL"] = "1"

    # Add PyTorch test directory to PYTHONPATH so that run_test.py and pytest
    # can locate test helpers and internal modules.
    test_dir = str(pytorch_dir / "test")
    old_pythonpath = os.getenv("PYTHONPATH", "")
    if old_pythonpath:
        os.environ["PYTHONPATH"] = f"{test_dir}:{old_pythonpath}"
    else:
        os.environ["PYTHONPATH"] = test_dir

    # Force update the PYTHONPATH to be part of the sys path.
    # Otherwise our current python process that will run pytest will NOT
    # find it and pytest will crash!
    if test_dir not in sys.path:
        sys.path.insert(0, test_dir)


def print_env() -> None:
    title = " TheRock PyTorch Test Environment "
    bar = f"{'=' * len(title)}"
    print(bar)
    print(title)
    print(bar)
    for var in THEROCK_ENV_VARS:
        val = os.environ.get(var, "<not set>")
        print(f"  {var}={val}")
    print(bar)
    sys.stdout.flush()


def cmd_arguments(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    # Extract passthrough pytest args after "--"
    try:
        rest_pos = argv.index("--")
    except ValueError:
        passthrough_args = []
    else:
        passthrough_args = argv[rest_pos + 1 :]
        argv = argv[:rest_pos]

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--amdgpu-family",
        type=str,
        default=os.getenv("AMDGPU_FAMILY", ""),
        help='AMDGPU family string (e.g. "gfx94X-dcgpu", "gfx110X-all"). '
        "Falls back to AMDGPU_FAMILY env var, then auto-detection.",
    )
    parser.add_argument(
        "--pytorch-version",
        type=str,
        default=os.getenv("PYTORCH_VERSION", ""),
        help='PyTorch version for skip-list lookup (e.g. "2.9", "2.12"). '
        "Auto-detected from the installed torch package if not set.",
    )
    parser.add_argument(
        "--pytorch-dir",
        type=Path,
        default=THIS_SCRIPT_DIR / "pytorch",
        help="Path to the PyTorch source checkout (must contain test/run_test.py).",
    )
    parser.add_argument(
        "--test-config",
        type=str,
        default=os.getenv("TEST_CONFIG", "default"),
        help='TEST_CONFIG value for run_test.py sharding/config logic (default: "default").',
    )
    parser.add_argument(
        "--shard",
        type=int,
        default=int(os.getenv("SHARD_NUMBER", "0")),
        help="1-indexed shard number (e.g. --shard 2 --num-shards 4 runs shard 2 of 4). "
        "Also reads SHARD_NUMBER env var. Set to 0 to disable sharding.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=int(os.getenv("NUM_TEST_SHARDS", "0")),
        help="Total number of shards. Also reads NUM_TEST_SHARDS env var. "
        "Must be set together with --shard.",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=None,
        metavar="TEST",
        help="Only run these test files (e.g. --include test_nn test_torch). "
        "Passed to run_test.py --include. Also settable via TESTS_TO_INCLUDE "
        "env var, which run_test.py reads directly. If neither is set, "
        "run_test.py runs all tests for the given test config.",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=None,
        metavar="TEST",
        help="Exclude these test files (e.g. --exclude test_dynamo test_inductor). "
        "Passed to run_test.py --exclude.",
    )
    parser.add_argument(
        "--exclude-jit-executor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --exclude-jit-executor to run_test.py (default: enabled). "
        "Use --no-exclude-jit-executor to include JIT executor tests.",
    )
    parser.add_argument(
        "--exclude-distributed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --exclude-distributed-tests to run_test.py (default: enabled). "
        "Use --no-exclude-distributed to include distributed tests. "
        "Automatically disabled when --test-config=distributed.",
    )
    parser.add_argument(
        "--exclude-quantization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --exclude-quantization-tests to run_test.py (default: enabled). "
        "Use --no-exclude-quantization to include quantization tests.",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Invert TheRock skip list: only run tests that are normally skipped.",
    )
    parser.add_argument(
        "-k",
        default="",
        help="Override the pytest -k expression (bypasses TheRock skip-test generation).",
    )
    parser.add_argument(
        "--cache",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable pytest caching (default). Use --no-cache when only having "
        "read-only access to the pytorch directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Pass --dry-run to run_test.py to list tests without running them.",
    )
    args = parser.parse_args(argv)

    if not args.pytorch_dir.exists():
        parser.error(f"Directory at '{args.pytorch_dir}' does not exist.")

    run_test_path = args.pytorch_dir / "test" / "run_test.py"
    if not run_test_path.exists():
        parser.error(f"run_test.py not found at '{run_test_path}'.")

    if (args.shard > 0) != (args.num_shards > 0):
        parser.error("--shard and --num-shards must both be set or both be unset.")

    if args.shard > 0 and args.shard > args.num_shards:
        parser.error(
            f"--shard ({args.shard}) cannot exceed --num-shards ({args.num_shards})."
        )

    return args, passthrough_args


def build_run_test_cmd(
    args: argparse.Namespace,
    tests_to_skip: str,
    passthrough_args: list[str],
) -> list[str]:
    """Build the command line for PyTorch's test/run_test.py.

    Assembles flags for sharding, test selection, skip expressions, and any
    extra pytest arguments that were passed after ``--``.
    """
    run_test_path = str(args.pytorch_dir / "test" / "run_test.py")
    cmd = [sys.executable, run_test_path]

    if args.exclude_jit_executor:
        cmd.append("--exclude-jit-executor")
    if args.exclude_distributed and args.test_config != "distributed":
        cmd.append("--exclude-distributed-tests")
    if args.exclude_quantization:
        cmd.append("--exclude-quantization-tests")

    cmd.append("--keep-going")
    cmd.append("--verbose")

    if args.dry_run:
        cmd.append("--dry-run")

    if args.shard > 0 and args.num_shards > 0:
        cmd.extend(["--shard", str(args.shard), str(args.num_shards)])

    if args.include:
        cmd.extend(["--include"] + args.include)
    if args.exclude:
        cmd.extend(["--exclude"] + args.exclude)

    if tests_to_skip:
        cmd.extend(["-k", tests_to_skip])

    if not args.cache:
        passthrough_args.append("-p")
        passthrough_args.append("no:cacheprovider")

    cmd.extend(passthrough_args)
    return cmd


def main(argv: list[str]) -> int:
    args, passthrough_args = cmd_arguments(argv)

    # Determine AMDGPU family and set HIP_VISIBLE_DEVICES BEFORE importing
    # torch or running pytest.  Once torch.cuda is initialized, changing
    # HIP_VISIBLE_DEVICES has no effect.  For unit tests we run on a single
    # device (policy="single") to avoid multi-GPU contention.
    ((first_arch, _),) = set_gpu_execution_policy(args.amdgpu_family, policy="single")
    print(f"Using AMDGPU family: {first_arch}")

    pytorch_version = args.pytorch_version
    if not pytorch_version:
        pytorch_version = detect_pytorch_version()
    print(f"Using PyTorch version: {pytorch_version}")

    if args.k:
        tests_to_skip = args.k
    else:
        tests_to_skip = get_tests(
            amdgpu_family=first_arch,
            pytorch_version=pytorch_version,
            platform=platform.system(),
            create_skip_list=not args.debug,
        )

    setup_env(
        pytorch_dir=args.pytorch_dir,
        test_config=args.test_config,
        amdgpu_family=args.amdgpu_family,
    )
    print_env()

    cmd = build_run_test_cmd(args, tests_to_skip, passthrough_args)
    print(f"Executing: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=str(args.pytorch_dir))
    print(f"run_test.py finished with return code: {result.returncode}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
