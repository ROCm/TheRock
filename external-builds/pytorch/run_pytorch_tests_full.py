#!/usr/bin/env python3
"""Runs the full PyTorch test suite on AMD GPUs via PyTorch's run_test.py,
with TheRock ROCm-specific skip-test integration and sharding support.

Mirrors how PyTorch CI's test.sh invokes test_python_shard():
    python test/run_test.py \\
        --exclude-jit-executor --exclude-distributed-tests \\
        --exclude-quantization-tests --shard N M --verbose
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

#Use mi300 test times because https://github.com/ROCm/TheRock/actions/workflows/test_pytorch_wheels.yml chooses mi300 as default
ROCM_BUILD_ENVIRONMENT = "linux-noble-rocm-py3.12-mi300"

THEROCK_ENV_VARS = [
    "CI",
    "BUILD_ENVIRONMENT",
    "PYTORCH_TEST_WITH_ROCM",
    "PYTORCH_TESTING_DEVICE_ONLY_FOR",
    "PYTORCH_PRINT_REPRO_ON_FAILURE",
    "MIOPEN_CUSTOM_CACHE_DIR",
    "TEST_CONFIG",
    "PYTHONPATH",
    "HIP_VISIBLE_DEVICES",
    "SHARD_NUMBER",
    "NUM_TEST_SHARDS",
    "TESTS_TO_INCLUDE",
]


def setup_env(pytorch_dir: Path, test_config: str) -> None:
    os.environ["CI"] = "1"
    os.environ["BUILD_ENVIRONMENT"] = ROCM_BUILD_ENVIRONMENT
    os.environ["PYTORCH_TEST_WITH_ROCM"] = "1"
    os.environ["PYTORCH_TESTING_DEVICE_ONLY_FOR"] = "cuda"
    os.environ["PYTORCH_PRINT_REPRO_ON_FAILURE"] = "0"
    os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = tempfile.mkdtemp()

    if test_config:
        os.environ["TEST_CONFIG"] = test_config

    test_dir = str(pytorch_dir / "test")
    old_pythonpath = os.getenv("PYTHONPATH", "")
    if old_pythonpath:
        os.environ["PYTHONPATH"] = f"{test_dir}:{old_pythonpath}"
    else:
        os.environ["PYTHONPATH"] = test_dir


def print_env() -> None:
    print("=== TheRock PyTorch Test Environment ===")
    for var in THEROCK_ENV_VARS:
        val = os.environ.get(var, "<not set>")
        print(f"  {var}={val}")
    print("=========================================")
    sys.stdout.flush()


def cmd_arguments(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    try:
        rest_pos = argv.index("--")
    except ValueError:
        passthrough_args = []
    else:
        passthrough_args = argv[rest_pos + 1:]
        argv = argv[:rest_pos]

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--amdgpu-family",
        type=str,
        default=os.getenv("AMDGPU_FAMILY", ""),
        help='AMDGPU family (e.g. "gfx942"). Auto-detected if not set.',
    )
    parser.add_argument(
        "--pytorch-version",
        type=str,
        default=os.getenv("PYTORCH_VERSION", ""),
        help='PyTorch version (e.g. "2.7" or "all"). Auto-detected if not set.',
    )
    parser.add_argument(
        "--pytorch-dir",
        type=Path,
        default=THIS_SCRIPT_DIR / "pytorch",
        help="Path to the PyTorch repository root.",
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
        help="1-indexed shard number to run. Also reads SHARD_NUMBER env var.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=int(os.getenv("NUM_TEST_SHARDS", "0")),
        help="Total number of shards. Also reads NUM_TEST_SHARDS env var.",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=None,
        metavar="TEST",
        help="Only run these test files (passed to run_test.py --include). "
        "Also settable via TESTS_TO_INCLUDE env var (run_test.py reads it directly). "
        "Default: run all tests.",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=None,
        metavar="TEST",
        help="Exclude these test files (passed to run_test.py --exclude).",
    )
    parser.add_argument(
        "--no-exclude-jit-executor",
        action="store_true",
        default=False,
        help="Do NOT pass --exclude-jit-executor (excluded by default).",
    )
    parser.add_argument(
        "--no-exclude-distributed",
        action="store_true",
        default=False,
        help="Do NOT pass --exclude-distributed-tests (excluded by default).",
    )
    parser.add_argument(
        "--no-exclude-quantization",
        action="store_true",
        default=False,
        help="Do NOT pass --exclude-quantization-tests (excluded by default).",
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
        "--dry-run",
        action="store_true",
        default=False,
        help="Pass --dry-run to run_test.py to list tests without running them.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Pass --verbose to run_test.py (default: True).",
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
        parser.error(f"--shard ({args.shard}) cannot exceed --num-shards ({args.num_shards}).")

    return args, passthrough_args


def build_run_test_cmd(
    args: argparse.Namespace,
    tests_to_skip: str,
    passthrough_args: list[str],
) -> list[str]:
    run_test_path = str(args.pytorch_dir / "test" / "run_test.py")
    cmd = [sys.executable, run_test_path]

    if not args.no_exclude_jit_executor:
        cmd.append("--exclude-jit-executor")
    if not args.no_exclude_distributed:
        cmd.append("--exclude-distributed-tests")
    if not args.no_exclude_quantization:
        cmd.append("--exclude-quantization-tests")

    cmd.append("--keep-going")

    if args.verbose:
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

    cmd.extend(passthrough_args)
    return cmd


def main() -> int:
    try:
        args, passthrough_args = cmd_arguments(sys.argv[1:])

        ((first_arch, _),) = set_gpu_execution_policy(
            args.amdgpu_family, policy="single"
        )
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

        setup_env(args.pytorch_dir, args.test_config)
        print_env()

        cmd = build_run_test_cmd(args, tests_to_skip, passthrough_args)
        print(f"Executing: {' '.join(cmd)}")

        result = subprocess.run(cmd, cwd=str(args.pytorch_dir))
        print(f"run_test.py finished with return code: {result.returncode}")
        return result.returncode

    except (ValueError, IndexError) as e:
        print(f"[ERROR] Exception in PyTorch full test runner: {e}")
        return 1


def force_exit_with_code(retcode: int) -> None:
    import signal

    retcode_file = Path("run_pytorch_tests_full_exit_code.txt")
    retcode_int = int(retcode)
    print(f"Writing retcode {retcode_int} to '{retcode_file}'")
    with open(retcode_file, "w") as f:
        f.write(str(retcode_int))

    sys.stdout.flush()
    os.kill(os.getpid(), signal.SIGTERM)


if __name__ == "__main__":
    retcode = main()
    if platform.system() == "Windows":
        force_exit_with_code(retcode)
    else:
        sys.exit(retcode)
