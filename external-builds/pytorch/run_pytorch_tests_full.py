#!/usr/bin/env python3
"""Runs the full PyTorch test suite on AMD GPUs via PyTorch's run_test.py,
with TheRock ROCm-specific skip-test integration and sharding support.

For the "default" and "distributed" configs, mirrors how PyTorch CI's
test.sh invokes test_python_shard():
    python test/run_test.py \\
        --exclude-jit-executor --exclude-distributed-tests \\
        --exclude-quantization-tests --shard N M --verbose

For the "inductor" config, mirrors test_inductor_shard() from test.sh
with two separate run_test.py invocations:
    1. Generic tests (test_modules, test_ops, …) with ``--inductor``
    2. Inductor unit tests without ``--inductor`` (avoids nested dynamo)

Usage examples:

    # Run all tests (no sharding):
    python run_pytorch_tests_full.py

    # Run shard 2 of 4 with the "default" config:
    python run_pytorch_tests_full.py --shard 2 --num-shards 4

    # Run only the test_nn test file:
    python run_pytorch_tests_full.py --include test_nn

    # Run a few specific test files:
    python run_pytorch_tests_full.py --include test_nn test_torch test_cuda

    # Run with the "inductor" config:
    python run_pytorch_tests_full.py --test-config inductor --shard 1 --num-shards 2

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
    check_pytorch_source_version,
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
# GPU types.  Since pytorch/pytorch#176445, each GPU gets its own key with
# inductor timings included, so a single map suffices for all configs.
AMDGPU_FAMILY_TO_BUILD_ENV = {
    "gfx90X-dcgpu": "linux-jammy-rocm-py3.10-mi200",
    "gfx94X-dcgpu": "linux-noble-rocm-py3.12-mi300",
    "gfx950-dcgpu": "linux-noble-rocm-py3.12-mi355",
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


PYTEST_TIMEOUT_SECONDS = 900  # 15 minutes per test function

# Test modules excluded at the run_test.py level (--exclude).  These are
# modules that hang or crash the subprocess in ways that pytest-timeout
# cannot catch (e.g. hanging during import or in C extensions).
# TODO: investigate the root cause and narrow the exclusions.
EXCLUDED_TEST_MODULES: list[str] = [
    "nn/test_convolution",  # hangs for 5+ hours, see run 53 shards 7 & 10
    "inductor/test_max_autotune",
    "inductor/test_torchinductor_opinfo_properties",
    "inductor/test_compiled_autograd",
    "distributed/_composable/fsdp/test_fully_shard_autograd",
    "distributed/_composable/test_composability/test_2d_composability",
    "distributed/_composable/test_composability/test_pp_composability",
    "distributed/_composable/test_replicate",
    "distributed/tensor/test_view_ops",
    "dynamo/test_dynamic_shapes",
    "functorch/test_control_flow",
]

# Inductor config: mirrors upstream test_inductor_shard() in .ci/pytorch/test.sh.
# The inductor config requires TWO separate run_test.py invocations:
#   1. Generic tests run with --inductor (sets PYTORCH_TEST_WITH_INDUCTOR=1)
#   2. Inductor unit tests run WITHOUT --inductor (avoids nested dynamo state)
# See: https://github.com/pytorch/pytorch/blob/main/.ci/pytorch/test.sh
INDUCTOR_GENERIC_TESTS = [
    "test_modules",
    "test_ops",
    "test_ops_gradients",
    "test_torch",
]
INDUCTOR_UNIT_TESTS = [
    "inductor/test_torchinductor",
    "inductor/test_torchinductor_opinfo",
    "inductor/test_aot_inductor",
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
    parser.add_argument(
        "--allow-version-mismatch",
        default=False,
        required=False,
        action=argparse.BooleanOptionalAction,
        help="""Allows version mismatches between pytorch test sources and installed packages. Defaults to False, so mismatched versions block running tests""",
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
    if args.test_config == "distributed":
        cmd.append("--distributed-tests")
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
    test_dir = args.pytorch_dir / "test"
    excludes = [m for m in EXCLUDED_TEST_MODULES if (test_dir / (m + ".py")).exists()]
    if args.exclude:
        excludes.extend(args.exclude)
    if excludes:
        cmd.extend(["--exclude"] + excludes)

    if tests_to_skip:
        cmd.extend(["-k", tests_to_skip])

    if not args.cache:
        passthrough_args.append("-p")
        passthrough_args.append("no:cacheprovider")

    passthrough_args.extend(["--timeout", str(PYTEST_TIMEOUT_SECONDS)])

    cmd.extend(passthrough_args)
    return cmd


def build_inductor_cmds(
    args: argparse.Namespace,
    tests_to_skip: str,
    passthrough_args: list[str],
) -> list[list[str]]:
    """Build the two run_test.py commands for the inductor config.

    Matches upstream ``test_inductor_shard()`` in ``.ci/pytorch/test.sh``:
      1. Generic tests (test_modules, test_ops, …) with ``--inductor``
      2. Inductor unit tests (inductor/test_torchinductor, …) *without*
         ``--inductor`` to avoid nested dynamo state
    """
    run_test_path = str(args.pytorch_dir / "test" / "run_test.py")

    extra = list(passthrough_args)
    if not args.cache:
        extra.extend(["-p", "no:cacheprovider"])
    extra.extend(["--timeout", str(PYTEST_TIMEOUT_SECONDS)])

    skip_args = ["-k", tests_to_skip] if tests_to_skip else []

    def _base_cmd() -> list[str]:
        cmd = [sys.executable, run_test_path]
        cmd.extend(["--keep-going", "--verbose"])
        if args.dry_run:
            cmd.append("--dry-run")
        if args.shard > 0 and args.num_shards > 0:
            cmd.extend(["--shard", str(args.shard), str(args.num_shards)])
        return cmd

    # 1. Generic tests WITH --inductor (enables TorchInductor backend)
    cmd1 = _base_cmd()
    cmd1.append("--inductor")
    cmd1.extend(["--include"] + INDUCTOR_GENERIC_TESTS)
    cmd1.extend(skip_args)
    cmd1.extend(extra)

    # 2. Inductor unit tests WITHOUT --inductor (nested dynamo guard)
    cmd2 = _base_cmd()
    cmd2.extend(["--include"] + INDUCTOR_UNIT_TESTS)
    cmd2.extend(skip_args)
    cmd2.extend(extra)

    return [cmd1, cmd2]


def _run_inductor(
    args: argparse.Namespace,
    tests_to_skip: str,
    passthrough_args: list[str],
) -> int:
    """Run the inductor test config as two run_test.py invocations.

    Matches upstream ``test_inductor_shard()`` in ``.ci/pytorch/test.sh``.
    Returns the worst (non-zero) return code from either invocation.
    """
    # Upstream runs verify_dynamo.py first as a quick smoke test.
    verify_script = args.pytorch_dir / "tools" / "dynamo" / "verify_dynamo.py"
    if verify_script.exists():
        print("Running verify_dynamo.py …")
        vr = subprocess.run(
            [sys.executable, str(verify_script)], cwd=str(args.pytorch_dir)
        )
        if vr.returncode != 0:
            print(f"verify_dynamo.py failed with return code {vr.returncode}")
            return vr.returncode
    else:
        print(f"verify_dynamo.py not found at {verify_script}, skipping")

    cmds = build_inductor_cmds(args, tests_to_skip, passthrough_args)
    labels = [
        "generic tests with --inductor",
        "inductor unit tests (no --inductor)",
    ]

    worst_rc = 0
    for label, cmd in zip(labels, cmds):
        print(f"\n{'=' * 60}")
        print(f"Inductor phase: {label}")
        print(f"{'=' * 60}")
        print(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(args.pytorch_dir))
        print(
            f"run_test.py [{label}] finished with return code: {result.returncode}",
            flush=True,
        )
        if result.returncode != 0:
            worst_rc = result.returncode

    return worst_rc


def main(argv: list[str]) -> int:
    args, passthrough_args = cmd_arguments(argv)
    check_pytorch_source_version(
        pytorch_dir=args.pytorch_dir, allow_mismatch=args.allow_version_mismatch
    )

    # Determine AMDGPU family and set HIP_VISIBLE_DEVICES BEFORE importing
    # torch or running pytest.  Once torch.cuda is initialized, changing
    # HIP_VISIBLE_DEVICES has no effect.  Distributed tests need all GPUs;
    # other configs use a single device to avoid multi-GPU contention.
    gpu_policy = "all" if args.test_config == "distributed" else "single"
    selected = set_gpu_execution_policy(args.amdgpu_family, policy=gpu_policy)
    first_arch = selected[0][0]
    unique_archs = sorted(set(arch for arch, _ in selected))
    device_ids = [str(dev_id) for _, dev_id in selected]
    print(
        f"Selected {len(selected)} GPU(s): "
        f"arch(es)={', '.join(unique_archs)}, "
        f"device(s)={', '.join(device_ids)}"
    )

    # get_tests amdgpu_family requires list[str]
    first_arch = [first_arch]

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

    if args.test_config == "inductor":
        return_code = _run_inductor(args, tests_to_skip, passthrough_args)
    else:
        cmd = build_run_test_cmd(args, tests_to_skip, passthrough_args)
        print(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(args.pytorch_dir))
        return_code = result.returncode
        print(f"run_test.py finished with return code: {return_code}")

    # Force-exit immediately.  PyTorch's run_test.py is known to hang after
    # all test files complete due to leaked daemon threads or orphan child
    # processes (https://github.com/ROCm/TheRock/issues/999).  os._exit()
    # terminates without waiting for threads or running atexit handlers.
    os._exit(return_code if return_code >= 0 else 1)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
