#!/usr/bin/env python3
"""Run PyTorch's CI test.sh with TheRock's ROCm test configuration."""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

from pytorch_utils import (
    check_pytorch_source_version,
    configure_gpu_visibility,
    detect_pytorch_version,
    reconcile_agent_visibility_env,
)
from skip_tests.create_skip_tests import get_tests


THIS_SCRIPT_DIR = Path(__file__).resolve().parent
PYTEST_TIMEOUT_SECONDS = 900

# Match generated-stats keys used by PyTorch's test sharding.
AMDGPU_FAMILY_TO_BUILD_ENV = {
    "gfx90X-dcgpu": "linux-jammy-rocm-py3.10-mi200",
    "gfx94X-dcgpu": "linux-noble-rocm-py3.12-mi300",
    "gfx950-dcgpu": "linux-noble-rocm-py3.12-mi355",
    "gfx110X-all": "linux-jammy-rocm-py3.10-navi31",
}
ROCM_BUILD_ENVIRONMENT_DEFAULT = "linux-noble-rocm-py3.12-mi300"

# Exclude modules that can hang or crash before pytest-timeout intervenes.
EXCLUDED_TEST_MODULES = [
    "nn/test_convolution",
    "inductor/test_max_autotune",
    "inductor/test_torchinductor_opinfo_properties",
    "inductor/test_compiled_autograd",
    "dynamo/test_dynamic_shapes",
    "functorch/test_control_flow",
]


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    try:
        separator = argv.index("--")
    except ValueError:
        pytest_args: list[str] = []
    else:
        pytest_args = argv[separator + 1 :]
        argv = argv[:separator]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amdgpu-family", default=os.getenv("AMDGPU_FAMILY", ""))
    parser.add_argument("--pytorch-version", default=os.getenv("PYTORCH_VERSION", ""))
    parser.add_argument("--pytorch-dir", type=Path, default=THIS_SCRIPT_DIR / "pytorch")
    parser.add_argument("--test-config", default=os.getenv("TEST_CONFIG", "default"))
    parser.add_argument(
        "--shard", type=int, default=int(os.getenv("SHARD_NUMBER", "0"))
    )
    parser.add_argument(
        "--num-shards", type=int, default=int(os.getenv("NUM_TEST_SHARDS", "0"))
    )
    parser.add_argument("--include", nargs="+")
    parser.add_argument("--exclude", nargs="+")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-k", default="")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--device-query", choices=["auto", "unique", "all"], default="auto"
    )
    parser.add_argument(
        "--gpu-policy", choices=["auto", "single", "all"], default="auto"
    )
    parser.add_argument(
        "--allow-version-mismatch",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args(argv)

    test_sh = args.pytorch_dir / ".ci" / "pytorch" / "test.sh"
    if not test_sh.is_file():
        parser.error(f"PyTorch test.sh not found at '{test_sh}'.")
    if (args.shard > 0) != (args.num_shards > 0):
        parser.error("--shard and --num-shards must both be set or both be unset.")
    if args.shard > args.num_shards:
        parser.error("--shard cannot exceed --num-shards.")

    is_distributed = args.test_config == "distributed"
    if args.device_query == "auto":
        args.device_query = "all" if is_distributed else "unique"
    if args.gpu_policy == "auto":
        args.gpu_policy = "all" if is_distributed else "single"
    return args, pytest_args


def configure_environment(
    args: argparse.Namespace,
    pytest_args: list[str],
    tests_to_skip: str,
) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CI", "1")
    env.setdefault(
        "BUILD_ENVIRONMENT",
        AMDGPU_FAMILY_TO_BUILD_ENV.get(
            args.amdgpu_family, ROCM_BUILD_ENVIRONMENT_DEFAULT
        ),
    )
    env.setdefault("PYTORCH_TEST_WITH_ROCM", "1")
    env.setdefault("PYTORCH_TESTING_DEVICE_ONLY_FOR", "cuda")
    env.setdefault("PYTORCH_PRINT_REPRO_ON_FAILURE", "0")
    env["MIOPEN_CUSTOM_CACHE_DIR"] = tempfile.mkdtemp()
    env["TEST_CONFIG"] = args.test_config

    if args.test_config != "distributed":
        env["PYTORCH_TEST_RUN_EVERYTHING_IN_SERIAL"] = "1"
    if args.shard:
        env["SHARD_NUMBER"] = str(args.shard)
        env["NUM_TEST_SHARDS"] = str(args.num_shards)
    if args.include:
        env["TESTS_TO_INCLUDE"] = " ".join(args.include)

    test_dir = args.pytorch_dir / "test"
    excluded = [
        name for name in EXCLUDED_TEST_MODULES if (test_dir / f"{name}.py").is_file()
    ]
    excluded.extend(args.exclude or [])
    if excluded:
        env["TESTS_TO_EXCLUDE"] = " ".join(excluded)

    addopts = shlex.split(env.get("PYTEST_ADDOPTS", ""))
    addopts.extend(pytest_args)
    if tests_to_skip:
        addopts.extend(["-k", tests_to_skip])
    if not args.cache:
        addopts.extend(["-p", "no:cacheprovider"])
    addopts.extend(["--timeout", str(PYTEST_TIMEOUT_SECONDS)])
    env["PYTEST_ADDOPTS"] = shlex.join(addopts)

    libpython_dir = sysconfig.get_config_var("LIBDIR") or str(Path(sys.prefix) / "lib")
    if platform.system() != "Windows" and Path(libpython_dir).is_dir():
        old_ld_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{libpython_dir}:{old_ld_path}" if old_ld_path else libpython_dir
        )
    return env


def main(argv: list[str]) -> int:
    args, pytest_args = parse_args(argv)
    check_pytorch_source_version(
        pytorch_dir=args.pytorch_dir,
        allow_mismatch=args.allow_version_mismatch,
    )

    reconcile_agent_visibility_env()
    selected_archs = configure_gpu_visibility(
        args.amdgpu_family, args.device_query, args.gpu_policy
    )
    pytorch_version = args.pytorch_version or detect_pytorch_version()
    tests_to_skip = args.k or get_tests(
        amdgpu_family=selected_archs,
        pytorch_version=pytorch_version,
        platform=platform.system(),
        create_skip_list=not args.debug,
    )
    env = configure_environment(args, pytest_args, tests_to_skip)

    test_sh = args.pytorch_dir / ".ci" / "pytorch" / "test.sh"
    print(f"Using PyTorch version: {pytorch_version}")
    print(f"Executing: bash {test_sh}", flush=True)
    result = subprocess.run(["bash", str(test_sh)], cwd=args.pytorch_dir, env=env)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
