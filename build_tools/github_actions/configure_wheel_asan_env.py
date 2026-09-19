#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exports the environment that ASAN-instrumented wheel test jobs need.

`rocm-sdk test` runs `rocm_sdk.initialize_process()`, which loads instrumented
ROCm libraries into the running interpreter with `ctypes.CDLL`. CPython is not
built with `-fsanitize=address`, so unless the ASAN runtime is already first in
the library list the load aborts with "ASan runtime does not come first in
initial library list" (#6331). Library order is fixed when the process is
exec'd, so the only fix available to the test job is to preload the runtime
before `rocm-sdk test` starts.

Unlike `configure_asan_env.py`, there is no artifact tree here: these jobs
install ROCm from wheels, so the runtime is discovered inside the installed
`rocm-sdk` core package instead of by asking a build tree's clang.

Missing binaries warn rather than fail, matching `configure_asan_env.py`.

Used by `test_rocm_wheels.yml`.
"""

import argparse
import importlib
import os
import platform
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configure_asan_env import STATIC_ASAN_ENV
from github_actions_api import gha_set_env

# `-fsanitize=address` emits libclang_rt.asan.so under a per-target runtime
# directory, and libclang_rt.asan-<arch>.so under the layout #8077 reintroduced.
# Both spellings ship in wheels depending on how the compiler was configured, so
# try the exact names first and fall back to a glob.
ASAN_RUNTIME_NAMES = (
    "libclang_rt.asan.so",
    f"libclang_rt.asan-{platform.machine()}.so",
)
ASAN_RUNTIME_GLOB = "libclang_rt.asan*.so"


def resolve_core_package_root() -> tuple[Optional[Path], Optional[str]]:
    """Locates the installed rocm-sdk core package.

    Returns (path, warning); exactly one is set.
    """
    try:
        from rocm_sdk import _dist_info as di
    except ModuleNotFoundError as e:
        return None, f"rocm_sdk is not importable, ASAN runtime not resolved: {e}"

    core_package_name = di.ALL_PACKAGES["core"].get_py_package_name()
    try:
        core_module = importlib.import_module(core_package_name)
    except ModuleNotFoundError as e:
        return (
            None,
            f"{core_package_name} is not installed, ASAN runtime not resolved: {e}",
        )

    if not core_module.__file__:
        return (
            None,
            f"{core_package_name} is a namespace package, ASAN runtime not resolved",
        )

    return Path(core_module.__file__).parent, None


def find_asan_runtime(root: Path) -> tuple[Optional[Path], Optional[str]]:
    """Searches an installed package tree for the shared ASAN runtime.

    Returns (path, warning); exactly one is set.
    """
    by_name: dict[str, Path] = {}
    for candidate in root.rglob(ASAN_RUNTIME_GLOB):
        if candidate.is_file():
            by_name.setdefault(candidate.name, candidate)

    if not by_name:
        return None, (
            f"no {ASAN_RUNTIME_GLOB} under {root}; the wheels under test are "
            "probably not ASAN-instrumented"
        )

    for name in ASAN_RUNTIME_NAMES:
        if name in by_name:
            return by_name[name].resolve(), None

    # An unexpected spelling still beats aborting the job over the name.
    fallback = by_name[sorted(by_name)[0]]
    return fallback.resolve(), None


def find_symbolizer(root: Path) -> Optional[Path]:
    """Locates llvm-symbolizer in an installed package tree, if it ships one."""
    for candidate in root.rglob("llvm-symbolizer*"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return None


def resolve_wheel_asan_env(root: Path) -> tuple[dict[str, str], list[str]]:
    """Returns (environment variables to export, warnings to surface)."""
    env = dict(STATIC_ASAN_ENV)
    warnings: list[str] = []

    runtime, warning = find_asan_runtime(root)
    if runtime:
        env["ASAN_RUNTIME_PATH"] = str(runtime)
    if warning:
        warnings.append(warning)

    symbolizer = find_symbolizer(root)
    if symbolizer:
        env["ASAN_SYMBOLIZER_PATH"] = str(symbolizer)
    else:
        warnings.append(
            f"llvm-symbolizer not found under {root}, ASAN reports will be unsymbolized"
        )

    return env, warnings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-root",
        type=Path,
        help=(
            "Installed package tree to search. Defaults to the rocm-sdk core "
            "package in the active environment."
        ),
    )
    args = parser.parse_args(argv)

    root = args.package_root
    if root is None:
        root, warning = resolve_core_package_root()
        if warning:
            print(f"::warning::{warning}")
            return 0

    env, warnings = resolve_wheel_asan_env(root)

    for warning in warnings:
        print(f"::warning::{warning}")

    for key, value in env.items():
        print(f"{key}={value}")
        gha_set_env({key: value})

    return 0


if __name__ == "__main__":
    sys.exit(main())
