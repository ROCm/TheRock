#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exports the environment that ASAN-instrumented test jobs need to GITHUB_ENV.

Test jobs for `asan` and `host-asan` artifact groups run binaries that link
against instrumented ROCm libraries. Two paths have to be discovered rather than
hardcoded:

* The ASAN runtime, so that executables built without `-fsanitize=address`
  (Python, hip_check) can preload it. Without the preload they abort with
  "ASan runtime does not come first in initial library list".
* The symbolizer, so that ASAN reports resolve to function names instead of
  raw hex offsets. This must be absolute: it is read by processes whose working
  directory is not the workspace root (hip_check runs from build/bin), where a
  relative path resolves to nothing.

ROCm reaches the runner two ways and both are supported here. `--artifacts-dir`
selects a build tree that artifacts were extracted into; without it the script
expects ROCm to be installed as Python packages, where `rocm-sdk-core` puts
`amdclang++` on PATH. Only toolchain discovery differs between the two; the
runtime lookup, the exported variables, and the failure behavior are shared.

Which steps consume these values is left to the workflow; this script only
exports them. Leak detection stays on: LSAN_OPTIONS points at a suppressions
file that silences the uninstrumented interpreter without hiding leaks in ROCm
libraries.

Used by `test_component.yml` and `test_rocm_wheels.yml`.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from github_actions_api import gha_set_env

# Applied to every ASAN test job.
#   detect_odr_violation=0      : instrumented ROCm libraries trip ODR checks
#                                 that are not actionable from a test job.
#   quarantine_size_mb=600      : bounds ASAN's quarantine for freed memory.
#   verify_asan_link_order=0    : uninstrumented binaries that load instrumented
#                                 ROCm libraries abort unless the runtime is
#                                 first in the library list. ASAN_RUNTIME_PATH
#                                 below lets callers preload it where they
#                                 control the process, but tests also invoke
#                                 tools they do not spawn directly (amdclang++,
#                                 for example). Disabling the check keeps those
#                                 running; it suppresses the guard rather than
#                                 fixing the ordering, so preload where you can.
STATIC_ASAN_ENV = {
    "ASAN_OPTIONS": (
        "detect_odr_violation=0:quarantine_size_mb=600:verify_asan_link_order=0"
    ),
    "HSA_XNACK": "1",
}

# `-fsanitize=address` emits libclang_rt.asan.so under a per-target runtime
# directory, and libclang_rt.asan-<arch>.so under the layout #8077 reintroduced.
# Which one a toolchain ships depends on how it was configured, so ask for both.
ASAN_RUNTIME_LIBS = (
    "libclang_rt.asan.so",
    f"libclang_rt.asan-{platform.machine()}.so",
)

# Compilers rocm-sdk-core exposes as console scripts, in preference order.
PYTHON_INSTALL_COMPILERS = ("amdclang++", "amdclang")

LSAN_SUPPRESSIONS = Path(__file__).resolve().parent / "lsan_suppressions.txt"


class AsanEnvironmentError(Exception):
    """Raised when a required ASAN path cannot be resolved."""


def _resolve_compiler(artifacts_dir: Optional[Path]) -> Path:
    """Locates the clang that owns the ASAN runtime under test.

    The artifact tree has a known layout; a Python install puts the compiler on
    PATH. Raises if neither turns one up.
    """
    if artifacts_dir is not None:
        clang = artifacts_dir / "llvm" / "bin" / "clang"
        if not os.access(clang, os.X_OK):
            raise AsanEnvironmentError(f"clang not found at {clang}")
        return clang

    for name in PYTHON_INSTALL_COMPILERS:
        found = shutil.which(name)
        if found:
            return Path(found)
    raise AsanEnvironmentError(
        f"none of {', '.join(PYTHON_INSTALL_COMPILERS)} are on PATH; "
        "rocm-sdk-core does not appear to be installed"
    )


def _ask_compiler(clang: Path, flag: str, name: str) -> Optional[Path]:
    """Runs a clang -print-*-name query, returning the path only if it exists.

    clang echoes the name back when it cannot find the file, so the result is
    only meaningful once it resolves to something on disk.
    """
    try:
        result = subprocess.run(
            [str(clang), f"-{flag}={name}"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError) as e:
        raise AsanEnvironmentError(f"could not query {clang} for {name}: {e}") from e

    candidate = Path(result.stdout.strip())
    return candidate.resolve() if candidate.is_file() else None


def _resolve_asan_runtime(clang: Path) -> Path:
    """Asks clang where its ASAN runtime lives. Raises if it ships none."""
    for name in ASAN_RUNTIME_LIBS:
        runtime = _ask_compiler(clang, "print-file-name", name)
        if runtime:
            return runtime
    raise AsanEnvironmentError(
        f"{clang} reports no {' or '.join(ASAN_RUNTIME_LIBS)}; "
        "the build under test is probably not ASAN-instrumented"
    )


def _resolve_symbolizer(
    artifacts_dir: Optional[Path], clang: Path
) -> tuple[Optional[Path], Optional[str]]:
    """Locates llvm-symbolizer beside the compiler.

    Unsymbolized reports are still usable, so this warns instead of raising.
    Returns (path, warning); exactly one is set.
    """
    if artifacts_dir is not None:
        symbolizer = artifacts_dir / "llvm" / "bin" / "llvm-symbolizer"
        if not os.access(symbolizer, os.X_OK):
            return (
                None,
                f"llvm-symbolizer not found at {symbolizer}, "
                "ASAN reports will be unsymbolized",
            )
        return symbolizer.resolve(), None

    symbolizer = _ask_compiler(clang, "print-prog-name", "llvm-symbolizer")
    if not symbolizer:
        return (
            None,
            f"{clang} reports no llvm-symbolizer, ASAN reports will be unsymbolized",
        )
    return symbolizer, None


def _resolve_library_path(artifacts_dir: Path) -> str:
    lib_dir = (artifacts_dir / "lib").resolve()
    parts = [str(lib_dir), str(lib_dir / "rocm_sysdeps" / "lib")]
    existing = os.environ.get("LD_LIBRARY_PATH")
    if existing:
        parts.append(existing)
    return ":".join(parts)


def resolve_asan_env(
    artifacts_dir: Optional[Path] = None,
) -> tuple[dict[str, str], list[str]]:
    """Returns (environment variables to export, warnings to surface).

    Raises AsanEnvironmentError if the runtime cannot be resolved.
    """
    env = dict(STATIC_ASAN_ENV)
    warnings: list[str] = []

    clang = _resolve_compiler(artifacts_dir)
    env["ASAN_RUNTIME_PATH"] = str(_resolve_asan_runtime(clang))

    symbolizer, warning = _resolve_symbolizer(artifacts_dir, clang)
    if symbolizer:
        env["ASAN_SYMBOLIZER_PATH"] = str(symbolizer)
    if warning:
        warnings.append(warning)

    # Python packages carry their own RPATH; only the artifact tree needs help
    # finding its libraries.
    if artifacts_dir is not None:
        env["LD_LIBRARY_PATH"] = _resolve_library_path(artifacts_dir)

    env["LSAN_OPTIONS"] = f"suppressions={LSAN_SUPPRESSIONS}"

    return env, warnings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help=(
            "Directory ROCm artifacts were extracted into "
            "(OUTPUT_ARTIFACTS_DIR). Omit when ROCm is installed as Python "
            "packages."
        ),
    )
    args = parser.parse_args(argv)

    env, warnings = resolve_asan_env(args.artifacts_dir)

    for warning in warnings:
        print(f"::warning::{warning}")
    print(f"Resolved ASAN runtime: {env['ASAN_RUNTIME_PATH']}")
    if "ASAN_SYMBOLIZER_PATH" in env:
        print(f"Resolved ASAN symbolizer: {env['ASAN_SYMBOLIZER_PATH']}")
    if "LD_LIBRARY_PATH" in env:
        print(f"Resolved LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}")

    gha_set_env(env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
