#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exports the environment that ASAN-instrumented test jobs need to GITHUB_ENV.

Test jobs for `asan` and `host-asan` artifact groups run binaries that link
against instrumented ROCm libraries. Two paths have to be discovered from the
downloaded artifacts rather than hardcoded:

* The ASAN runtime, so that executables built without `-fsanitize=address`
  (Python, hip_check) can preload it. Without the preload they abort with
  "ASan runtime does not come first in initial library list".
* The symbolizer, so that ASAN reports resolve to function names instead of
  raw hex offsets. This must be absolute: it is read by processes whose working
  directory is not the workspace root (hip_check runs from build/bin), where a
  relative path resolves to nothing.

Which steps consume these values is left to the workflow; this script only
exports them. Leak suppression in particular is deliberately not set here: a
job-wide `detect_leaks=0` would disable leak detection for the component tests
as well, so the workflow scopes that to the sanity step's own `env`.

Missing binaries warn rather than fail, preserving the behavior of the inline
shell this replaced.

Used by `test_component.yml`.
"""

import argparse
import os
import platform
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


def _asan_runtime_library_name() -> str:
    """Returns Clang's shared ASAN runtime name for the current host."""
    machine = platform.machine().lower()
    arch = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "arm64": "aarch64",
    }.get(machine, machine)
    return f"libclang_rt.asan-{arch}.so"


def _resolve_asan_runtime(
    artifacts_dir: Path,
) -> tuple[Optional[Path], Optional[str]]:
    """Asks the artifact tree's clang where its ASAN runtime lives.

    Returns (path, warning); exactly one is set.
    """
    clang = artifacts_dir / "llvm" / "bin" / "clang"
    if not os.access(clang, os.X_OK):
        return None, f"clang not found at {clang}, ASAN runtime path not resolved"

    runtime_lib = _asan_runtime_library_name()
    try:
        result = subprocess.run(
            [str(clang), f"-print-file-name={runtime_lib}"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError) as e:
        return None, f"could not query {clang} for the ASAN runtime: {e}"

    runtime = Path(result.stdout.strip())
    if not runtime.is_file():
        return None, f"ASAN runtime not found at {runtime}"
    return runtime.resolve(), None


def _resolve_symbolizer(artifacts_dir: Path) -> tuple[Optional[Path], Optional[str]]:
    """Locates llvm-symbolizer in the artifact tree.

    Returns (path, warning); exactly one is set.
    """
    symbolizer = artifacts_dir / "llvm" / "bin" / "llvm-symbolizer"
    if not os.access(symbolizer, os.X_OK):
        return (
            None,
            f"llvm-symbolizer not found at {symbolizer}, "
            "ASAN reports will be unsymbolized",
        )
    return symbolizer.resolve(), None


def _resolve_library_path(artifacts_dir: Path) -> str:
    lib_dir = (artifacts_dir / "lib").resolve()
    parts = [str(lib_dir), str(lib_dir / "rocm_sysdeps" / "lib")]
    existing = os.environ.get("LD_LIBRARY_PATH")
    if existing:
        parts.append(existing)
    return ":".join(parts)


def resolve_asan_env(artifacts_dir: Path) -> tuple[dict[str, str], list[str]]:
    """Returns (environment variables to export, warnings to surface)."""
    env = dict(STATIC_ASAN_ENV)
    warnings: list[str] = []

    runtime, warning = _resolve_asan_runtime(artifacts_dir)
    if runtime:
        env["ASAN_RUNTIME_PATH"] = str(runtime)
    if warning:
        warnings.append(warning)

    symbolizer, warning = _resolve_symbolizer(artifacts_dir)
    if symbolizer:
        env["ASAN_SYMBOLIZER_PATH"] = str(symbolizer)
    if warning:
        warnings.append(warning)

    env["LD_LIBRARY_PATH"] = _resolve_library_path(artifacts_dir)

    return env, warnings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Directory ROCm artifacts were extracted into (OUTPUT_ARTIFACTS_DIR).",
    )
    args = parser.parse_args(argv)

    env, warnings = resolve_asan_env(args.artifacts_dir)

    for warning in warnings:
        print(f"::warning::{warning}")
    if "ASAN_RUNTIME_PATH" in env:
        print(f"Resolved ASAN runtime: {env['ASAN_RUNTIME_PATH']}")
    if "ASAN_SYMBOLIZER_PATH" in env:
        print(f"Resolved ASAN symbolizer: {env['ASAN_SYMBOLIZER_PATH']}")
    print(f"Resolved LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}")

    gha_set_env(env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
