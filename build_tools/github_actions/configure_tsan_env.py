#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Export the runtime environment for host-TSAN artifact tests.

Host-TSAN executables are required to link the shared compiler-rt runtime
directly. This helper locates that runtime and llvm-symbolizer in the fetched
artifact tree, configures deterministic failure behavior, and exposes the
paths for diagnostics. It deliberately does not set LD_PRELOAD: preloading
TSAN into an uninstrumented Python or shell process weakens the direct-linkage
contract and can introduce reports outside the component under test.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from github_actions_api import gha_set_env

TSAN_RUNTIME_LIB = "libclang_rt.tsan.so"
STATIC_TSAN_OPTIONS = (
    "halt_on_error=1:exitcode=86:history_size=7:second_deadlock_stack=1"
)


def _resolve_tsan_runtime(
    artifacts_dir: Path,
) -> tuple[Optional[Path], Optional[str]]:
    clang = artifacts_dir / "llvm" / "bin" / "clang"
    if not os.access(clang, os.X_OK):
        return None, f"clang not found at {clang}, TSAN runtime path not resolved"

    for runtime_name in (TSAN_RUNTIME_LIB, "libclang_rt.tsan-x86_64.so"):
        try:
            result = subprocess.run(
                [str(clang), f"-print-file-name={runtime_name}"],
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, OSError) as error:
            return None, f"could not query {clang} for the TSAN runtime: {error}"
        runtime = Path(result.stdout.strip())
        if runtime.is_file():
            return runtime.resolve(), None

    return None, "TSAN runtime was not found by the artifact compiler"


def resolve_tsan_env(artifacts_dir: Path) -> tuple[dict[str, str], list[str]]:
    env = {"TSAN_OPTIONS": STATIC_TSAN_OPTIONS}
    warnings: list[str] = []
    runtime_library_dir: Optional[Path] = None

    runtime, warning = _resolve_tsan_runtime(artifacts_dir)
    if runtime:
        env["TSAN_RUNTIME_PATH"] = str(runtime)
        runtime_library_dir = runtime.parent
    if warning:
        warnings.append(warning)

    symbolizer = artifacts_dir / "llvm" / "bin" / "llvm-symbolizer"
    if os.access(symbolizer, os.X_OK):
        symbolizer = symbolizer.resolve()
        env["TSAN_SYMBOLIZER_PATH"] = str(symbolizer)
        env["TSAN_OPTIONS"] += f":external_symbolizer_path={symbolizer}"
    else:
        warnings.append(
            f"llvm-symbolizer not found at {symbolizer}, TSAN reports will be unsymbolized"
        )

    lib_dir = (artifacts_dir / "lib").resolve()
    library_paths = []
    if runtime_library_dir:
        library_paths.append(str(runtime_library_dir))
    library_paths.extend(
        [
            str(lib_dir),
            str(lib_dir / "rocm_sysdeps" / "lib"),
            str(lib_dir / "llvm" / "lib"),
        ]
    )
    existing = os.environ.get("LD_LIBRARY_PATH")
    if existing:
        library_paths.append(existing)
    env["LD_LIBRARY_PATH"] = ":".join(library_paths)
    return env, warnings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    env, warnings = resolve_tsan_env(args.artifacts_dir)
    for warning in warnings:
        print(f"::warning::{warning}")
    if "TSAN_RUNTIME_PATH" not in env:
        print("::error::The artifact compiler did not provide a shared TSAN runtime")
        return 1
    if "TSAN_RUNTIME_PATH" in env:
        print(f"Resolved TSAN runtime: {env['TSAN_RUNTIME_PATH']}")
    if "TSAN_SYMBOLIZER_PATH" in env:
        print(f"Resolved TSAN symbolizer: {env['TSAN_SYMBOLIZER_PATH']}")
    print(f"Resolved LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}")
    gha_set_env(env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
