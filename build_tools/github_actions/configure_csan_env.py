#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Export the environment needed to test CSAN-instrumented ROCm artifacts.

The runtime must precede HSA in the process library list so its interceptors
initialize GPU reporting. Per-process report files also preserve GPU findings
that a successful CTest invocation may otherwise hide from its console output.
"""

import argparse
from collections.abc import Mapping
import os
from pathlib import Path
import subprocess
import sys

from github_actions_api import gha_set_env


class CsanEnvironmentError(RuntimeError):
    """The artifacts cannot provide a usable CSAN test environment."""


def _query_compiler(compiler: Path, query: str, name: str) -> Path | None:
    try:
        result = subprocess.run(
            [str(compiler), f"-{query}={name}"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CsanEnvironmentError(
            f"Could not query artifact compiler {compiler} for {name}: {exc}"
        ) from exc
    answer = result.stdout.strip()
    # Clang echoes the requested name when it cannot find the file. Do not
    # mistake an unrelated file with that name in the working directory for it.
    if not answer or answer == name:
        return None
    path = Path(answer)
    return path.resolve() if path.is_file() else None


def resolve_csan_env(
    artifacts_dir: Path,
    reports_dir: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Resolve artifact-owned paths without changing the process environment."""
    existing = dict(os.environ if environ is None else environ)
    artifacts_dir = artifacts_dir.resolve()
    reports_dir = reports_dir.resolve()
    compiler = next(
        (
            path
            for path in (
                artifacts_dir / "llvm" / "bin" / "clang",
                artifacts_dir / "lib" / "llvm" / "bin" / "clang",
            )
            if path.is_file() and os.access(path, os.X_OK)
        ),
        None,
    )
    if compiler is None:
        raise CsanEnvironmentError(
            f"No artifact compiler found in {artifacts_dir}; include the LLVM "
            "compiler artifacts instead of using a system compiler"
        )

    runtime = None
    for name in ("libclang_rt.csan.so", "libclang_rt.csan-x86_64.so"):
        runtime = _query_compiler(compiler, "print-file-name", name)
        if runtime is not None:
            break
    if runtime is None:
        raise CsanEnvironmentError(
            f"CSAN runtime not found through {compiler}; verify that the "
            "selected build includes its CSAN compiler runtime artifacts"
        )

    prefix = str(reports_dir / "csan")
    if any(char in prefix for char in ('"', "\r", "\n")):
        raise CsanEnvironmentError(
            "The CSAN report path cannot contain double quotes or newlines"
        )
    options = ":".join(
        filter(
            None,
            (
                existing.get("CSAN_OPTIONS", ""),
                # Later options win. Keep unrelated caller options, but ensure
                # findings remain fatal and reports reach the directory scanned
                # by check_csan_reports.py.
                "halt_on_error=1",
                "color=never",
                f'log_path="{prefix}"',
            ),
        )
    )
    preload = str(runtime)
    if existing.get("LD_PRELOAD"):
        preload += ":" + existing["LD_PRELOAD"]
    library_paths = [
        str(artifacts_dir / "lib"),
        str(artifacts_dir / "lib" / "rocm_sysdeps" / "lib"),
        str(runtime.parent),
    ]
    if existing.get("LD_LIBRARY_PATH"):
        library_paths.append(existing["LD_LIBRARY_PATH"])
    env = {
        "CSAN_RUNTIME_PATH": str(runtime),
        "LD_PRELOAD": preload,
        "LD_LIBRARY_PATH": ":".join(library_paths),
        "CSAN_OPTIONS": options,
    }
    warnings = []
    symbolizer = _query_compiler(compiler, "print-prog-name", "llvm-symbolizer")
    if symbolizer is not None:
        env["CSAN_SYMBOLIZER_PATH"] = str(symbolizer)
    else:
        warnings.append(
            f"{compiler} cannot locate llvm-symbolizer; CSAN reports will be unsymbolized"
        )
    if any("\n" in value or "\r" in value for value in env.values()):
        raise CsanEnvironmentError("CSAN environment values cannot contain newlines")
    return env, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        env, warnings = resolve_csan_env(args.artifacts_dir, args.reports_dir)
        args.reports_dir.mkdir(parents=True, exist_ok=True)
    except (CsanEnvironmentError, OSError) as exc:
        print(f"CSAN environment setup failed: {exc}", file=sys.stderr)
        return 1
    for warning in warnings:
        print(f"::warning::{warning}")
    print(f"Resolved CSAN runtime: {env['CSAN_RUNTIME_PATH']}")
    print(f"CSAN reports: {args.reports_dir.resolve()}")
    gha_set_env(env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
