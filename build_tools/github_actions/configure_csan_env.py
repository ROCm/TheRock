#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Export the environment needed to test CSAN-instrumented ROCm artifacts.

The runtime must precede HSA in the process library list so its interceptors
initialize GPU reporting. HSA must also be globally visible when OpenCL loads
it as a local dependency. A small preload initializer activates report files
through the sanitizer API because the pinned runtime only parses log_path.
These files preserve GPU findings that successful CTest tests can hide.
"""

import argparse
from collections.abc import Mapping
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from github_actions_api import gha_set_env

REPORT_PRELOAD_SOURCE = Path(__file__).with_name("csan_report_preload.c")
REPORT_PRELOAD_NAME = "libtherock_csan_report_preload.so"


class CsanEnvironmentError(RuntimeError):
    """The artifacts cannot provide a usable CSAN test environment."""


def _artifact_compiler(artifacts_dir: Path) -> Path:
    for path in (
        artifacts_dir / "llvm" / "bin" / "clang",
        artifacts_dir / "lib" / "llvm" / "bin" / "clang",
    ):
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise CsanEnvironmentError(
        f"No artifact compiler found in {artifacts_dir}; include the LLVM "
        "compiler artifacts instead of using a system compiler"
    )


def build_report_preload(artifacts_dir: Path, reports_dir: Path) -> Path:
    """Build a plain C initializer using the compiler shipped by the artifacts."""
    compiler = _artifact_compiler(artifacts_dir.resolve())
    reports_dir = reports_dir.resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    destination = reports_dir / REPORT_PRELOAD_NAME
    # Replacing a completed file atomically avoids rewriting a DSO that an
    # existing process might still have mapped when setup is repeated.
    temporary = tempfile.NamedTemporaryFile(
        prefix=".csan-report-preload-", suffix=".so", dir=reports_dir, delete=False
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    compile_env = os.environ.copy()
    compile_env.pop("LD_PRELOAD", None)
    try:
        subprocess.run(
            [
                str(compiler),
                "-shared",
                "-fPIC",
                "-O2",
                str(REPORT_PRELOAD_SOURCE),
                "-o",
                str(temporary_path),
            ],
            env=compile_env,
            capture_output=True,
            text=True,
            check=True,
        )
        temporary_path.chmod(0o755)
        temporary_path.replace(destination)
    except (OSError, subprocess.CalledProcessError) as exc:
        details = (
            exc.stderr if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        )
        raise CsanEnvironmentError(
            f"Could not build CSAN report initializer with {compiler}: {details}"
        ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


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
    compiler = _artifact_compiler(artifacts_dir)

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

    hsa_runtime = next(
        (
            path.resolve()
            for path in (
                artifacts_dir / "lib" / "libhsa-runtime64.so.1",
                artifacts_dir / "lib" / "libhsa-runtime64.so",
            )
            if path.is_file()
        ),
        None,
    )
    if hsa_runtime is None:
        raise CsanEnvironmentError(
            f"HSA runtime not found in {artifacts_dir / 'lib'}; include the "
            "core runtime artifacts so CSAN can resolve HSA for OpenCL"
        )

    report_preload = reports_dir / REPORT_PRELOAD_NAME
    # The dynamic loader splits LD_PRELOAD on both whitespace and colons;
    # shell quoting cannot protect a library pathname inside its value.
    for description, path in (
        ("CSAN runtime", runtime),
        ("HSA runtime", hsa_runtime),
        ("report initializer", report_preload),
    ):
        if any(char == ":" or char.isspace() for char in str(path)):
            raise CsanEnvironmentError(
                f"The {description} preload path cannot contain whitespace or ':': {path}"
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
    # The shared CSAN offload runtime resolves HSA functions with RTLD_NEXT.
    # OpenCL may dlopen HSA with RTLD_LOCAL, which is invisible to that lookup.
    # Preload the matching artifact HSA globally, after the CSAN interceptors.
    preload = f"{runtime}:{hsa_runtime}:{report_preload}"
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
        "CSAN_HSA_RUNTIME_PATH": str(hsa_runtime),
        "CSAN_REPORT_PREFIX": prefix,
        "CSAN_REPORT_PRELOAD_PATH": str(report_preload),
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
        build_report_preload(args.artifacts_dir, args.reports_dir)
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
