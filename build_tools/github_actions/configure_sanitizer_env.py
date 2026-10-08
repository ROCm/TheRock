#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configure ASAN or TSAN runtime paths for local development and artifact tests.

Use --output-format shell for shell exports or the default github format to
append to GITHUB_ENV. Preloading is scoped by the caller to processes that load
instrumented libraries. Leak detection remains enabled for component tests.
"""

import argparse
import logging
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from github_actions_api import gha_set_env

# Preserve the ASAN component-test policy: suppress ODR reports, bound the
# freed-memory quarantine, and allow tools that load instrumented libraries
# without preloading. Callers should still preload the runtime when possible.
# The workflow disables leak detection only for its driver sanity probe.
STATIC_SANITIZER_ENV = {
    "asan": {
        "ASAN_OPTIONS": (
            "detect_odr_violation=0:quarantine_size_mb=600:verify_asan_link_order=0"
        ),
        "HSA_XNACK": "1",
    },
    "tsan": {
        "TSAN_OPTIONS": (
            "halt_on_error=1:exitcode=86:history_size=7:second_deadlock_stack=1"
        ),
    },
}
BUILD_VARIANT_SANITIZERS = {
    "asan": "asan",
    "host-asan": "asan",
    "asan-debug": "asan",
    "host-asan-debug": "asan",
    "tsan": "tsan",
}


def _find_tool(artifacts_dir: Path, name: str) -> Path | None:
    # Artifact installs expose llvm/bin; native installs may only expose lib/llvm/bin.
    for directory in (artifacts_dir / "llvm/bin", artifacts_dir / "lib/llvm/bin"):
        tool = directory / name
        if os.access(tool, os.X_OK):
            return tool.resolve()
    logging.warning("%s not found under %s", name, artifacts_dir)
    return None


def _resolve_runtime(artifacts_dir: Path, sanitizer: str) -> Path | None:
    clang = _find_tool(artifacts_dir, "clang")
    if clang is None:
        return None
    # LLVM_ENABLE_PER_TARGET_RUNTIME_DIR controls whether the name has an arch suffix.
    names = (
        f"libclang_rt.{sanitizer}.so",
        f"libclang_rt.{sanitizer}-{platform.machine()}.so",
    )
    for name in names:
        try:
            result = subprocess.run(
                [str(clang), f"-print-file-name={name}"],
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, OSError) as error:
            logging.warning(
                "Could not query %s for the %s runtime: %s", clang, sanitizer, error
            )
            return None
        runtime = Path(result.stdout.strip())
        if runtime.is_file():
            return runtime.resolve()
    logging.warning(
        "%s runtime not found, tried %s", sanitizer.upper(), ", ".join(names)
    )
    return None


def resolve_sanitizer_env(artifacts_dir: Path, build_variant: str) -> dict[str, str]:
    sanitizer = BUILD_VARIANT_SANITIZERS.get(build_variant)
    if sanitizer is None:
        raise ValueError(f"Unsupported sanitizer build variant: {build_variant}")
    prefix = sanitizer.upper()
    env = STATIC_SANITIZER_ENV[sanitizer].copy()
    runtime = _resolve_runtime(artifacts_dir, sanitizer)
    if runtime:
        env[f"{prefix}_RUNTIME_PATH"] = str(runtime)
    symbolizer = _find_tool(artifacts_dir, "llvm-symbolizer")
    if symbolizer:
        env[f"{prefix}_SYMBOLIZER_PATH"] = str(symbolizer)
        if sanitizer == "tsan":
            env["TSAN_OPTIONS"] += f":external_symbolizer_path={symbolizer}"
    else:
        logging.warning("%s reports will be unsymbolized", prefix)

    lib_dir = (artifacts_dir / "lib").resolve()
    library_paths = [str(runtime.parent)] if runtime else []
    library_paths.extend(
        [str(lib_dir), str(lib_dir / "rocm_sysdeps/lib"), str(lib_dir / "llvm/lib")]
    )
    if existing := os.environ.get("LD_LIBRARY_PATH"):
        library_paths.append(existing)
    env["LD_LIBRARY_PATH"] = ":".join(library_paths)
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument(
        "--build-variant", choices=BUILD_VARIANT_SANITIZERS, required=True
    )
    parser.add_argument(
        "--output-format", choices=("github", "shell"), default="github"
    )
    args = parser.parse_args(argv)
    env = resolve_sanitizer_env(args.artifacts_dir, args.build_variant)
    # ASAN keeps its existing warn-only policy for missing runtime artifacts.
    if args.build_variant == "tsan" and "TSAN_RUNTIME_PATH" not in env:
        logging.error("The artifact compiler did not provide a shared TSAN runtime")
        return 1
    if args.output_format == "shell":
        for key, value in env.items():
            print(f"export {key}={shlex.quote(value)}")
    else:
        gha_set_env(env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
