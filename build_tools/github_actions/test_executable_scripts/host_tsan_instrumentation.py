#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared fail-closed enforcement for native host-TSAN tests."""

import os
import re
import subprocess
from pathlib import Path
from typing import Mapping


_CLANG_TSAN_NEEDED_RE = re.compile(
    r"NEEDED.*libclang_rt\.tsan(?:-[^.\s]+)?\.so(?:\.\d+)*"
)
TSAN_OPTIONS = "halt_on_error=1:exitcode=86:history_size=7:second_deadlock_stack=1"


def parse_gtest_listed_tests(output: str) -> list[str]:
    """Return normalized full test names from --gtest_list_tests output."""
    suite = None
    names = []
    for line in output.splitlines():
        uncommented = line.split("#", 1)[0].rstrip()
        if not uncommented.strip():
            continue
        if not line[0].isspace() and uncommented.endswith("."):
            suite = uncommented.strip()
        elif suite is not None and line[0].isspace():
            names.append(f"{suite}{uncommented.strip()}")
    return names


def require_gtest_execution(output: str) -> None:
    """Reject GoogleTest's exit-zero zero-test and skipped-test outcomes."""
    if re.search(r"\[\s*SKIPPED\s*\]", output):
        raise RuntimeError("host-TSAN GoogleTest execution skipped tests")
    if re.search(r"\bRunning 0 tests? from 0 test suites?\b", output):
        raise RuntimeError("host-TSAN GoogleTest execution ran zero tests")


def native_host_tsan_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a deterministic environment which cannot rely on preloading."""
    env = dict(os.environ if source is None else source)
    env.pop("LD_PRELOAD", None)
    options = TSAN_OPTIONS
    symbolizer = env.get("TSAN_SYMBOLIZER_PATH")
    if symbolizer:
        options += f":external_symbolizer_path={symbolizer}"
    env["TSAN_OPTIONS"] = options

    runtime = env.get("TSAN_RUNTIME_PATH")
    if runtime:
        runtime_dir = os.fspath(Path(runtime).parent)
        entries = [
            entry for entry in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if entry
        ]
        if runtime_dir not in entries:
            env["LD_LIBRARY_PATH"] = os.pathsep.join([runtime_dir, *entries])
    return env


def require_no_gpu_nodes() -> None:
    """Refuse to call a CPU-only selector when GPU device nodes are exposed."""
    present = [path for path in (Path("/dev/kfd"), Path("/dev/dri")) if path.exists()]
    if present:
        raise RuntimeError(
            f"host-only runner unexpectedly exposes GPU nodes: {present}"
        )


def require_direct_clang_tsan(
    executable: Path, env: Mapping[str, str] | None = None
) -> None:
    """Require an ELF executable to have a direct Clang-TSAN DT_NEEDED."""
    if not executable.is_file():
        raise RuntimeError(f"required host-TSAN executable is missing: {executable}")

    check_env = native_host_tsan_environment(env)
    bin_dir = check_env.get("THEROCK_BIN_DIR")
    artifact_readelf = (
        Path(bin_dir).resolve().parent / "llvm" / "bin" / "llvm-readelf"
        if bin_dir
        else Path("__missing_artifact_llvm_readelf__")
    )
    readelf = os.fspath(artifact_readelf) if artifact_readelf.is_file() else "readelf"
    result = subprocess.run(
        [readelf, "--dynamic", str(executable)],
        capture_output=True,
        text=True,
        env=check_env,
        check=True,
    )
    if not _CLANG_TSAN_NEEDED_RE.search(result.stdout):
        raise RuntimeError(
            "required executable is not directly linked to Clang TSAN: "
            f"{executable}"
        )
