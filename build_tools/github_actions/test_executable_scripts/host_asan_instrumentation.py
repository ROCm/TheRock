#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared enforcement for directly instrumented native host-ASAN tests."""

import os
import re
import subprocess
from pathlib import Path
from typing import Mapping

_CLANG_ASAN_NEEDED_RE = re.compile(
    r"NEEDED.*libclang_rt\.asan(?:-[^.\s]+)?\.so(?:\.\d+)*"
)


def native_host_asan_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an execution environment that cannot rely on ASAN preloading."""
    env = dict(os.environ if source is None else source)
    env.pop("LD_PRELOAD", None)
    return env


def require_direct_clang_asan(
    executable: Path, env: Mapping[str, str] | None = None
) -> None:
    """Require a native executable to have a direct Clang-ASAN DT_NEEDED."""
    if not executable.is_file():
        raise RuntimeError(f"required host-ASAN executable is missing: {executable}")

    check_env = native_host_asan_environment(env)
    result = subprocess.run(
        ["readelf", "-d", str(executable)],
        capture_output=True,
        text=True,
        env=check_env,
        check=True,
    )
    if not _CLANG_ASAN_NEEDED_RE.search(result.stdout):
        raise RuntimeError(
            "required executable is not directly linked to Clang ASAN: " f"{executable}"
        )
