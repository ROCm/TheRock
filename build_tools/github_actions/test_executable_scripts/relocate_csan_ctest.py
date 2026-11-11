# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Relocate rocprofiler-compute's installed CSAN CMake-env test launchers."""

import os
from pathlib import Path
import re
import shutil
from collections.abc import Mapping


_BUILD_CMAKE = "/usr/local/therock-tools/bin/cmake"
_TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|[^\s()]+')
_ADD_TEST = re.compile(r"^\s*add_test\((.*)\)\s*$", re.MULTILINE)


def _quoted(value: str) -> str:
    # Preserve a single CMake argument, including semicolons and backslashes.
    return (
        '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$") + '"'
    )


def relocate_csan_compute_launchers(
    test_dir: Path, env: Mapping[str, str], *, cmake: str | None = None
) -> int:
    """Rebind only generated CMake-env launchers; keep test commands intact.

    The build launcher replaces LD_PRELOAD, losing the artifact HSA runtime and
    report initializer. Require the complete prepared chain before replacing it.
    Other variants do not have CSAN_REPORT_PREFIX and are left byte-identical.
    """
    if not env.get("CSAN_REPORT_PREFIX"):
        return 0
    preload = env.get("LD_PRELOAD", "")
    expected = [
        env.get("CSAN_RUNTIME_PATH", ""),
        env.get("CSAN_HSA_RUNTIME_PATH", ""),
        env.get("CSAN_REPORT_PRELOAD_PATH", ""),
    ]
    if not all(expected) or re.split(r"[:\s]+", preload)[:3] != expected:
        raise ValueError(
            "CSAN CTest relocation requires the prepared runtime:HSA:report preload chain"
        )
    runner_cmake = cmake or shutil.which("cmake", path=env.get("PATH"))
    if not runner_cmake or not os.access(runner_cmake, os.X_OK):
        raise ValueError(
            "No executable runner CMake available for CSAN CTest relocation"
        )

    changed = 0
    for path in sorted(test_dir.rglob("CTestTestfile.cmake")):
        original = path.read_text()

        def replace_launcher(match: re.Match) -> str:
            nonlocal changed
            body = match.group(1)
            tokens = list(_TOKEN.finditer(body))
            values = [token.group().strip('"') for token in tokens]
            # Generated add_test(name executable args...) only; never replace
            # similarly named data arguments, properties, or arbitrary paths.
            if len(values) < 5 or values[2:4] != ["-E", "env"]:
                return match.group()
            executable = values[1]
            if Path(executable).name != "cmake":
                return match.group()
            try:
                end = values.index("--", 4)
            except ValueError:
                return match.group()
            edits = []
            if executable == _BUILD_CMAKE and not Path(executable).is_file():
                edits.append((tokens[1], str(runner_cmake)))
            for index in range(4, end):
                if (
                    values[index].startswith("LD_PRELOAD=")
                    and values[index] != "LD_PRELOAD=" + preload
                ):
                    edits.append((tokens[index], "LD_PRELOAD=" + preload))
            if not edits:
                return match.group()
            for token, replacement in sorted(
                edits, key=lambda edit: edit[0].start(), reverse=True
            ):
                body = (
                    body[: token.start()] + _quoted(replacement) + body[token.end() :]
                )
            changed += 1
            start, end = match.span(1)
            return (
                match.group()[: start - match.start()]
                + body
                + match.group()[end - match.start() :]
            )

        updated = _ADD_TEST.sub(replace_launcher, original)
        if updated != original:
            path.write_text(updated)
    return changed
